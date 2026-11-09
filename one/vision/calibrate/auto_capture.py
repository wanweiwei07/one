"""Move a robot around the ChArUco board and record EVERY camera on it at
each pose: the images for each camera's lens fit and, with the joint
angles, for its hand-eye -- one run, no lens calibration needed first.

The cameras are only known roughly: a nominal K per camera (e.g. focal
length / pixel pitch, centre of the picture; fisheye model, no distortion)
and, optionally, a rough flange -> camera tf. Two phases:

1. bootstrap -- small turns of J4 / J5 / J6 around the start pose (as
   ``handeye_capture``); the nominal K gives each camera a rough
   flange -> camera tf and the board's place on the table. A camera that
   saw too little takes its rough tf and another camera's board.
2. coverage -- for each camera, views that look at the board: its centre
   at each of TARGETS, near the middle of the picture, from different
   tilts, distances and rolls -- tilted, the wide-angle picture puts the
   board's far side out to the edges, where the lens distorts most. Kept
   only with an IK solution, collision free (``build``'s planning
   context) and most of the board predicted in the picture.

Each phase is planned before it moves; the plan is printed and 'g' starts
it. q / Esc stops before the next pose. At the end the arm goes back to
the start pose. Each camera's images and ``poses.json`` go to its own
folder (cleared at the start), the layout ``lens.fit`` and
``handeye.calibrate`` read.
"""

import os
import glob
import json
import time

import cv2
import numpy as np

import one.utils.math as oum
from one.drivers.cobotta import BCapError

from one.utils.keys import _wait_for_g
from one.vision.calibrate.charuco import (NROW, NCOLUMN, SQUARE_SIZE,
                                          MIN_CORNERS, charuco_board)
from one.vision.calibrate import handeye as che
from one.vision.calibrate import handeye_capture as chc


# where the board centre is put in the picture (share of width, height):
# near the middle, the camera looks AT the board (default; a camera with
# part of its picture blocked passes its own)
TARGETS = ((0.5, 0.5), (0.38, 0.4), (0.62, 0.4), (0.38, 0.6), (0.62, 0.6))
N_PER_TARGET = 4                # views per target, from different sides
TILTS_DEG = (0, 25, 40)         # camera -> board centre line from vertical
AZIMUTHS_DEG = (0, 60, 120, 180, 240, 300)
DISTANCES = (0.18, 0.24, 0.30)  # m, camera to board centre (default; pass
                                # the cameras' working range)
ROLLS_DEG = (0, 30, -30)
IN_VIEW = 0.8                   # share of the corners predicted in picture
MIN_HEIGHT = 0.07               # m, camera above the board at least
TIP_CLEARANCE = 0.04            # m, the tool tip above the board at least
COVERAGE_GRID = 4               # coverage printed on a grid this fine


class _Group:
    """The cameras as one ``camera`` for ``handeye_capture.Session``."""

    def __init__(self, cameras):
        self.cameras = cameras

    def close(self):
        for cam in self.cameras.values():
            cam.close()


class Session(chc.Session):
    """The arm and its cameras for one run.

    ``cameras``: ``{name: camera}``, open, each with ``frame()`` and
    ``close()`` (closed by :meth:`close`). ``detectors``: ``{name:
    BoardDetector}`` with the NOMINAL K -- planning only. ``data_dirs``:
    ``{name: folder}``. ``priors``: ``{name: rough flange -> camera tf}``,
    for a camera the bootstrap could not place (optional). ``distances``
    (m, camera to board centre): the cameras' working range. ``tool_tip``:
    ``robot -> world point`` of what sticks out furthest (e.g. a tool's
    tcp) after ``robot.fk``; views bringing it nearer the board than
    TIP_CLEARANCE are left out -- the collision model only keeps it off
    the ground, not off the board and its backing. ``targets``: where the
    board centre goes in the picture (default TARGETS). ``blocked_top``:
    the share of the picture's height at the top that the tool hides;
    corners there do not count as seen.
    """

    def __init__(self, build, host, cameras, detectors, data_dirs,
                 priors=None, distances=DISTANCES, tool_tip=None,
                 targets=TARGETS, blocked_top=0.0):
        super().__init__(build, host, _Group(cameras), None, None,
                         distances=distances)
        self.cameras = cameras
        self.detectors = detectors
        self.data_dirs = data_dirs
        self.priors = priors or {}
        self.tool_tip = tool_tip
        self.targets = tuple(targets)
        self.blocked_top = blocked_top
        self.records = {name: [] for name in cameras}
        self.img_sizes = {}

    # -- recording --------------------------------------------------------
    def record(self, label):
        """Every camera's image + the measured joint angles at the current
        pose; ``{name: n corners}``."""
        time.sleep(chc.SETTLE)
        images = {name: cam.frame() for name, cam in self.cameras.items()}
        qs = [float(q) for q in
              self.cobotta.joint_angles(ndof=self.robot.ndof)]
        found = {}
        for name, image in images.items():
            self.img_sizes[name] = image.shape[1::-1]
            img_name = f'{len(self.records[name]):03d}.png'
            cv2.imwrite(os.path.join(self.data_dirs[name], img_name), image)
            self.records[name].append({'image': img_name, 'qs': qs,
                                       'label': label})
            found[name] = self.detectors[name].corners(image)[1]
        print(f'  {label}: corners '
              + ', '.join(f'{name} {n}' for name, n in found.items()))
        return found

    def save(self):
        for name, recs in self.records.items():
            with open(os.path.join(self.data_dirs[name], che.RECORD_NAME),
                      'w', encoding='utf-8') as f:
                json.dump(recs, f, indent=2)

    def poses_seen(self, name):
        """``(flange tfs, detected corners)`` of ``name``'s usable images."""
        qs_list, observations = [], []
        for rec in self.records[name]:
            image = cv2.imread(os.path.join(self.data_dirs[name],
                                            rec['image']))
            found, _ = self.detectors[name].corners(image)
            if found is not None:
                qs_list.append(rec['qs'])
                observations.append(found)
        return che.flange_tfs(self.robot, qs_list), observations

    # -- planning ---------------------------------------------------------
    def rough_poses(self):
        """``{name: (flange -> camera tf, board in world)}`` from the
        bootstrap images (nominal K), priors for the rest; None when no
        camera saw the board well enough."""
        rough = {}
        for name in self.cameras:
            flanges, observations = self.poses_seen(name)
            if len(observations) < 4:
                print(f'{name}: only {len(observations)} usable bootstrap '
                      'images')
                continue
            print(f'{name}: rough hand-eye (nominal K):')
            tf_flange_cam, tf_w_board, per_image = che.solve(
                self.detectors[name], flanges, observations)
            print(f'  flange -> camera '
                  f'{np.round(tf_flange_cam[:3, 3] * 100, 1)} cm, rms '
                  f'{np.sqrt(np.mean(np.square(per_image))):.1f} px')
            if name in self.priors:
                off = tf_flange_cam[:3, 3] - self.priors[name][:3, 3]
                print(f'  {np.linalg.norm(off) * 100:.1f} cm from the prior')
            rough[name] = (tf_flange_cam, tf_w_board)
        if not rough:
            return None
        tf_w_board = next(iter(rough.values()))[1]
        for name in self.cameras:
            if name not in rough and name in self.priors:
                print(f"{name}: its prior and another camera's board")
                rough[name] = (self.priors[name], tf_w_board)
        return rough

    def plan_views(self, start, name, tf_flange_cam, tf_w_board):
        """``[(label, qs)]``: up to N_PER_TARGET views per target of
        ``name``, each target from other sides."""
        w, h = self.img_sizes[name]
        center = (tf_w_board @ np.array([NCOLUMN * SQUARE_SIZE / 2,
                                         NROW * SQUARE_SIZE / 2, 0, 1]))[:3]
        self.robot.fk(qs=start)
        x_ref = (np.array(self.robot.tcp('flange').tf, dtype=np.float64)
                 @ tf_flange_cam)[:3, 0]
        tf_cam_flange = np.linalg.inv(tf_flange_cam)
        sides = [(t, a) for t in TILTS_DEG
                 for a in (AZIMUTHS_DEG if t else (0,))]
        dists = self.distances
        n_try = len(dists) * len(ROLLS_DEG)
        views = []
        for k, (tu, tv) in enumerate(self.targets):
            # the board centre's ray in the camera, by the nominal K
            xn, yn = self.detectors[name].normalize(
                np.array([[tu * w, tv * h]])).ravel()
            ray = np.array([xn, yn, 1.0]) / np.linalg.norm([xn, yn, 1.0])
            got = 0
            # each target starts at another side, distance and roll
            for i, (tilt, azim) in enumerate(sides[k:] + sides[:k]):
                if got >= N_PER_TARGET:
                    break
                up = chc._rot((0, 0, 1), azim) @ chc._rot((0, 1, 0), tilt) \
                    @ np.array([0.0, 0.0, 1.0])
                for j in range(n_try):
                    dist = dists[(i + j) % len(dists)]
                    roll = ROLLS_DEG[(i + j // len(dists)) % len(ROLLS_DEG)]
                    tf_w_cam = self._camera_pose(center, up, dist, ray,
                                                 x_ref, roll)
                    if tf_w_cam is None or not self._sees(
                            name, np.linalg.inv(tf_w_cam) @ tf_w_board):
                        continue
                    qs = self._reachable(tf_w_cam @ tf_cam_flange, start)
                    if qs is None or not self._tip_clear(qs, center[2]):
                        continue
                    views.append((f'{name}: board at ({tu:.2f}, {tv:.2f}) '
                                  f'tilt {tilt} azim {azim} dist '
                                  f'{dist * 100:.0f}cm roll {roll:+d}', qs))
                    got += 1
                    break
            if got < N_PER_TARGET:
                print(f'{name}: board at ({tu:.2f}, {tv:.2f}): only {got} '
                      'reachable, collision-free views')
        return views

    @staticmethod
    def _camera_pose(center, up, dist, ray, x_ref, roll):
        """The camera tf at ``center + dist * up`` with the board centre
        on ``ray`` (camera frame), x as near ``x_ref`` as that allows, then
        turned ``roll`` deg about the line of sight; None when too low."""
        pos = center + dist * up
        if pos[2] - center[2] < MIN_HEIGHT:
            return None
        to_center = -up
        rot = np.asarray(oum.rotmat_between_vecs(ray, to_center),
                         dtype=np.float64)
        x0 = rot[:, 0] - (rot[:, 0] @ to_center) * to_center
        xr = x_ref - (x_ref @ to_center) * to_center
        phi = np.degrees(np.arctan2(to_center @ np.cross(x0, xr), x0 @ xr))
        rot = chc._rot(to_center, phi + roll) @ rot
        return che._tf(pos, rot)

    def _tip_clear(self, qs, board_z):
        """The tool tip at ``qs`` at least TIP_CLEARANCE above the board."""
        if self.tool_tip is None:
            return True
        self.robot.fk(qs=qs)
        return float(self.tool_tip(self.robot)[2]) - board_z >= TIP_CLEARANCE

    def _sees(self, name, tf_cam_board):
        """Enough of the board in ``name``'s picture (nominal K)."""
        obj = self.detectors[name].board.getChessboardCorners()
        cam = (tf_cam_board[:3, :3] @ obj.T).T + tf_cam_board[:3, 3]
        front = cam[:, 2] > 0.03
        if front.mean() < IN_VIEW:
            return False
        px = self.detectors[name].project(tf_cam_board, obj[front])
        w, h = self.img_sizes[name]
        m = chc.IMAGE_MARGIN
        top = max(m, self.blocked_top * h)
        inside = ((px[:, 0] > m) & (px[:, 0] < w - m)
                  & (px[:, 1] > top) & (px[:, 1] < h - m))
        return inside.sum() >= IN_VIEW * len(obj)

    # -- the run ----------------------------------------------------------
    def run(self):
        start = self.qs_now()
        print(f'start: {np.round(np.degrees(start), 2)} deg')
        if not self.pln_ctx.is_state_valid(start):
            print('start: in collision in the model -- refusing to move')
            return False
        if max(self.record('start').values()) < MIN_CORNERS:
            print('no camera sees the board well from the start pose -- '
                  'point them at it first')
            return False
        boot = self.bootstrap_goals(start)
        print(f'\nphase 1: {len(boot)} bootstrap turns around the start pose '
              f'at {chc.SPEED:.0f}%:')
        for label, _ in boot:
            print(f'  {label}')
        if not _wait_for_g('bootstrap'):
            return False
        self.cobotta.enable()
        try:
            if not self.visit(boot):
                return False
            if not self.go(start):
                print('cannot get back to the start pose')
                return False
            rough = self.rough_poses()
            if rough is None:
                print('no camera could be placed -- stopping')
                return False
            views = []
            for name, (tf_flange_cam, tf_w_board) in rough.items():
                views += self.plan_views(start, name, tf_flange_cam,
                                         tf_w_board)
            # nearest-first, so the arm does not cross back and forth
            ordered, cur = [], start
            while views:
                i = int(np.argmin([np.max(np.abs(q - cur))
                                   for _, q in views]))
                ordered.append(views.pop(i))
                cur = ordered[-1][1]
            print(f'\nphase 2: {len(ordered)} views (every camera records at '
                  'each):')
            for label, qs in ordered:
                print(f'  {label}: {np.round(np.degrees(qs), 1)} deg')
            if not ordered or not _wait_for_g('views'):
                return False
            self.visit(ordered)
            print('back to the start pose')
            self.go(start)
        finally:
            self.save()
        return True


def coverage(data_dir, grid=COVERAGE_GRID):
    """Print, per cell of a ``grid`` x ``grid`` division of the picture,
    how many images have board corners there (the lens fit wants every
    cell, the corners most); the number of usable images."""
    detector = cv2.aruco.CharucoDetector(charuco_board())
    counts = np.zeros((grid, grid), dtype=int)
    n_used = 0
    for path in sorted(glob.glob(os.path.join(data_dir, '*.png'))):
        image = cv2.imread(path)
        corners, ids, _, _ = detector.detectBoard(image)
        if ids is None or len(ids) < MIN_CORNERS:
            continue
        n_used += 1
        h, w = image.shape[:2]
        px = corners.reshape(-1, 2)
        rows = np.clip((px[:, 1] * grid / h).astype(int), 0, grid - 1)
        cols = np.clip((px[:, 0] * grid / w).astype(int), 0, grid - 1)
        for r, c in set(zip(rows, cols)):
            counts[r, c] += 1
    print(f'{data_dir}: {n_used} usable images; images with corners in '
          'each cell of the picture:')
    for row in counts:
        print('  ' + ' '.join(f'{v:3d}' for v in row))
    return n_used


def run(make_session, data_dirs, calibrate):
    """Clear the ``data_dirs`` (``{name: folder}``), record with
    ``make_session()`` (a :class:`Session` recording into them), print
    each camera's coverage, then ``calibrate()``."""
    for data_dir in data_dirs.values():
        os.makedirs(data_dir, exist_ok=True)
        for path in glob.glob(os.path.join(data_dir, '*.png')) + glob.glob(
                os.path.join(data_dir, che.RECORD_NAME)):
            os.remove(path)
    try:
        session = make_session()
    except (OSError, BCapError, RuntimeError) as exc:
        print(exc)
        return False
    try:
        if not session.run():
            return False
    except (OSError, BCapError) as exc:
        print(f'arm: {exc} -- stopping')
        return False
    finally:
        session.close()
    print()
    for data_dir in data_dirs.values():
        coverage(data_dir)
    print()
    return calibrate()
