"""Move a robot through ~20 poses that look at the ChArUco board on the
table, record an image and the joint angles at each, then solve the
hand-eye calibration (``handeye``). The arm is a COBOTTA (b-CAP,
``one.drivers.cobotta``).

Start with the camera seeing the whole board (the board lies flat on the
table, at about the robot base's height). Two phases:

1. bootstrap -- small turns of J4 / J5 / J6 alone around the start pose
   (the board stays in view); they give a rough flange -> camera transform.
2. views -- with that, camera poses on a cap over the board centre: tilted
   0-35 deg from vertical in six directions, two distances, a little roll.
   Kept only when there is an IK solution, the arm is collision free (in
   the planning context of ``build``) and the whole board is
   predicted to be in the picture.

Everything of a phase is planned before it moves; the plan is printed and
'g' starts it. q / Esc stops before the next pose. At the end the arm goes
back to the start pose. The images and joint angles go to
``data_dir`` (cleared at the start of each run).
"""

import os
import glob
import json
import time

import cv2
import numpy as np

import one.motion.probabilistic.rrt as ompr
from one.drivers.cobotta import BCapError, CobottaMotion

from one.utils.keys import _quit_pressed, _wait_for_g
from one.vision.calibrate.cameras import index_of
from one.vision.calibrate.charuco import NROW, NCOLUMN, SQUARE_SIZE
from one.vision.calibrate import handeye as che


SPEED = 30.0
SETTLE = 0.5                    # s after arriving, before the image
FLUSH_FRAMES = 10               # stale frames the camera has buffered
# bootstrap: (joint index, deg) turns from the start pose
BOOTSTRAP = ((3, 10), (3, -10), (4, 10), (4, -10), (5, 15), (5, -15))
TILTS_DEG = (20, 35, 0)         # camera axis from vertical
AZIMUTHS_DEG = (0, 60, 120, 180, 240, 300)
DISTANCES = (0.20, 0.25)        # m, camera to board centre (default; a
                                # camera focused nearer passes its own)
ROLLS_DEG = (0, 20, -20)
N_VIEWS = 14
IMAGE_MARGIN = 15               # px, board corners kept this far inside
IN_VIEW = 1.0                   # share of the corners that must be in it
_EXTEND_STEP_SIZE = np.pi / 36
_MAX_ITERS = 3000
_SHORTCUT_ITERS = 200


def _rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, dtype=np.float64)
                         * np.deg2rad(deg))[0]


class WebCam:
    """The webcam on hub ``port`` (or port path, ``cameras.index_of``),
    kept open for the whole run."""

    def __init__(self, port, label='camera'):
        self.label = label
        try:
            index = index_of(port)
        except LookupError as exc:
            raise RuntimeError(f'{label}: {exc}')
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f'{label}: cannot open')

    def frame(self):
        """A frame taken now, not one from the buffer."""
        for _ in range(FLUSH_FRAMES):
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError(f'{self.label}: gives no image')
        return frame

    def close(self):
        self.cap.release()


class Session:
    """The arm, the camera and the board detector of a hand-eye run.

    ``build``: ``() -> (ground, collider, robot, pln_ctx)`` of the arm the
    camera is on (e.g. ``coripps.tool.home_sampler._build``), ``host`` its controller.
    ``camera``: open, with ``frame()`` (BGR, taken now) and ``close()``;
    closed by :meth:`close`, also when connecting the arm fails.
    ``distances`` (m, camera to board centre) of the views; ``in_view``: the
    share of the board's corners a view must have in the picture (below 1
    when the board is too big for the picture at those distances).
    """

    def __init__(self, build, host, camera, detector, data_dir,
                 distances=DISTANCES, in_view=IN_VIEW):
        self.camera = camera
        try:
            _, _, self.robot, self.pln_ctx = build()
            self.cobotta = CobottaMotion(host, speed=SPEED).connect()
        except BaseException:
            camera.close()
            raise
        self.detector = detector
        self.data_dir = data_dir
        self.distances = tuple(distances)
        self.in_view = in_view
        self.records = []
        self.img_size = None

    def close(self):
        self.camera.close()
        self.cobotta.disconnect()       # powers off, gives the arm back

    # -- moving -----------------------------------------------------------
    def qs_now(self):
        return self.pln_ctx.enforce_bounds(
            self.cobotta.joint_angles(ndof=self.robot.ndof))

    def path_to(self, start, goal):
        """A collision-free joint path, or None."""
        if self.pln_ctx.is_motion_valid(start, goal):
            return [start, goal]
        planner = ompr.RRTConnectPlanner(pln_ctx=self.pln_ctx,
                                         extend_step_size=_EXTEND_STEP_SIZE)
        path = planner.solve(start=start, goal=goal, max_iters=_MAX_ITERS)
        if not path:
            return None
        return ompr.shortcut_path(path, self.pln_ctx, n_iter=_SHORTCUT_ITERS)

    def go(self, goal):
        """Move to ``goal``; False when no path."""
        path = self.path_to(self.qs_now(), goal)
        if path is None:
            print('  no collision-free path -- skipped')
            return False
        self.cobotta.move_path(path)
        return True

    # -- recording --------------------------------------------------------
    def record(self, label):
        """Image + measured joint angles at the current pose; the board's
        camera tf, or None when it is not seen well enough."""
        time.sleep(SETTLE)
        image = self.camera.frame()
        self.img_size = image.shape[1::-1]
        qs = self.cobotta.joint_angles(ndof=self.robot.ndof)
        tf, n = self.detector.pose(image)
        name = f'{len(self.records):03d}.png'
        cv2.imwrite(os.path.join(self.data_dir, name), image)
        self.records.append({'image': name, 'qs': [float(q) for q in qs],
                             'label': label})
        print(f'  {name} {label}: {n} corners'
              + ('' if tf is not None else ' -- too few, not usable'))
        return tf

    def save(self):
        with open(os.path.join(self.data_dir, che.RECORD_NAME), 'w',
                  encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)

    # -- planning ---------------------------------------------------------
    def bootstrap_goals(self, start):
        goals = []
        for jnt, deg in BOOTSTRAP:
            goal = start.copy()
            goal[jnt] += np.deg2rad(deg)
            if self.pln_ctx.is_motion_valid(start, goal):
                goals.append((f'J{jnt + 1} {deg:+d} deg', goal))
            else:
                print(f'bootstrap J{jnt + 1} {deg:+d} deg: out of limits or '
                      'in collision -- left out')
        return goals

    def plan_views(self, start, tf_flange_cam, tf_w_board):
        """Up to N_VIEWS ``(label, qs)`` around the board, nearest-first
        from ``start``."""
        center_board = np.array([NCOLUMN * SQUARE_SIZE / 2,
                                 NROW * SQUARE_SIZE / 2, 0.0, 1.0])
        center = (tf_w_board @ center_board)[:3]
        self.robot.fk(qs=start)
        tf_w_cam0 = (np.array(self.robot.tcp('flange').tf, dtype=np.float64)
                     @ tf_flange_cam)
        x_ref = tf_w_cam0[:3, 0]
        tf_cam_flange = np.linalg.inv(tf_flange_cam)
        views = []
        # one view per (tilt, direction): the first distance / roll that
        # works, the preferred distance and roll changing from view to view
        for tilt in TILTS_DEG:
            for azim in (AZIMUTHS_DEG if tilt else (0,)):
                if len(views) >= N_VIEWS:
                    break
                up = _rot((0, 0, 1), azim) @ _rot((0, 1, 0), tilt) \
                    @ np.array([0.0, 0.0, 1.0])
                z = -up
                x = x_ref - (x_ref @ z) * z
                x /= np.linalg.norm(x)
                k_d = len(views) % len(self.distances)
                k_r = len(views) % len(ROLLS_DEG)
                view = None
                for dist in self.distances[k_d:] + self.distances[:k_d]:
                    for roll in ROLLS_DEG[k_r:] + ROLLS_DEG[:k_r]:
                        rx = _rot(z, roll) @ x
                        tf_w_cam = che._tf(center + dist * up, np.column_stack(
                            [rx, np.cross(z, rx), z]))
                        qs = self._reachable(tf_w_cam @ tf_cam_flange, start)
                        if qs is not None and self._sees_board(
                                np.linalg.inv(tf_w_cam) @ tf_w_board):
                            view = (f'tilt {tilt} azim {azim} dist '
                                    f'{dist * 100:.0f}cm roll {roll:+d}', qs)
                            break
                    if view is not None:
                        break
                if view is None:
                    print(f'view tilt {tilt} azim {azim}: no reachable, '
                          'collision-free pose that sees the board')
                else:
                    views.append(view)
        # visit nearest-first so the arm does not cross back and forth
        ordered, cur = [], start
        while views:
            i = int(np.argmin([np.max(np.abs(q - cur)) for _, q in views]))
            ordered.append(views.pop(i))
            cur = ordered[-1][1]
        return ordered

    def _reachable(self, tf_w_flange, near):
        """Collision-free joints for a flange pose, nearest ``near``."""
        sols = self.robot.ik(tf_w_flange[:3, 3].astype(np.float32),
                             tf_w_flange[:3, :3].astype(np.float32),
                             ref_qs=near)
        best = None
        for qs in sols:
            qs = np.asarray(qs, dtype=np.float32)
            if not self.pln_ctx.is_state_valid(qs):
                continue
            if best is None or (np.max(np.abs(qs - near))
                                < np.max(np.abs(best - near))):
                best = qs
        return best

    def _sees_board(self, tf_cam_board):
        if tf_cam_board[2, 3] <= 0:     # behind the camera
            return False
        px = self.detector.project(tf_cam_board)
        w, h = self.img_size
        m = IMAGE_MARGIN
        inside = ((px[:, 0] > m) & (px[:, 0] < w - m)
                  & (px[:, 1] > m) & (px[:, 1] < h - m))
        return bool(inside.mean() >= self.in_view)

    # -- the run ----------------------------------------------------------
    def poses_seen(self):
        """``(flange tfs, detected board corners)`` of the usable
        records."""
        qs_list, observations = [], []
        for rec in self.records:
            image = cv2.imread(os.path.join(self.data_dir, rec['image']))
            found, _ = self.detector.corners(image)
            if found is not None:
                qs_list.append(rec['qs'])
                observations.append(found)
        return che.flange_tfs(self.robot, qs_list), observations

    def visit(self, goals):
        """Go to each ``(label, qs)`` and record; False when stopped."""
        for label, goal in goals:
            if _quit_pressed():
                print('stopped')
                return False
            print(label)
            if self.go(goal):
                self.record(label)
                self.save()
        return True

    def run(self):
        start = self.qs_now()
        print(f'start: {np.round(np.degrees(start), 2)} deg')
        if not self.pln_ctx.is_state_valid(start):
            print('start: in collision in the model -- refusing to move')
            return False
        if self.record('start') is None:
            print('the board is not seen well from the start pose -- point '
                  'the camera at it first')
            return False
        boot = self.bootstrap_goals(start)
        print(f'\nphase 1: {len(boot)} bootstrap turns around the start pose '
              f'at {SPEED:.0f}%:')
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
            flanges, observations = self.poses_seen()
            if len(observations) < 4:
                print(f'only {len(observations)} usable bootstrap images -- '
                      'stopping')
                return False
            print('\nrough hand-eye from the bootstrap:')
            tf_flange_cam, tf_w_board, per_image = che.solve(
                self.detector, flanges, observations)
            che.report(tf_flange_cam, tf_w_board, per_image)
            views = self.plan_views(start, tf_flange_cam, tf_w_board)
            print(f'\nphase 2: {len(views)} views:')
            for label, qs in views:
                print(f'  {label}: {np.round(np.degrees(qs), 1)} deg')
            if not views or not _wait_for_g('views'):
                return False
            self.visit(views)
            print('back to the start pose')
            self.go(start)
        finally:
            self.save()
        return True


def run(make_session, data_dir, calibrate):
    """Clear ``data_dir``, record with ``make_session()`` (a :class:`Session`
    recording into ``data_dir``), then solve with ``calibrate()``."""
    os.makedirs(data_dir, exist_ok=True)
    for path in glob.glob(os.path.join(data_dir, '*.png')):
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
    print('\nfinal hand-eye calibration:')
    return calibrate()
