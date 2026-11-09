"""Stereo calibration of two webcams from ChArUco images taken at the same
time (same file name in two folders, e.g. by :func:`capture`): the pose
of camera b in camera a, straight from the board seen by both -- no robot
in the loop, unlike two hand-eyes put together.

:func:`capture`: both pictures live side by side with the corners drawn;
space saves each picture that sees enough corners (both under one name),
q quits -- one pass serves both lens fits (each folder on its own,
``lens.fit``) and the stereo fit (the names saved in both). The two
cameras are not synchronised -- hold the board (or the cameras) still for
the space -- it saves only once the corners have stood still (STILL_PX)
for STILL_FRAMES frames in both. Reach every part of each picture, corners and edges most, at
different tilts and distances, and 15-20 of the views over the part the
two share. 25-35 in all.

Each lens is fixed at its own fit (``lens.fit``); the pose between the
two and the board's pose in camera a at each pair are fitted to the
corners both images found, reprojected into both (robust least squares,
as ``handeye.solve``). Pairs with an image a lens fit dropped are left
out, and so are, after a first solve, the pairs fitting much worse than
the rest (the board moved between the two pictures) -- then solved
again. Not ``cv2.fisheye.stereoCalibrate``: in OpenCV
4.12 it fails an output-size assertion whatever it is given.
"""

import os
import glob
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from one.vision.calibrate import lens
from one.vision.calibrate import handeye as che
from one.vision.calibrate.cameras import index_of
from one.vision.calibrate.charuco import (MIN_CORNERS, charuco_board,
                                          _delete_old, _next_number)


WINDOW = 'stereo pairs'
# saved only when every corner seen in both pictures moved less than
# STILL_PX (median) frame to frame for STILL_FRAMES frames in a row: the
# two cameras' exposures are up to a frame apart, a moving board puts a
# different board pose in each
STILL_PX = 0.3
STILL_FRAMES = 3


def _motion(last, now):
    """Median px the corners found in both ``{id: px}`` moved; inf when
    none."""
    ids = set(last) & set(now)
    if not ids:
        return np.inf
    return float(np.median([np.linalg.norm(now[i] - last[i]) for i in ids]))


def _detect(path, detector):
    """``{corner id: pixel}`` of one image."""
    corners, ids, _, _ = detector.detectBoard(cv2.imread(path))
    if ids is None:
        return {}
    return dict(zip(ids.ravel().tolist(), corners.reshape(-1, 2)))


def common_corners(dir_a, dir_b):
    """``(names, object points, pixels in a, pixels in b, image size)`` of
    the image pairs (same name in both folders) with at least MIN_CORNERS
    corners found in BOTH."""
    board = charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    obj_all = board.getChessboardCorners()
    names, obj_pts, img_a, img_b, size = [], [], [], [], None
    for path_a in sorted(glob.glob(os.path.join(dir_a, '*.png'))):
        name = os.path.basename(path_a)
        path_b = os.path.join(dir_b, name)
        if not os.path.exists(path_b):
            continue
        size = cv2.imread(path_a).shape[1::-1]
        found_a, found_b = _detect(path_a, detector), _detect(path_b, detector)
        ids = sorted(set(found_a) & set(found_b))
        if len(ids) < MIN_CORNERS:
            print(f'{name}: only {len(ids)} corners in both, skipped')
            continue
        names.append(name)
        obj_pts.append(obj_all[ids].reshape(-1, 1, 3).astype(np.float64))
        img_a.append(np.array([found_a[i] for i in ids],
                              dtype=np.float64).reshape(-1, 1, 2))
        img_b.append(np.array([found_b[i] for i in ids],
                              dtype=np.float64).reshape(-1, 1, 2))
    return names, obj_pts, img_a, img_b, size


def capture(port_a, port_b, dir_a, dir_b, labels=('a', 'b')):
    """Live view of the webcams on hub ports ``port_a`` / ``port_b`` (or
    port paths, ``cameras.index_of``) side by side with the board's
    corners drawn; space saves, under one name, each frame with
    MIN_CORNERS corners into ``dir_a`` / ``dir_b`` (a pair where both
    have them), q or Esc quits. Images already there are kept or deleted
    first (asked). True unless a camera failed."""
    caps = []
    try:
        for port, label in zip((port_a, port_b), labels):
            try:
                cap = cv2.VideoCapture(index_of(port), cv2.CAP_DSHOW)
            except LookupError as exc:
                print(exc)
                return False
            caps.append(cap)
            if not cap.isOpened():
                print(f'{label}: cannot open')
                return False
        for out_dir in (dir_a, dir_b):
            os.makedirs(out_dir, exist_ok=True)
            _delete_old(out_dir)
        number = max(_next_number(dir_a), _next_number(dir_b))
        detector = cv2.aruco.CharucoDetector(charuco_board())
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print(f'{labels[0]} -> {dir_a}, {labels[1]} -> {dir_b}; space saves, '
              'q quits')
        n_saved = n_pairs = 0
        last = [{}, {}]
        n_still = 0
        while True:
            frames, found, shown = [], [], []
            # both grabbed before either is decoded or searched: the two
            # pictures as close in time as the cameras allow
            if not all(cap.grab() for cap in caps):
                print('a camera was lost')
                return False
            for cap, label in zip(caps, labels):
                ok, frame = cap.retrieve()
                if not ok:
                    print(f'{label}: lost')
                    return False
                corners, ids, _, _ = detector.detectBoard(frame)
                pic = frame.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(pic, corners, ids)
                frames.append(frame)
                found.append({} if ids is None else dict(zip(
                    ids.ravel().tolist(), corners.reshape(-1, 2))))
                shown.append(pic)
            n = len(set(found[0]) & set(found[1]))
            ok_each = [len(f) >= MIN_CORNERS for f in found]
            moved = max((_motion(lf, f) for lf, f, ok
                         in zip(last, found, ok_each) if ok), default=np.inf)
            n_still = n_still + 1 if moved < STILL_PX else 0
            last = found
            still = n_still >= STILL_FRAMES
            color = ((0, 200, 0) if n >= MIN_CORNERS else (0, 200, 200)
                     if any(ok_each) else (0, 0, 255))
            both = np.hstack(shown)
            cv2.putText(both, f'{labels[0]} {len(found[0])}  {labels[1]} '
                        f'{len(found[1])}  both {n}  saved {n_saved} '
                        f'(pairs {n_pairs})  '
                        + ('still' if still else 'HOLD STILL'), (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(WINDOW, both)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                return True
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return True
            if key == ord(' '):
                if not any(ok_each):
                    print(f'fewer than {MIN_CORNERS} corners in each, not '
                          'saved')
                    continue
                if not still:
                    print(f'the board moved ({moved:.2f} px a frame), not '
                          'saved -- hold it still')
                    continue
                name = f'{number:03d}.png'
                for out_dir, frame, ok in zip((dir_a, dir_b), frames,
                                              ok_each):
                    if ok:
                        cv2.imwrite(os.path.join(out_dir, name), frame)
                number += 1
                n_saved += 1
                n_pairs += n >= MIN_CORNERS
                saved = ' + '.join(l for l, ok in zip(labels, ok_each) if ok)
                print(f'{name}: {saved} ({n} corners in both)')
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


def calibrate_handeye(robot, det_a, dir_a, det_b, dir_b, tf_a_b, out_a,
                      out_b):
    """Both cameras' hand-eyes as ONE fit: camera a's flange -> camera tf
    from the hand-eye runs of both (``dir_a`` seen by a, ``dir_b`` seen by
    b; the board may lie elsewhere in each), camera b held at ``tf_a_b``
    (:func:`fit`) from a -- twice the poses for one transform, and the two
    results agree with the stereo pose by construction. Writes ``out_a``
    / ``out_b`` (the layout of ``handeye.calibrate``); True when solved."""
    runs = []
    for det, data_dir in ((det_a, dir_a), (det_b, dir_b)):
        names, flanges, obs = che.load_run(robot, det, data_dir)
        if len(names) < 5:
            print(f'{data_dir}: only {len(names)} usable poses; need more')
            return False
        runs.append((det, names, flanges, obs))
    tf_a_b = np.asarray(tf_a_b, dtype=np.float64)
    inv_ab = np.linalg.inv(tf_a_b)
    # start: each run on its own (closed form); camera a from a's run
    x0, w_a0 = che._initial_guess(det_a, runs[0][2], runs[0][3])
    _, w_b0 = che._initial_guess(det_b, runs[1][2], runs[1][3])
    inv_f = [[np.linalg.inv(f) for f in run[2]] for run in runs]

    def residuals(p):
        inv_x = np.linalg.inv(che._tf_from_vec(p[:6]))
        res = []
        for k, (det, _, _, obs) in enumerate(runs):
            w = che._tf_from_vec(p[6 + 6 * k:12 + 6 * k])
            cam = inv_x if k == 0 else inv_ab @ inv_x
            res += [(det.project(cam @ f @ w, obj) - img).ravel()
                    for f, (obj, img) in zip(inv_f[k], obs)]
        return np.concatenate(res)

    p0 = np.concatenate([che._vec_from_tf(x0), che._vec_from_tf(w_a0),
                         che._vec_from_tf(w_b0)])
    sol = least_squares(residuals, p0, loss='huber', f_scale=che.HUBER_PX,
                        x_scale='jac')
    x_a = che._tf_from_vec(sol.x[:6])
    res = residuals(sol.x).reshape(-1, 2)
    sizes = np.cumsum([0] + [len(img) for run in runs
                             for _, img in run[3]])
    per_image = [float(np.sqrt(np.mean(np.sum(res[a:b] ** 2, axis=1))))
                 for a, b in zip(sizes[:-1], sizes[1:])]
    n_a = len(runs[0][1])
    print(f'{n_a} + {len(runs[1][1])} poses, camera b held at the stereo '
          'pose')
    for k, (label, x, out_path) in enumerate(
            (('a', x_a, out_a), ('b', x_a @ tf_a_b, out_b))):
        print(f'\n== camera {label}')
        w = che._tf_from_vec(sol.x[6 + 6 * k:12 + 6 * k])
        per = per_image[:n_a] if k == 0 else per_image[n_a:]
        che.report(x, w, per, runs[k][1])
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'tf_flange_cam': x.tolist(), 'tf_w_board': w.tolist(),
                       'rms_px': float(np.sqrt(np.mean(np.square(per)))),
                       'n_poses': len(per), 'joint': True}, f, indent=2)
        print(out_path)
    return True


def load_stereo(path):
    """The 4x4 pose of camera b in camera a (``x_a = tf @ x_b``)."""
    with open(path, encoding='utf-8') as f:
        return np.array(json.load(f)['tf_a_b'])


def _solve(obj, px_a, px_b, det_a, det_b):
    """``(tf_a_b, per pair RMS px, RMS px)`` fitted to the pairs."""
    # start: each image's board pose alone (PnP on undistorted points)
    poses_a, poses_b = [], []
    for o, pa, pb in zip(obj, px_a, px_b):
        for det, px, poses in ((det_a, pa, poses_a), (det_b, pb, poses_b)):
            _, rvec, tvec = cv2.solvePnP(o, det.normalize(px), np.eye(3),
                                         None)
            poses.append(che._tf(tvec, cv2.Rodrigues(rvec)[0]))
    starts = [ta @ np.linalg.inv(tb) for ta, tb in zip(poses_a, poses_b)]
    tf0 = starts[int(np.argmin([np.median([np.linalg.norm(
        s[:3, 3] - t[:3, 3]) for t in starts]) for s in starts]))]

    def residuals(p):
        inv_ab = np.linalg.inv(che._tf_from_vec(p[:6]))
        res = []
        for k, (o, pa, pb) in enumerate(zip(obj, px_a, px_b)):
            tf_a_board = che._tf_from_vec(p[6 + 6 * k:12 + 6 * k])
            res.append((det_a.project(tf_a_board, o) - pa).ravel())
            res.append((det_b.project(inv_ab @ tf_a_board, o) - pb).ravel())
        return np.concatenate(res)

    p0 = np.concatenate([che._vec_from_tf(tf0)]
                        + [che._vec_from_tf(t) for t in poses_a])
    sol = least_squares(residuals, p0, loss='huber', f_scale=che.HUBER_PX,
                        x_scale='jac')
    res = residuals(sol.x)
    sizes = np.cumsum([0] + [2 * (len(pa) + len(pb))
                             for pa, pb in zip(px_a, px_b)])
    per_pair = [float(np.sqrt(np.mean(np.square(res[i:j])) * 2))
                for i, j in zip(sizes[:-1], sizes[1:])]
    rms = float(np.sqrt(np.mean(np.square(res)) * 2))
    return che._tf_from_vec(sol.x[:6]), per_pair, rms


def fit(dir_a, dir_b, intrinsics_a, intrinsics_b, out_path):
    """Solve the pose between the cameras whose images are in ``dir_a`` /
    ``dir_b`` (lens fits at ``intrinsics_a`` / ``intrinsics_b``), write
    ``out_path``; True when solved."""
    names, obj_pts, img_a, img_b, size = common_corners(dir_a, dir_b)
    dropped = set()
    for path in (intrinsics_a, intrinsics_b):
        with open(path, encoding='utf-8') as f:
            dropped |= set(json.load(f).get('dropped', []))
    keep = [i for i, n in enumerate(names) if n not in dropped]
    if len(keep) < len(names):
        print(f'{len(names) - len(keep)} pairs left out: an image the lens '
              'fit dropped')
    if len(keep) < 5:
        print(f'only {len(keep)} usable image pairs; need more')
        return False
    det_a = che.BoardDetector(*lens.load_intrinsics(intrinsics_a))
    det_b = che.BoardDetector(*lens.load_intrinsics(intrinsics_b))
    names = [names[i] for i in keep]
    obj = [obj_pts[i].reshape(-1, 3) for i in keep]
    px_a = [img_a[i].reshape(-1, 2) for i in keep]
    px_b = [img_b[i].reshape(-1, 2) for i in keep]
    tf_a_b, per_pair, rms = _solve(obj, px_a, px_b, det_a, det_b)
    bad = lens.outliers(per_pair)
    if bad.any():
        print(f'dropped {int(bad.sum())} pairs fitting much worse (the board '
              'moved between the two pictures?): ' + ', '.join(
                  f'{names[i]} {per_pair[i]:.2f}px'
                  for i in np.flatnonzero(bad)))
        names, obj, px_a, px_b = ([x[i] for i in np.flatnonzero(~bad)]
                                  for x in (names, obj, px_a, px_b))
        tf_a_b, per_pair, rms = _solve(obj, px_a, px_b, det_a, det_b)
    angle = np.degrees(np.arccos(np.clip(
        (np.trace(tf_a_b[:3, :3]) - 1) / 2, -1, 1)))
    print(f'{len(names)} image pairs, '
          f'{sum(len(o) for o in obj)} corners in both')
    print(f'RMS reprojection error {rms:.3f} px (both cameras)')
    print(f'camera b in camera a: pos {np.round(tf_a_b[:3, 3] * 1000, 2)} '
          f'mm, baseline {np.linalg.norm(tf_a_b[:3, 3]) * 1000:.2f} mm, '
          f'rotation {angle:.2f} deg')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'tf_a_b': tf_a_b.tolist(), 'rms_px': float(rms),
                   'n_pairs': len(names), 'image_size': list(size)}, f,
                  indent=2)
    print(out_path)
    return True
