"""Hand-eye calibration of a camera on a robot's flange: where the camera
sits on it.

Input: the images and joint angles recorded by ``handeye_capture`` (the
board lies still on the table, the robot looks at it from many poses).
Unknowns: the flange -> camera tf X and the board's pose in the world W.
For image i with flange pose F_i (FK of the measured joints), the board
corners must project to where they were detected:

    corner pixel  =  lens( (F_i X)^-1 W  * corner on the board )

X and W (12 numbers) are fitted to all corners of all images at once by
nonlinear least squares on the pixel error (Huber loss, so a few bad
corners do not pull). The start value comes from OpenCV's closed-form
hand-eye (Park), which is only an initial guess here. The lens is the
fisheye of a wide-angle webcam or a plain pinhole (e.g. a RealSense's
rectified frame) (:class:`BoardDetector`).

Prints the pixel residual (overall and per image) and the board's height
and tilt as a check -- it lies flat on the table at about the robot base's
height, which the fit does not know. :func:`calibrate` writes the result
to a json.
"""

import os
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

import one.utils.math as oum

from one.vision.calibrate.charuco import MIN_CORNERS, charuco_board


RECORD_NAME = 'poses.json'      # the joint angles, next to the images
HUBER_PX = 1.0                  # residuals beyond this count linearly
N_WORST = 5


def _tf(pos, rotmat):
    """4x4 tf in float64 (``oum.tf_from_pos_rotmat`` gives float32)."""
    tf = np.eye(4)
    tf[:3, :3] = rotmat
    tf[:3, 3] = np.ravel(pos)
    return tf


def _tf_from_vec(v):
    """4x4 tf from (rotation vector, position)."""
    return _tf(v[3:], cv2.Rodrigues(np.ascontiguousarray(v[:3]))[0])


def _vec_from_tf(tf):
    return np.concatenate([cv2.Rodrigues(np.ascontiguousarray(
        tf[:3, :3]))[0].ravel(), tf[:3, 3]])


def load_handeye(path):
    """The 4x4 flange -> camera tf saved at ``path``."""
    with open(path, encoding='utf-8') as f:
        return np.array(json.load(f)['tf_flange_cam'])


class BoardDetector:
    """The board's corners and pose in a camera from one image. ``fisheye``:
    ``K``, ``D`` are of OpenCV's fisheye model (the webcams); else of the
    pinhole one (e.g. a RealSense's rectified frame, D zeros)."""

    def __init__(self, K, D, fisheye=True):
        self.board = charuco_board()
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.K, self.D, self.fisheye = K, D, fisheye

    def normalize(self, img):
        """Pixels (n, 2) -> normalized, undistorted image coordinates
        (n, 1, 2)."""
        img = img.reshape(-1, 1, 2)
        if self.fisheye:
            return cv2.fisheye.undistortPoints(img, self.K, self.D)
        return cv2.undistortPoints(img, self.K, self.D)

    def corners(self, image):
        """``(board points (n, 3), pixels (n, 2))`` of the detected corners,
        or None below MIN_CORNERS; and n."""
        corners, ids, _, _ = self.detector.detectBoard(image)
        n = 0 if ids is None else len(ids)
        if n < MIN_CORNERS:
            return None, n
        obj, img = self.board.matchImagePoints(corners, ids)
        return (obj.reshape(-1, 3).astype(np.float64),
                img.reshape(-1, 2).astype(np.float64)), n

    def pose(self, image):
        """``(4x4 board-in-camera tf, n corners)``; tf None below
        MIN_CORNERS."""
        found, n = self.corners(image)
        if found is None:
            return None, n
        obj, img = found
        # undistort to normalized coordinates, then a plain pinhole PnP
        ok, rvec, tvec = cv2.solvePnP(obj, self.normalize(img), np.eye(3),
                                      None)
        if not ok:
            return None, n
        return _tf(tvec, cv2.Rodrigues(rvec)[0]), n

    def project(self, tf_cam_board, obj=None):
        """Pixels of board points ``obj`` (all corners by default) for a
        board-in-camera tf."""
        if obj is None:
            obj = self.board.getChessboardCorners()
        rvec = cv2.Rodrigues(np.ascontiguousarray(tf_cam_board[:3, :3],
                                                  dtype=np.float64))[0]
        tvec = np.ascontiguousarray(tf_cam_board[:3, 3],
                                    dtype=np.float64).reshape(3, 1)
        obj = np.ascontiguousarray(obj, dtype=np.float64).reshape(-1, 1, 3)
        if self.fisheye:
            proj, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, self.K,
                                                self.D)
        else:
            proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.D)
        return proj.reshape(-1, 2)


def flange_tfs(robot, qs_list):
    """``robot``'s flange tf in the world for each joint vector."""
    tfs = []
    for qs in qs_list:
        robot.fk(qs=np.asarray(qs, dtype=np.float32))
        tfs.append(np.array(robot.tcp('flange').tf, dtype=np.float64))
    return tfs


def _initial_guess(detector, tfs_w_flange, observations):
    """``(X, W)`` from OpenCV's closed-form hand-eye (Park) and the board
    pose it implies from the first image."""
    tfs_cam_board = []
    for obj, img in observations:
        _, rvec, tvec = cv2.solvePnP(obj, detector.normalize(img), np.eye(3),
                                     None)
        tfs_cam_board.append(_tf(tvec, cv2.Rodrigues(rvec)[0]))
    rot, pos = cv2.calibrateHandEye(
        [t[:3, :3] for t in tfs_w_flange], [t[:3, 3] for t in tfs_w_flange],
        [t[:3, :3] for t in tfs_cam_board], [t[:3, 3] for t in tfs_cam_board],
        method=cv2.CALIB_HAND_EYE_PARK)
    x = _tf(pos, rot)
    return x, tfs_w_flange[0] @ x @ tfs_cam_board[0]


def solve(detector, tfs_w_flange, observations):
    """``(X flange -> camera, W board in world, per-image RMS px)`` fitted
    to all detected corners. ``observations``: ``(board points, pixels)``
    per image, as from :meth:`BoardDetector.corners`."""
    x0, w0 = _initial_guess(detector, tfs_w_flange, observations)
    inv_flanges = [np.linalg.inv(f) for f in tfs_w_flange]

    def residuals(p):
        inv_x = np.linalg.inv(_tf_from_vec(p[:6]))
        w = _tf_from_vec(p[6:])
        return np.concatenate([
            (detector.project(inv_x @ inv_f @ w, obj) - img).ravel()
            for inv_f, (obj, img) in zip(inv_flanges, observations)])

    p0 = np.concatenate([_vec_from_tf(x0), _vec_from_tf(w0)])
    fit = least_squares(residuals, p0, loss='huber', f_scale=HUBER_PX,
                        x_scale='jac')
    x, w = _tf_from_vec(fit.x[:6]), _tf_from_vec(fit.x[6:])
    res = residuals(fit.x).reshape(-1, 2)
    sizes = np.cumsum([0] + [len(img) for _, img in observations])
    per_image = [float(np.sqrt(np.mean(np.sum(res[a:b] ** 2, axis=1))))
                 for a, b in zip(sizes[:-1], sizes[1:])]
    moved = np.linalg.norm(x[:3, 3] - x0[:3, 3])
    print(f'  fit moved the camera {moved * 1000:.2f} mm from the '
          'closed-form start')
    return x, w, per_image


def report(tf_flange_cam, tf_w_board, per_image, names=None):
    """Print the result and the board check."""
    pos = tf_flange_cam[:3, 3]
    rpy = np.degrees(oum.euler_from_rotmat(tf_flange_cam[:3, :3]))
    print(f'camera in the flange frame: pos {np.round(pos * 1000, 2)} mm, '
          f'rpy {np.round(rpy, 2)} deg')
    errors = np.array(per_image)
    print(f'reprojection error: RMS {np.sqrt(np.mean(errors ** 2)):.3f} px '
          f'over {len(errors)} images, worst image {errors.max():.3f} px')
    if names is not None:
        print('  worst: ' + ', '.join(
            f'{names[i]} {errors[i]:.2f}px'
            for i in np.argsort(errors)[::-1][:N_WORST]))
    tilt = np.degrees(np.arccos(min(abs(float(tf_w_board[2, 2])), 1.0)))
    print(f'board origin in the world: '
          f'{np.round(tf_w_board[:3, 3] * 1000, 2)} mm, tilt from horizontal '
          f'{tilt:.2f} deg (it lies flat at about the base height, so the '
          'tilt should be ~0)')


def load_run(robot, detector, data_dir):
    """``(names, flange tfs, observations)`` of the usable images of one
    recorded run (images + ``poses.json`` in ``data_dir``, taken by
    ``robot``, seen by ``detector``'s camera)."""
    with open(os.path.join(data_dir, RECORD_NAME), encoding='utf-8') as f:
        records = json.load(f)
    names, qs_list, observations = [], [], []
    for rec in records:
        found, n = detector.corners(
            cv2.imread(os.path.join(data_dir, rec['image'])))
        if found is None:
            print(f"{rec['image']}: only {n} corners, skipped")
            continue
        names.append(rec['image'])
        qs_list.append(rec['qs'])
        observations.append(found)
    return names, flange_tfs(robot, qs_list), observations


def calibrate(robot, detector, data_dir, out_path):
    """Solve from the images and ``poses.json`` in ``data_dir`` (taken by
    ``robot``, seen by ``detector``'s camera) and write ``out_path``."""
    names, flanges, observations = load_run(robot, detector, data_dir)
    if len(names) < 5:
        print(f'only {len(names)} usable poses; need more')
        return False
    print(f'{len(names)} poses, '
          f'{sum(len(img) for _, img in observations)} corners')
    tf_flange_cam, tf_w_board, per_image = solve(
        detector, flanges, observations)
    report(tf_flange_cam, tf_w_board, per_image, names)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'tf_flange_cam': tf_flange_cam.tolist(),
                   'tf_w_board': tf_w_board.tolist(),
                   'rms_px': float(np.sqrt(np.mean(np.square(per_image)))),
                   'n_poses': len(names)}, f, indent=2)
    print(out_path)
    return True
