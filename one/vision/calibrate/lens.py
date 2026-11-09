"""Lens calibration of a webcam from the ChArUco images saved by
``charuco.capture``.

Fisheye model (OpenCV ``cv2.fisheye``: K + 4 distortion coefficients) -- on
this wide-angle lens it fits as well as the 8-coefficient rational pinhole
model and much better than the 5-coefficient one. :func:`fit` drops the
images that fit much worse than the rest (blur, a bent board, a corner
found wrong) and fits again, prints the RMS reprojection error and the
worst images kept and writes the result -- with the dropped images'
names -- to a json.
"""

import os
import glob
import json

import cv2
import numpy as np

from one.vision.calibrate.charuco import MIN_CORNERS, charuco_board


N_WORST = 5                     # how many of the worst images to list
# an image is dropped when its RMS is above median + OUTLIER_MADS robust
# std of all images', and above OUTLIER_MIN_PX
OUTLIER_MADS = 4
OUTLIER_MIN_PX = 0.5


def outliers(errors):
    """Mask of the ``errors`` far above the rest (see OUTLIER_MADS)."""
    errors = np.asarray(errors)
    med = np.median(errors)
    std = 1.4826 * np.median(np.abs(errors - med))
    return errors > max(med + OUTLIER_MADS * std, OUTLIER_MIN_PX)


def load_intrinsics(path):
    """``(K, D)`` of the fisheye model saved at ``path``."""
    with open(path, encoding='utf-8') as f:
        cal = json.load(f)
    return np.array(cal['K']), np.array(cal['D'])


def detect_all(image_dir):
    """``(names, object points, image points, image size)`` for every image
    with at least MIN_CORNERS board corners found."""
    board = charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    names, obj_pts, img_pts, size = [], [], [], None
    for path in sorted(glob.glob(os.path.join(image_dir, '*.png'))):
        image = cv2.imread(path)
        size = image.shape[1::-1]
        corners, ids, _, _ = detector.detectBoard(image)
        n = 0 if ids is None else len(ids)
        if n < MIN_CORNERS:
            print(f'{os.path.basename(path)}: only {n} corners, skipped')
            continue
        obj, img = board.matchImagePoints(corners, ids)
        names.append(os.path.basename(path))
        obj_pts.append(obj.reshape(-1, 1, 3).astype(np.float64))
        img_pts.append(img.reshape(-1, 1, 2).astype(np.float64))
    return names, obj_pts, img_pts, size


def calibrate(obj_pts, img_pts, size):
    """``(rms, K, D, rvecs, tvecs)`` of the fisheye model."""
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    return cv2.fisheye.calibrate(obj_pts, img_pts, size, None, None,
                                 flags=flags, criteria=criteria)


def per_image_rms(obj_pts, img_pts, K, D, rvecs, tvecs):
    """RMS reprojection error (px) of each image."""
    errors = []
    for obj, img, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K, D)
        errors.append(float(np.sqrt(np.mean(np.sum(
            (proj - img).reshape(-1, 2) ** 2, axis=1)))))
    return errors


def fit(image_dir, out_path):
    """Calibrate from the images in ``image_dir``, write ``out_path``."""
    names, obj_pts, img_pts, size = detect_all(image_dir)
    if len(names) < 10:
        print(f'only {len(names)} usable images in {image_dir}; take more')
        return False
    dropped = []
    while True:
        rms, K, D, rvecs, tvecs = calibrate(obj_pts, img_pts, size)
        errors = per_image_rms(obj_pts, img_pts, K, D, rvecs, tvecs)
        bad = outliers(errors)
        if not bad.any():
            break
        dropped += [(names[i], errors[i]) for i in np.flatnonzero(bad)]
        names, obj_pts, img_pts = ([x[i] for i in np.flatnonzero(~bad)]
                                   for x in (names, obj_pts, img_pts))
    if dropped:
        print(f'dropped {len(dropped)} images fitting much worse: '
              + ', '.join(f'{n} {e:.2f}px' for n, e in dropped))
    print(f'{len(names)} images, {sum(len(p) for p in img_pts)} corners, '
          f'{size[0]}x{size[1]}')
    print(f'RMS reprojection error {rms:.3f} px')
    print(f'fx {K[0, 0]:.2f}  fy {K[1, 1]:.2f}  cx {K[0, 2]:.2f}  '
          f'cy {K[1, 2]:.2f}')
    print(f'D (k1..k4) {np.round(D.ravel(), 5)}')
    print('worst images:', ', '.join(
        f'{names[i]} {errors[i]:.2f}px'
        for i in np.argsort(errors)[::-1][:N_WORST]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'model': 'fisheye',
                   'image_size': [int(size[0]), int(size[1])],
                   'K': K.tolist(),
                   'D': D.ravel().tolist(),
                   'rms_px': float(rms),
                   'n_images': len(names),
                   'dropped': [n for n, _ in dropped]}, f, indent=2)
    print(out_path)
    return True
