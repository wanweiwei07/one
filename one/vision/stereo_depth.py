"""Depth from a calibrated pair of cameras: rectify, disparity, depth,
points.

Rectification turns both pictures onto one plane so that a point lies on
the same row in both (Bouguet: the new x axis along the baseline, z the
mean of the two optical axes); the rotations are computed here and the
pictures remapped with ``cv2.fisheye.initUndistortRectifyMap`` (or the
pinhole one) -- not ``cv2.fisheye.stereoRectify``. Both rectified
pictures share one pinhole K (focal ``f``, centre of ``size``).

The rectified pictures stay upright (their x along camera a's x, which
way the baseline points decides only which picture is on the left).
Camera a is the reference: disparity d > 0 per pixel of ITS rectified
picture (when a is the right-hand camera, both pictures are mirrored for
the matcher and the result mirrored back), depth Z = f B / d, and points
come back in camera a's own frame.

Cameras that are not one rigid unit (two webcams) drift after the stereo
calibration: the rows stop meeting, by a px or so, changing over time.
``row_offset`` measures that per pair from SIFT matches (0.04-0.08 px off
a board's corners) and ``align_rows`` takes it out before matching.
"""

import cv2
import numpy as np


# SGBM defaults, for 640x480 at 0.1-0.3 m with a ~3 cm baseline
NUM_DISPARITIES = 96            # px, a multiple of 16
BLOCK_SIZE = 5                  # px, odd
UNIQUENESS = 10                 # %
SPECKLE_WINDOW = 100            # px, small blobs of disparity dropped
SPECKLE_RANGE = 2               # px
# rows (StereoRectifier.row_offset): SIFT features per picture, Lowe's
# ratio, matches further off a row than ROW_MATCH_SPAN px dropped, outlier
# rounds; too few matches / too large an offset: no fit
ROW_FEATURES = 4000
ROW_RATIO = 0.7
ROW_MATCH_SPAN = 15.0           # px
ROW_FIT_ROUNDS = 3
ROW_MIN_MATCHES = 50
ROW_MAX_OFFSET = 3.0            # px


class StereoRectifier:
    """The rectification of cameras a, b (``K``, ``D`` each; ``tf_a_b``:
    camera b's 4x4 pose in camera a), into pictures of ``size`` with
    focal ``f`` (px; default: camera a's)."""

    def __init__(self, K_a, D_a, K_b, D_b, tf_a_b, size=(640, 480),
                 f=None, fisheye=True):
        tf_a_b = np.asarray(tf_a_b, dtype=np.float64)
        t = tf_a_b[:3, 3]
        self.baseline = float(np.linalg.norm(t))
        x = t / self.baseline
        # upright: x as camera a's x; then b is right of a when along +x
        self.a_left = bool(x[0] >= 0)
        x = x if self.a_left else -x
        z = np.array([0.0, 0.0, 1.0]) + tf_a_b[:3, 2]
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        # camera a / b coordinates -> the common rectified frame
        self.R_a = np.stack([x, y, z])
        self.R_b = self.R_a @ tf_a_b[:3, :3]
        self.size = tuple(size)
        self.f = float(K_a[0, 0] if f is None else f)
        self.K = np.array([[self.f, 0.0, size[0] / 2],
                           [0.0, self.f, size[1] / 2],
                           [0.0, 0.0, 1.0]])
        init = (cv2.fisheye.initUndistortRectifyMap if fisheye
                else cv2.initUndistortRectifyMap)
        self.maps = []
        for K, D, R in ((K_a, D_a, self.R_a), (K_b, D_b, self.R_b)):
            D = np.asarray(D, dtype=np.float64)
            D = D.reshape(4, 1) if fisheye else D
            self.maps.append(init(np.asarray(K, dtype=np.float64), D, R,
                                  self.K, self.size, cv2.CV_32FC1))

    def rectify(self, img_a, img_b):
        """Both pictures rectified."""
        return tuple(cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
                     for img, (m1, m2) in zip((img_a, img_b), self.maps))

    def disparity(self, rect_a, rect_b, matcher=None):
        """Disparity (px) per pixel of the rectified picture a, by
        ``matcher(left, right)`` (default :func:`disparity_sgbm`)."""
        matcher = disparity_sgbm if matcher is None else matcher
        if self.a_left:
            return matcher(rect_a, rect_b)
        return matcher(rect_a[:, ::-1], rect_b[:, ::-1])[:, ::-1]

    def depth(self, disparity):
        """Z (m) per pixel of the rectified picture a; nan where the
        disparity is not valid."""
        with np.errstate(divide='ignore', invalid='ignore'):
            z = self.f * self.baseline / disparity
        z[~(disparity > 0)] = np.nan
        return z

    def points(self, disparity, mask=None):
        """``(points (n, 3) in camera a's frame, (rows, cols) in the
        rectified picture a)`` of the valid pixels (and ``mask``)."""
        z = self.depth(disparity)
        keep = np.isfinite(z) if mask is None else np.isfinite(z) & mask
        v, u = np.nonzero(keep)
        z = z[v, u]
        rect = np.stack([(u - self.K[0, 2]) * z / self.f,
                         (v - self.K[1, 2]) * z / self.f, z], axis=1)
        return rect @ self.R_a, (v, u)

    def project_a(self, pts_a):
        """Pixels in the rectified picture a of points in camera a's
        frame."""
        rect = np.asarray(pts_a, dtype=np.float64) @ self.R_a.T
        return rect[:, :2] / rect[:, 2:] * self.f + self.K[:2, 2]

    def row_offset(self, rect_a, rect_b):
        """``(coef, n)``: how far rectified b's rows are off a's, ``dy =
        v_a - v_b = c0 + c1 u + c2 v`` at b's pixel (u, v), fitted to SIFT
        matches (ratio test, wrong-way and far-off matches dropped, then
        outliers over ROW_FIT_ROUNDS); n the matches kept. ``coef`` None
        when fewer than ROW_MIN_MATCHES or past ROW_MAX_OFFSET px at a
        corner of the picture. A drift of the cameras after calibration
        shows up here (a px of it skews the disparity on slanted edges by
        1 / tan of their angle) -- :meth:`align_rows` takes it out."""
        sift = cv2.SIFT_create(ROW_FEATURES)
        grey = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3
                else img for img in (rect_a, rect_b)]
        ka, da = sift.detectAndCompute(grey[0], None)
        kb, db = sift.detectAndCompute(grey[1], None)
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return None, 0
        good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2)
                if m.distance < ROW_RATIO * n.distance]
        if len(good) < ROW_MIN_MATCHES:
            return None, len(good)
        pa = np.array([ka[m.queryIdx].pt for m in good])
        pb = np.array([kb[m.trainIdx].pt for m in good])
        dy = pa[:, 1] - pb[:, 1]
        dx = (pa[:, 0] - pb[:, 0]) * (1 if self.a_left else -1)
        ok = (dx > 0) & (np.abs(dy) < ROW_MATCH_SPAN)
        A, y = np.c_[np.ones(ok.sum()), pb[ok]], dy[ok]
        for _ in range(ROW_FIT_ROUNDS):
            if len(y) < ROW_MIN_MATCHES:
                return None, len(y)
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
            res = np.abs(y - A @ coef)
            keep = res < 3 * np.median(res) + 0.05
            A, y = A[keep], y[keep]
        w, h = self.size
        corners = np.array([[1, 0, 0], [1, w, 0], [1, 0, h], [1, w, h]])
        if np.abs(corners @ coef).max() > ROW_MAX_OFFSET:
            return None, len(y)
        return coef, len(y)

    def align_rows(self, rect_b, coef):
        """Rectified b moved up / down by :meth:`row_offset`'s dy, its rows
        on a's."""
        h, w = rect_b.shape[:2]
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        shift = coef[0] + coef[1] * uu + coef[2] * vv
        return cv2.remap(rect_b, uu, (vv - shift).astype(np.float32),
                         cv2.INTER_LINEAR)


def disparity_sgbm(rect_a, rect_b, num_disparities=NUM_DISPARITIES,
                   block_size=BLOCK_SIZE, min_disparity=0):
    """Disparity (px, float32; nan where none) of two rectified pictures
    by semi-global block matching."""
    gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            for img in (rect_a, rect_b)]
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disparity, numDisparities=num_disparities,
        blockSize=block_size, P1=8 * block_size ** 2,
        P2=32 * block_size ** 2, uniquenessRatio=UNIQUENESS,
        speckleWindowSize=SPECKLE_WINDOW, speckleRange=SPECKLE_RANGE,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(gray[0], gray[1]).astype(np.float32) / 16.0
    disp[disp <= min_disparity] = np.nan
    return disp
