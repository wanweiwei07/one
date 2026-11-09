"""The ChArUco board for calibrating the cameras, and taking lens
calibration images of it.

Printing and detecting both use the constants here, so they always agree.
Print the PDF (:func:`make_pdf`, already next to this file) at 100 % (no
"fit to page") on A4, fix it flat on something stiff, then measure one
square with a caliper and put that in SQUARE_SIZE.

For the lens images (:func:`capture`), move the board (or the camera) so
that over the set the board reaches every part of the picture -- corners
and edges matter most, that is where the lens distorts -- at different
tilts and distances. 20-30 images.
"""

import os
import time

import cv2
from cv2 import aruco

from one.vision.ar_marker.make_pdfboard import make_charuco_board

from one.vision.calibrate.cameras import index_of


NROW = 10                       # squares down the (portrait A4) page
NCOLUMN = 7                     # squares across
SQUARE_SIZE = 0.025             # m -- replace with the measured print
MARKER_RATIO = .57              # marker side / square side
MARKER_DICT = aruco.DICT_4X4_250
MIN_CORNERS = 12                # fewer in an image: the image is not used
WINDOW = 'charuco capture'
NAME = f'charuco_{NROW}x{NCOLUMN}_{SQUARE_SIZE * 1000:g}mm'


def charuco_board():
    """The board as OpenCV detects it (sizes in m)."""
    return aruco.CharucoBoard((NCOLUMN, NROW), SQUARE_SIZE,
                              MARKER_RATIO * SQUARE_SIZE,
                              aruco.getPredefinedDictionary(MARKER_DICT))


def make_pdf():
    """Write the printable board next to this file; its path."""
    return make_charuco_board(NROW, NCOLUMN, marker_dict=MARKER_DICT,
                              square_size=SQUARE_SIZE * 1000,
                              marker_ratio=MARKER_RATIO,
                              save_path=os.path.dirname(
                                  os.path.abspath(__file__)),
                              name=NAME)


def _delete_old(out_dir):
    """Images already in ``out_dir`` would be fitted together with the new
    ones -- wrong once the lens (focus, zoom) was touched. Ask whether to
    delete them."""
    old = sorted(f for f in os.listdir(out_dir) if f.endswith('.png'))
    if not old:
        return
    newest = max(os.path.getmtime(os.path.join(out_dir, f)) for f in old)
    print(f'{len(old)} images already in {out_dir} (newest '
          f'{time.strftime("%Y-%m-%d %H:%M", time.localtime(newest))}). '
          'The lens fit uses all of them: keep them only if the lens was not '
          'touched since.')
    if input('delete them? y / Enter keeps: ').strip().lower() == 'y':
        for f in old:
            os.remove(os.path.join(out_dir, f))
        print(f'deleted {len(old)} images')


def _next_number(out_dir):
    """One past the highest ``NNN.png`` in ``out_dir`` (never overwrites,
    also when some were taken out)."""
    nums = [int(f[:-4]) for f in os.listdir(out_dir)
            if f.endswith('.png') and f[:-4].isdigit()]
    return max(nums) + 1 if nums else 0


def capture(port, out_dir, label='camera'):
    """Live view of the webcam on hub ``port`` with the board's corners
    drawn; space saves the frame into ``out_dir`` (only with MIN_CORNERS),
    q or Esc quits. Images already there are kept or deleted first
    (asked). True unless the camera failed."""
    try:
        index = index_of(port)
    except LookupError as exc:
        print(exc)
        return False
    os.makedirs(out_dir, exist_ok=True)
    _delete_old(out_dir)
    n_saved = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    number = _next_number(out_dir)
    detector = cv2.aruco.CharucoDetector(charuco_board())
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f'{label}: cannot open')
        return False
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print(f'{label} -> {out_dir} ({n_saved} images already there); '
          'space saves, q quits')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f'{label}: lost')
                return False
            corners, ids, _, _ = detector.detectBoard(frame)
            n = 0 if ids is None else len(ids)
            shown = frame.copy()
            if n:
                cv2.aruco.drawDetectedCornersCharuco(shown, corners, ids)
            color = (0, 200, 0) if n >= MIN_CORNERS else (0, 0, 255)
            cv2.putText(shown, f'corners {n}  saved {n_saved}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                return True
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return True
            if key == ord(' '):
                if n < MIN_CORNERS:
                    print(f'only {n} corners, not saved')
                    continue
                path = os.path.join(out_dir, f'{number:03d}.png')
                cv2.imwrite(path, frame)     # the raw frame, no drawing
                number += 1
                n_saved += 1
                print(f'{path} ({n} corners)')
    finally:
        cap.release()
        cv2.destroyAllWindows()

