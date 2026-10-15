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

import cv2
from cv2 import aruco

from one.vision.ar_marker.make_pdfboard import make_charuco_board

from one.vision.calibrate.cameras import find_by_port


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


def capture(port, out_dir, label='camera'):
    """Live view of the webcam on hub ``port`` with the board's corners
    drawn; space saves the frame into ``out_dir`` (only with MIN_CORNERS),
    q or Esc quits. True unless the camera failed."""
    by_port = find_by_port()
    if port not in by_port:
        print(f'no camera on hub port {port} (found: {by_port})')
        return False
    os.makedirs(out_dir, exist_ok=True)
    n_saved = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    detector = cv2.aruco.CharucoDetector(charuco_board())
    cap = cv2.VideoCapture(by_port[port], cv2.CAP_DSHOW)
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
                path = os.path.join(out_dir, f'{n_saved:03d}.png')
                cv2.imwrite(path, frame)     # the raw frame, no drawing
                n_saved += 1
                print(f'{path} ({n} corners)')
    finally:
        cap.release()
        cv2.destroyAllWindows()

