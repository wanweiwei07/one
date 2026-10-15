"""Dual-lens USB camera (Amazon B0HCVLX4RY).

The device enumerates as a single UVC camera named "USB Camera" and streams
both lenses side by side in one frame: 2560x720 = two 1280x720 views.

Set the resolution BEFORE the fourcc: in the other order DSHOW silently keeps
uncompressed YUY2, which USB bandwidth limits to 5 fps at 2560x720. MSMF
negotiates MJPG on its own but needs ~30 s to open, too slow to be usable here.
Quit with q or ESC. Ctrl-C usually works too, but it can land inside a
DirectShow call that never returns, leaving a process no kill can reach
until the camera is unplugged.
"""
import ctypes
import os

import cv2

CAM_INDEX = 2  # "USB Camera"; shifts when other cameras are plugged/unplugged
WIDTH, HEIGHT = 2560, 720  # 2560x800 / 1280x480 / 1280x400 also work
DISPLAY_SCALE = 0.5
BRIGHTNESS = -32  # the sensor blows out white walls at its default 0; -32 keeps
# the highlights, -64 is darker again. Gain/gamma/contrast barely move it.


# Ctrl-C often lands inside a DirectShow call that never returns, so python
# never gets to raise KeyboardInterrupt. Windows runs this handler on its own
# thread, which can still tear the process down.
@ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
def on_console_ctrl(event):
    os._exit(0)


ctypes.windll.kernel32.SetConsoleCtrlHandler(on_console_ctrl, True)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    for _ in range(10):  # the first reads come back empty while the stream starts
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            if w / h > 2.5:  # side-by-side frame, not a single-lens camera
                print(f"index {index}: {w}x{h} side-by-side")
                return cap
            print(f"index {index}: {w}x{h} single lens, skipping")
            break
    cap.release()
    return None


cap = open_camera(CAM_INDEX)
for i in range(5):  # index moved -- find the one giving a side-by-side frame
    if cap is not None:
        break
    if i != CAM_INDEX:
        cap = open_camera(i)
if cap is None:
    raise RuntimeError("no dual-lens camera found")

misses = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 100:
                raise RuntimeError("camera stopped delivering frames")
            continue
        misses = 0
        half = frame.shape[1] // 2
        left, right = frame[:, :half], frame[:, half:]
        cv2.imshow("left", cv2.resize(left, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
        cv2.imshow("right", cv2.resize(right, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
