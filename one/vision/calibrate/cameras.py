"""Find USB webcams of one model by the hub port they are plugged into.

When the cameras are the same model and even report the same USB serial
string, neither the name nor the serial tells them apart, and OpenCV's
index changes from PC to PC. What stays put is WHICH PORT of their USB hub
each one is plugged into: the hub may move to another PC, the camera cables
stay in the same hub ports. So a camera is found as "the VID:PID webcam on
hub port N", and that is mapped to whatever OpenCV index it has today.

Windows only: the port comes from the device tree (``cfgmgr32``), the OpenCV
index from ``cv2_enumerate_cameras``, matched by the device instance id.
"""

import re
import ctypes

import cv2
from cv2_enumerate_cameras import enumerate_cameras


# the default model ("VGA USB Camera"); other models pass their own
CAM_VID = 0x32E4
CAM_PID = 0x9210

_CM_LOCATE_DEVNODE_NORMAL = 0
_CM_DRP_LOCATION_INFORMATION = 0x0E
_CR_SUCCESS = 0


def _instance_id(path):
    """The device instance id inside a DirectShow device path, e.g.
    ``\\\\?\\usb#vid_32e4&pid_9210&mi_00#a&1b516722&0&0000#{...}`` ->
    ``USB\\VID_32E4&PID_9210&MI_00\\A&1B516722&0&0000``."""
    m = re.match(r'\\\\\?\\(usb)#([^#]+)#([^#]+)#', path, re.IGNORECASE)
    return None if m is None else '\\'.join(m.groups()).upper()


def _hub_port(instance_id):
    """The port number on its hub of the USB device whose interface is
    ``instance_id`` (its parent's location, ``Port_#000N.Hub_#00MM``)."""
    cfg = ctypes.windll.cfgmgr32
    node, parent = ctypes.c_uint32(), ctypes.c_uint32()
    if cfg.CM_Locate_DevNodeW(ctypes.byref(node), ctypes.c_wchar_p(instance_id),
                              _CM_LOCATE_DEVNODE_NORMAL) != _CR_SUCCESS:
        return None
    if cfg.CM_Get_Parent(ctypes.byref(parent), node, 0) != _CR_SUCCESS:
        return None
    buf = ctypes.create_unicode_buffer(256)
    size = ctypes.c_uint32(ctypes.sizeof(buf))
    if cfg.CM_Get_DevNode_Registry_PropertyW(
            parent, _CM_DRP_LOCATION_INFORMATION, None, buf,
            ctypes.byref(size), 0) != _CR_SUCCESS:
        return None
    m = re.match(r'Port_#(\d+)', buf.value)
    return None if m is None else int(m.group(1))


def find_by_port(vid=CAM_VID, pid=CAM_PID, backend=cv2.CAP_DSHOW):
    """``{hub port: OpenCV index}`` for every ``vid:pid`` camera present."""
    found = {}
    for cam in enumerate_cameras(backend):
        if cam.vid != vid or cam.pid != pid:
            continue
        instance = _instance_id(cam.path)
        port = None if instance is None else _hub_port(instance)
        if port is not None:
            found[port] = cam.index
    return found


if __name__ == '__main__':
    print(f'{CAM_VID:04X}:{CAM_PID:04X} cameras by hub port -> OpenCV index:',
          find_by_port())
