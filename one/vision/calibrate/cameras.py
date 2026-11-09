"""Find USB webcams of one model by the hub port they are plugged into.

When the cameras are the same model and even report the same USB serial
string, neither the name nor the serial tells them apart, and OpenCV's
index changes from PC to PC. What stays put is WHICH PORT of their USB hub
each one is plugged into: the hub may move to another PC, the camera cables
stay in the same hub ports. So a camera is found as "the VID:PID webcam on
hub port N", and that is mapped to whatever OpenCV index it has today.

With cameras on more than one hub the port alone repeats (two hubs both
have a port 3), so a camera can also be given as its port path read up
the tree: ``(3, 7)`` = port 3 of the hub that is on port 7. A plain ``N``
is ``(N,)``; give as many levels as it takes to tell the cameras apart.

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


def _port_path(instance_id):
    """The hub ports of the USB device whose interface is ``instance_id``,
    read up the tree: its own port first, then its hub's port on the next
    hub, ... (each parent's location, ``Port_#000N.Hub_#00MM``), up to the
    root hub. ``()`` when it cannot be read."""
    cfg = ctypes.windll.cfgmgr32
    node = ctypes.c_uint32()
    if cfg.CM_Locate_DevNodeW(ctypes.byref(node), ctypes.c_wchar_p(instance_id),
                              _CM_LOCATE_DEVNODE_NORMAL) != _CR_SUCCESS:
        return ()
    ports = []
    while True:
        parent = ctypes.c_uint32()
        if cfg.CM_Get_Parent(ctypes.byref(parent), node, 0) != _CR_SUCCESS:
            break
        node = parent
        buf = ctypes.create_unicode_buffer(256)
        size = ctypes.c_uint32(ctypes.sizeof(buf))
        if cfg.CM_Get_DevNode_Registry_PropertyW(
                node, _CM_DRP_LOCATION_INFORMATION, None, buf,
                ctypes.byref(size), 0) != _CR_SUCCESS:
            break
        m = re.match(r'Port_#(\d+)', buf.value)
        if m is None:
            break
        ports.append(int(m.group(1)))
    return tuple(ports)


def find_by_port(vid=CAM_VID, pid=CAM_PID, backend=cv2.CAP_DSHOW):
    """``{port path: OpenCV index}`` for every ``vid:pid`` camera present
    (port path: :func:`_port_path`, the camera's own hub port first)."""
    found = {}
    for cam in enumerate_cameras(backend):
        if cam.vid != vid or cam.pid != pid:
            continue
        instance = _instance_id(cam.path)
        ports = () if instance is None else _port_path(instance)
        if ports:
            found[ports] = cam.index
    return found


def index_of(port, vid=CAM_VID, pid=CAM_PID, backend=cv2.CAP_DSHOW):
    """The OpenCV index of the ``vid:pid`` camera on ``port``: a hub port
    ``N`` or a port path ``(N, M, ...)`` (camera's port first). Raises
    LookupError unless exactly one camera matches -- never a guess."""
    want = (port,) if isinstance(port, int) else tuple(port)
    found = find_by_port(vid, pid, backend)
    hits = [index for ports, index in found.items()
            if ports[:len(want)] == want]
    if len(hits) != 1:
        raise LookupError(f'{len(hits)} cameras on hub port {want} '
                          f'(found, port path -> index: {found})')
    return hits[0]


if __name__ == '__main__':
    print(f'{CAM_VID:04X}:{CAM_PID:04X} cameras by port path -> OpenCV index:',
          find_by_port())
