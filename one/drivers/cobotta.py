"""b-CAP (ORiN2) client for DENSO controllers + a COBOTTA joint-state reader.

Wire format is little endian throughout (DENSO "Specifications of b-CAP
Communication", RC8):

    request  SOH | len u32 | serial u16 | reserved u16 | func_id u32 |
             argc u16 | args... | EOT
    response SOH | len u32 | serial u16 | reserved u16 | hresult u32 |
             argc u16 | args... | EOT
    arg      len u32 (= 2 + 4 + len(data)) | vt u16 | count u32 | data

``len`` counts the whole packet, SOH and EOT included, so the fixed part is
16 bytes. A VT_BSTR is a u32 byte count followed by UTF-16LE bytes.
"""

import socket
import struct

import numpy as np

_SOH = 0x01
_EOT = 0x04
_PACKET_BASE_SIZE = 16
_DEFAULT_PORT = 5007

# VARIANT type ids
VT_EMPTY = 0
VT_NULL = 1
VT_I2 = 2
VT_I4 = 3
VT_R4 = 4
VT_R8 = 5
VT_CY = 6
VT_DATE = 7
VT_BSTR = 8
VT_BOOL = 11
VT_VARIANT = 12
VT_UI1 = 17
VT_UI2 = 18
VT_UI4 = 19
VT_ARRAY = 0x2000

_SCALAR_FMT = {
    VT_UI1: '<B',
    VT_I2: '<h',
    VT_UI2: '<H',
    VT_BOOL: '<h',
    VT_I4: '<i',
    VT_UI4: '<I',
    VT_R4: '<f',
    VT_R8: '<d',
    VT_DATE: '<d',
    VT_CY: '<q',
}

# b-CAP function ids
_FUNC_SERVICE_START = 1
_FUNC_SERVICE_STOP = 2
_FUNC_CONTROLLER_CONNECT = 3
_FUNC_CONTROLLER_DISCONNECT = 4
_FUNC_CONTROLLER_GETROBOT = 7
_FUNC_CONTROLLER_GETVARIABLE = 9
_FUNC_CONTROLLER_EXECUTE = 17
_FUNC_ROBOT_GETVARIABLE = 62
_FUNC_ROBOT_EXECUTE = 64
_FUNC_ROBOT_MOVE = 72
_FUNC_ROBOT_RELEASE = 84
_FUNC_VARIABLE_GETVALUE = 101
_FUNC_VARIABLE_RELEASE = 111


class BCapError(IOError):
    """A controller-side b-CAP failure (HRESULT with the error bit set)."""

    def __init__(self, hresult, func_id):
        self.hresult = hresult
        self.func_id = func_id
        super().__init__(
            f'b-CAP function {func_id} failed, hresult=0x{hresult:08X}')


def _pack_item(base_vt, value):
    if base_vt == VT_VARIANT:
        return _pack_variant_body(*_auto_arg(value))
    if base_vt == VT_BSTR:
        payload = str(value).encode('utf-16-le')
        return struct.pack('<I', len(payload)) + payload
    if base_vt in (VT_EMPTY, VT_NULL):
        return b''
    if base_vt == VT_BOOL:
        return struct.pack('<h', -1 if value else 0)
    if base_vt == VT_CY:
        return struct.pack('<q', int(round(float(value) * 10000.0)))
    try:
        fmt = _SCALAR_FMT[base_vt]
    except KeyError:
        raise ValueError(f'cannot encode variant type {base_vt}')
    return struct.pack(fmt, value)


def _pack_variant_body(vt, value):
    """Serialize ``vt u16 | count u32 | data``, the layout shared by an argument
    body and by one element of a VT_VARIANT array."""
    base_vt = vt & ~VT_ARRAY
    items = list(value) if vt & VT_ARRAY else [value]
    data = b''.join(_pack_item(base_vt, item) for item in items)
    return struct.pack('<HI', vt, len(items)) + data


def _pack_arg(vt, value):
    body = _pack_variant_body(vt, value)
    return struct.pack('<I', len(body)) + body


def _is_integral(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value):
    return _is_integral(value) or isinstance(value, (float, np.floating))


def _auto_arg(value):
    """Guess the variant type for one python value passed to a b-CAP call."""
    if value is None:
        return (VT_EMPTY, None)
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return (VT_BSTR, value)
    if isinstance(value, bool):
        return (VT_BOOL, value)
    if _is_integral(value):
        return (VT_I4, int(value))
    if isinstance(value, (float, np.floating)):
        return (VT_R8, float(value))
    if isinstance(value, np.ndarray):
        return (VT_R8 | VT_ARRAY, [float(item) for item in np.ravel(value)])
    if isinstance(value, list):
        if value and all(_is_integral(item) for item in value):
            return (VT_I4 | VT_ARRAY, [int(item) for item in value])
        if all(_is_number(item) for item in value):
            return (VT_R8 | VT_ARRAY, [float(item) for item in value])
        # mixed, e.g. a Move pose [[j1..j6], 'J', '@0']
        return (VT_VARIANT | VT_ARRAY, value)
    raise ValueError(f'cannot encode {type(value)} as a b-CAP argument')


def _read_data(buf, offset, vt, count):
    base_vt = vt & ~VT_ARRAY
    items = []
    for _ in range(max(int(count), 0)):
        if base_vt == VT_BSTR:
            n_bytes, = struct.unpack_from('<I', buf, offset)
            offset += 4
            items.append(buf[offset:offset + n_bytes].decode('utf-16-le'))
            offset += n_bytes
        elif base_vt == VT_VARIANT:
            sub_vt, sub_count = struct.unpack_from('<HI', buf, offset)
            offset += 6
            sub_value, offset = _read_data(buf, offset, sub_vt, sub_count)
            items.append(sub_value)
        elif base_vt in (VT_EMPTY, VT_NULL):
            items.append(None)
        else:
            try:
                fmt = _SCALAR_FMT[base_vt]
            except KeyError:
                raise IOError(f'cannot decode variant type {base_vt}')
            item, = struct.unpack_from(fmt, buf, offset)
            offset += struct.calcsize(fmt)
            if base_vt == VT_BOOL:
                item = item != 0
            elif base_vt == VT_CY:
                item = item / 10000.0
            items.append(item)
    if vt & VT_ARRAY:
        return items, offset
    return (items[0] if items else None), offset


def _read_arg(buf, offset):
    arg_len, = struct.unpack_from('<I', buf, offset)
    vt, count = struct.unpack_from('<HI', buf, offset + 4)
    value, _ = _read_data(buf, offset + 10, vt, count)
    return value, offset + 4 + arg_len


class BCapClient:
    """Minimal synchronous b-CAP client over TCP."""

    def __init__(self, host, port=_DEFAULT_PORT, timeout=3.0):
        self.host = host
        self.port = int(port)
        self._sock = socket.create_connection((host, self.port),
                                              timeout=timeout)
        self._sock.settimeout(timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._serial = 0

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _recv_exactly(self, n_bytes):
        chunks = []
        remaining = n_bytes
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise ConnectionError('b-CAP socket closed by the controller')
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _request(self, func_id, args=()):
        self._serial = self._serial % 0xFFFF + 1
        body = b''.join(_pack_arg(vt, value) for vt, value in args)
        packet = (bytes([_SOH]) +
                  struct.pack('<IHHIH', _PACKET_BASE_SIZE + len(body),
                              self._serial, 0, func_id, len(args)) +
                  body +
                  bytes([_EOT]))
        self._sock.sendall(packet)
        return self._receive(func_id)

    def _receive(self, func_id):
        head = self._recv_exactly(5)
        if head[0] != _SOH:
            raise IOError(f'bad b-CAP start byte 0x{head[0]:02X}')
        total, = struct.unpack_from('<I', head, 1)
        if total < _PACKET_BASE_SIZE:
            raise IOError(f'bad b-CAP packet length {total}')
        packet = head + self._recv_exactly(total - 5)
        if packet[-1] != _EOT:
            raise IOError(f'bad b-CAP end byte 0x{packet[-1]:02X}')
        hresult, argc = struct.unpack_from('<IH', packet, 9)
        if hresult & 0x80000000:
            raise BCapError(hresult, func_id)
        offset = 15
        values = []
        for _ in range(argc):
            value, offset = _read_arg(packet, offset)
            values.append(value)
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def service_start(self, option=''):
        return self._request(_FUNC_SERVICE_START, [(VT_BSTR, option)])

    def service_stop(self):
        return self._request(_FUNC_SERVICE_STOP)

    def controller_connect(self, name='', provider='CaoProv.DENSO.VRC',
                           machine='localhost', option=''):
        return self._request(_FUNC_CONTROLLER_CONNECT, [
            (VT_BSTR, name),
            (VT_BSTR, provider),
            (VT_BSTR, machine),
            (VT_BSTR, option),
        ])

    def controller_disconnect(self, hctrl):
        return self._request(_FUNC_CONTROLLER_DISCONNECT, [(VT_I4, hctrl)])

    def controller_getrobot(self, hctrl, name='Arm', option=''):
        return self._request(_FUNC_CONTROLLER_GETROBOT, [
            (VT_I4, hctrl), (VT_BSTR, name), (VT_BSTR, option)])

    def controller_getvariable(self, hctrl, name, option=''):
        return self._request(_FUNC_CONTROLLER_GETVARIABLE, [
            (VT_I4, hctrl), (VT_BSTR, name), (VT_BSTR, option)])

    def controller_execute(self, hctrl, command, param=None):
        return self._request(_FUNC_CONTROLLER_EXECUTE, [
            (VT_I4, hctrl), (VT_BSTR, command), _auto_arg(param)])

    def robot_getvariable(self, hrobot, name, option=''):
        return self._request(_FUNC_ROBOT_GETVARIABLE, [
            (VT_I4, hrobot), (VT_BSTR, name), (VT_BSTR, option)])

    def robot_execute(self, hrobot, command, param=None):
        return self._request(_FUNC_ROBOT_EXECUTE, [
            (VT_I4, hrobot), (VT_BSTR, command), _auto_arg(param)])

    def robot_move(self, hrobot, comp, pose, option=''):
        return self._request(_FUNC_ROBOT_MOVE, [
            (VT_I4, hrobot), (VT_I4, comp), (VT_BSTR, pose), (VT_BSTR, option)])

    def robot_release(self, hrobot):
        return self._request(_FUNC_ROBOT_RELEASE, [(VT_I4, hrobot)])

    def variable_getvalue(self, hvar):
        return self._request(_FUNC_VARIABLE_GETVALUE, [(VT_I4, hvar)])

    def variable_release(self, hvar):
        return self._request(_FUNC_VARIABLE_RELEASE, [(VT_I4, hvar)])


class Cobotta:
    """Read-only b-CAP session with one COBOTTA controller.

    Only observes: no TakeArm, no motor power, nothing that can move the arm.
    The ``@CURRENT_ANGLE`` handle is opened once at connect, so each
    ``joint_angles`` poll costs a single round trip.
    """

    def __init__(self, host, port=_DEFAULT_PORT, timeout=3.0,
                 provider='CaoProv.DENSO.VRC', robot_name='Arm'):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.provider = provider
        self.robot_name = robot_name
        self._client = None
        self._hctrl = None
        self._hrobot = None
        self._hangle = None

    @property
    def is_connected(self):
        return self._hangle is not None

    def connect(self):
        self._client = BCapClient(self.host, self.port, self.timeout)
        try:
            self._client.service_start('')
            self._hctrl = self._client.controller_connect(
                '', self.provider, 'localhost', '')
            self._hrobot = self._client.controller_getrobot(
                self._hctrl, self.robot_name, '')
            self._hangle = self._client.robot_getvariable(
                self._hrobot, '@CURRENT_ANGLE', '')
        except Exception:
            self.disconnect()
            raise
        return self

    def joint_angles(self, ndof=6):
        """Return the current joint angles in radians, J1 first."""
        if not self.is_connected:
            raise RuntimeError(f'{self.host} is not connected')
        values = self._client.variable_getvalue(self._hangle)
        if not isinstance(values, (list, tuple)):
            values = [values]
        if len(values) < ndof:
            raise IOError(
                f'{self.host} reported {len(values)} joints, expected {ndof}')
        return np.deg2rad(
            np.asarray(values[:ndof], dtype=np.float32)).astype(np.float32)

    def disconnect(self):
        for release, handle in ((BCapClient.variable_release, self._hangle),
                                (BCapClient.robot_release, self._hrobot),
                                (BCapClient.controller_disconnect, self._hctrl)):
            if handle is None or self._client is None:
                continue
            try:
                release(self._client, handle)
            except (OSError, BCapError):
                pass
        self._hangle = None
        self._hrobot = None
        self._hctrl = None
        if self._client is not None:
            try:
                self._client.service_stop()
            except (OSError, BCapError):
                pass
            self._client.close()
            self._client = None

    def __enter__(self):
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        return False


class CobottaMotion(Cobotta):
    """A COBOTTA session that can also MOVE the arm.

    Everything :class:`Cobotta` reads stays available; on top of that this
    class takes the arm, powers the motors, and plays joint waypoints with
    ``Move``. Arm control is scoped and explicit: nothing can move until
    :meth:`enable` is called, and :meth:`disable` -- also run by :meth:`close`
    and on context exit -- powers the motors off and gives the arm back.

    Angles are radians, J1 first, matching :meth:`Cobotta.joint_angles`, so a
    joint path planned against the simulated robot can be replayed as is.
    """

    MOVE_PTP = 1
    MOVE_LINEAR = 2

    def __init__(self, host, port=_DEFAULT_PORT, timeout=30.0,
                 provider='CaoProv.DENSO.VRC', robot_name='Arm',
                 speed=10.0):
        # a Move blocks until the arm arrives, so the socket timeout has to
        # cover a whole motion, not just a round trip.
        super().__init__(host, port=port, timeout=timeout,
                         provider=provider, robot_name=robot_name)
        self.speed = float(speed)
        self._is_armed = False

    @property
    def is_armed(self):
        return self._is_armed

    def enable(self, speed=None, clear_error=True):
        """Take the arm and switch the motors on. Raises when the controller
        refuses (e.g. it is in manual mode, or another client holds the arm).

        Already enabled, this only updates the speed: powering the motors costs
        a brake release on both edges, so it is not something to cycle per
        motion -- enable once, run as many paths as you like, disable at the
        end.
        """
        if not self.is_connected:
            raise RuntimeError(f'{self.host} is not connected')
        if speed is not None:
            self.speed = float(speed)
        if self._is_armed:
            self.set_speed(self.speed)
            return self
        if clear_error:
            try:
                self._client.controller_execute(self._hctrl, 'ClearError', '')
            except BCapError:
                pass
        self._client.robot_execute(self._hrobot, 'TakeArm', [0, 0])
        self._is_armed = True
        try:
            self._client.robot_execute(self._hrobot, 'Motor', [1, 0])
            self.set_speed(self.speed)
        except Exception:
            self.disable()
            raise
        return self

    def disable(self):
        """Switch the motors off and give the arm back. Safe to call twice."""
        if not self._is_armed:
            return
        self._is_armed = False
        for command, param in (('Motor', [0, 0]), ('GiveArm', None)):
            try:
                self._client.robot_execute(self._hrobot, command, param)
            except (OSError, BCapError):
                pass

    def set_speed(self, speed):
        """Set the external speed/accel/decel percentages (1..100)."""
        self.speed = float(np.clip(speed, 1.0, 100.0))
        self._client.robot_execute(
            self._hrobot, 'ExtSpeed', [self.speed, self.speed, self.speed])

    def move_qs(self, qs, pass_motion=False, option=''):
        """Move to one joint configuration (radians, J1 first) by PTP.

        ``pass_motion`` issues ``@P`` instead of ``@0``: the controller may
        start the next command before this one settles, which keeps a
        multi-waypoint path smooth.
        """
        if not self._is_armed:
            raise RuntimeError(f'{self.host} arm is not enabled')
        degrees = np.degrees(np.asarray(qs, dtype=np.float64).ravel())
        joints = ', '.join(f'{angle:.4f}' for angle in degrees)
        pose = f"@{'P' if pass_motion else '0'} J({joints})"
        return self._client.robot_move(
            self._hrobot, self.MOVE_PTP, pose, option)

    def move_path(self, path, option='', on_waypoint=None):
        """Play a joint path. Intermediate waypoints pass through (``@P``);
        the last one stops precisely (``@0``). ``on_waypoint(i, qs)`` is called
        before each move, so a caller can log or mirror the motion."""
        path = list(path)
        last = len(path) - 1
        for i, qs in enumerate(path):
            if on_waypoint is not None:
                on_waypoint(i, qs)
            self.move_qs(qs, pass_motion=(i < last), option=option)
        return len(path)

    def disconnect(self):
        if self._is_armed:
            self.disable()
        super().disconnect()


if __name__ == '__main__':
    import sys

    host = sys.argv[1] if len(sys.argv) > 1 else '192.168.0.101'
    with Cobotta(host) as cobotta:
        qs = cobotta.joint_angles()
        print(f'{host} qs_deg={np.round(np.degrees(qs), 3)}')
