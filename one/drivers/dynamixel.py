"""Dynamixel Protocol 2.0 over a serial (RS485/TTL) port -- no vendor SDK.

Written against the X-series control table (XH430 / XM430 and friends), the
same way ``cubemars.py`` speaks its own protocol: pyserial plus a packet
builder, so nothing has to be installed beyond pyserial.

Packet layout (Protocol 2.0)::

    FF FF FD 00  ID  LEN_L LEN_H  INST  PARAM...  CRC_L CRC_H

``LEN`` counts everything after itself (instruction + parameters + the two CRC
bytes) AFTER byte stuffing. Stuffing is what keeps a payload from imitating the
header: any ``FF FF FD`` inside the instruction/parameter region gets an extra
``FD`` appended, and the receiver strips it again.

Only the instructions a robot program actually needs are implemented (ping,
read, write, reboot); sync/bulk are deliberately left out.
"""

import struct
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:                                  # pragma: no cover
    serial = None
    list_ports = None


_HEADER = b'\xff\xff\xfd\x00'

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_REBOOT = 0x08
INST_STATUS = 0x55

# Hardware error bits reported in the status packet's error field, and in the
# Hardware Error Status register.
_HW_ERROR_BITS = {
    0x01: 'input voltage',
    0x04: 'overheating',
    0x08: 'motor encoder',
    0x10: 'electrical shock',
    0x20: 'overload',
}

_ALERT_BIT = 0x80          # status error field: "read Hardware Error Status"

# X-series control table: name -> (address, size in bytes). Signed entries are
# listed in _SIGNED below.
CONTROL_TABLE = {
    # EEPROM (writing these needs torque OFF)
    'model_number': (0, 2),
    'id': (7, 1),
    'baud_rate': (8, 1),
    'operating_mode': (11, 1),
    'temperature_limit': (31, 1),
    'max_voltage_limit': (32, 2),
    'min_voltage_limit': (34, 2),
    'pwm_limit': (36, 2),
    'current_limit': (38, 2),
    'velocity_limit': (44, 4),
    'max_position_limit': (48, 4),
    'min_position_limit': (52, 4),
    # RAM
    'torque_enable': (64, 1),
    'led': (65, 1),
    'hardware_error_status': (70, 1),
    'velocity_i_gain': (76, 2),
    'velocity_p_gain': (78, 2),
    'position_d_gain': (80, 2),
    'position_i_gain': (82, 2),
    'position_p_gain': (84, 2),
    'goal_pwm': (100, 2),
    'goal_current': (102, 2),
    'goal_velocity': (104, 4),
    'profile_acceleration': (108, 4),
    'profile_velocity': (112, 4),
    'goal_position': (116, 4),
    'moving': (122, 1),
    'moving_status': (123, 1),
    'present_pwm': (124, 2),
    'present_current': (126, 2),
    'present_velocity': (128, 4),
    'present_position': (132, 4),
    'present_input_voltage': (144, 2),
    'present_temperature': (146, 1),
}

_SIGNED = {'goal_pwm', 'goal_current', 'goal_velocity', 'present_pwm',
           'present_current', 'present_velocity', 'present_position',
           'goal_position', 'min_position_limit', 'max_position_limit'}

# Operating Mode (register 11)
MODE_CURRENT = 0
MODE_VELOCITY = 1
MODE_POSITION = 3
MODE_EXTENDED_POSITION = 4
MODE_CURRENT_POSITION = 5        # position control with a current ceiling
MODE_PWM = 16

# Model numbers seen on the X series, so a ping can say what answered.
MODEL_NAMES = {
    1000: 'XH430-W350', 1010: 'XH430-W210',
    1040: 'XH430-V350', 1050: 'XH430-V210',
    1020: 'XM430-W350', 1030: 'XM430-W210',
    1060: 'XL430-W250', 1120: 'XM540-W270', 1130: 'XM540-W150',
}

TICKS_PER_REV = 4096             # 0.087891 deg per tick
CURRENT_UNIT_MA = 2.69           # XH430 / XM430 current register unit


class DynamixelError(RuntimeError):
    """A bad packet, a timeout, or an error flag reported by the servo."""


def crc16(data):
    """CRC-16/IBM (poly 0x8005, init 0) over ``data`` -- the Protocol 2.0 CRC."""
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xffff if crc & 0x8000 else \
                (crc << 1) & 0xffff
    return crc


def stuff(payload):
    """Insert the extra ``FD`` after every ``FF FF FD`` in ``payload``."""
    out = bytearray()
    run = 0                                   # how much of FF FF FD matched
    for byte in payload:
        out.append(byte)
        if run == 0 and byte == 0xff:
            run = 1
        elif run == 1 and byte == 0xff:
            run = 2
        elif run == 2 and byte == 0xfd:
            out.append(0xfd)                  # the stuffing byte
            run = 0
        elif run == 2 and byte == 0xff:
            run = 2                           # FF FF FF -> still two in a row
        else:
            run = 1 if byte == 0xff else 0
    return bytes(out)


def unstuff(payload):
    """Drop the stuffing bytes ``stuff`` inserted."""
    out = bytearray()
    i = 0
    while i < len(payload):
        out.append(payload[i])
        if (payload[i] == 0xfd and len(out) >= 3 and
                out[-3] == 0xff and out[-2] == 0xff):
            i += 1                            # skip the duplicated FD
        i += 1
    return bytes(out)


def build_packet(dxl_id, instruction, params=b''):
    """Return the complete instruction packet for one servo."""
    body = stuff(bytes([instruction]) + bytes(params))
    length = len(body) + 2                    # + CRC
    frame = bytearray(_HEADER)
    frame.append(dxl_id & 0xff)
    frame += struct.pack('<H', length)
    frame += body
    frame += struct.pack('<H', crc16(frame))
    return bytes(frame)


def describe_hardware_error(bits):
    """Human-readable list of the hardware error bits set in ``bits``."""
    names = [name for bit, name in _HW_ERROR_BITS.items() if bits & bit]
    return ', '.join(names) if names else 'none'


def serial_ports():
    """Every serial port as ``(device, usb_serial, description)``.

    ``usb_serial`` is the serial number burned into the USB adapter's chip
    (``None`` for non-USB ports). It is what :func:`find_port` matches on.
    """
    if list_ports is None:
        raise DynamixelError('pyserial is not installed')
    return sorted((p.device, p.serial_number, p.description)
                  for p in list_ports.comports())


def find_port(usb_serial):
    """The port the USB adapter with serial number ``usb_serial`` is on.

    The COM number is handed out per machine (COM3 here, COM7 on the next
    laptop); the adapter's serial number is not, so matching on it is what
    lets one config work on every PC without editing.
    """
    ports = serial_ports()
    for device, sn, _ in ports:
        if sn == usb_serial:
            return device
    seen = ', '.join(f'{d} ({sn})' for d, sn, _ in ports if sn) or 'none'
    raise DynamixelError(
        f'no USB adapter with serial number {usb_serial} -- is it plugged '
        f'in? USB serial ports present: {seen}')


class Dynamixel:
    """One servo on one serial port.

    ``connect`` opens the port, every call is a blocking transaction, and the
    caller is expected to ``close`` (or use the context manager). Register
    access goes by NAME through :data:`CONTROL_TABLE`, so a wrong address or
    width cannot be spelled at the call site.

    The servo is addressed by its USB adapter's serial number, not by a COM
    name: ``connect`` looks the port up, so the same code works on any PC.
    """

    def __init__(self, usb_serial, baudrate=57600, dxl_id=1, timeout=0.1,
                 retries=2):
        self.usb_serial = usb_serial
        self.port = None                      # resolved by connect()
        self.baudrate = int(baudrate)
        self.dxl_id = int(dxl_id)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._ser = None

    # ---- connection -------------------------------------------------------
    def connect(self):
        if serial is None:
            raise DynamixelError('pyserial is not installed')
        self.port = find_port(self.usb_serial)
        try:
            self._ser = serial.Serial(self.port, self.baudrate,
                                      timeout=self.timeout,
                                      write_timeout=self.timeout)
        except OSError as exc:               # SerialException is an OSError
            raise DynamixelError(
                f'cannot open {self.port} ({exc}) -- something else has it '
                f'open') from exc
        self._ser.reset_input_buffer()
        return self

    def disconnect(self):
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    @property
    def is_connected(self):
        return self._ser is not None and self._ser.is_open

    # ---- transactions -----------------------------------------------------
    def _read_status(self):
        """Read one status packet; return its parameter bytes."""
        deadline = time.time() + self.timeout
        buf = bytearray()
        while time.time() < deadline:
            chunk = self._ser.read(64)
            if chunk:
                buf += chunk
                start = buf.find(_HEADER)
                if start >= 0 and len(buf) - start >= 7:
                    length = struct.unpack_from('<H', buf, start + 5)[0]
                    total = start + 7 + length
                    if len(buf) >= total:
                        return self._parse_status(bytes(buf[start:total]))
            elif buf:
                break
        raise DynamixelError(
            f'no status packet from id {self.dxl_id} on {self.port} '
            f'(got {bytes(buf)!r})')

    def _parse_status(self, frame):
        if crc16(frame[:-2]) != struct.unpack('<H', frame[-2:])[0]:
            raise DynamixelError(f'status CRC mismatch: {frame!r}')
        body = unstuff(frame[7:-2])
        if body[0] != INST_STATUS:
            raise DynamixelError(f'not a status packet: {frame!r}')
        err = body[1]
        # The alert bit only says a hardware error is LATCHED; the instruction
        # was still carried out and the data is valid. Failing on it would make
        # the latched error unreadable -- hardware_error() could never run.
        if err & ~_ALERT_BIT:
            raise DynamixelError(
                f'servo {self.dxl_id} returned error 0x{err:02x}')
        return body[2:]

    def _transact(self, instruction, params=b'', expect_status=True):
        if not self.is_connected:
            raise DynamixelError(f'adapter {self.usb_serial} is not open')
        packet = build_packet(self.dxl_id, instruction, params)
        last = None
        for _ in range(self.retries + 1):
            self._ser.reset_input_buffer()
            self._ser.write(packet)
            self._ser.flush()
            if not expect_status:
                return b''
            try:
                return self._read_status()
            except DynamixelError as exc:
                last = exc
        raise last

    # ---- instructions -----------------------------------------------------
    def ping(self):
        """Return ``(model_number, firmware_version)``; raises if silent."""
        params = self._transact(INST_PING)
        model, firmware = struct.unpack('<HB', params[:3])
        return model, firmware

    @staticmethod
    def model_name(model_number):
        return MODEL_NAMES.get(model_number, f'model {model_number}')

    def reboot(self):
        self._transact(INST_REBOOT)

    def read(self, name):
        """Read one register by name."""
        address, size = CONTROL_TABLE[name]
        params = self._transact(INST_READ,
                                struct.pack('<HH', address, size))
        if len(params) < size:
            raise DynamixelError(f'short read of {name}: {params!r}')
        value = int.from_bytes(params[:size], 'little',
                               signed=name in _SIGNED)
        return value

    def write(self, name, value):
        """Write one register by name."""
        address, size = CONTROL_TABLE[name]
        payload = int(value).to_bytes(size, 'little', signed=name in _SIGNED)
        self._transact(INST_WRITE, struct.pack('<H', address) + payload)

    # ---- convenience ------------------------------------------------------
    def torque(self, on):
        self.write('torque_enable', 1 if on else 0)

    def set_operating_mode(self, mode):
        """Operating mode lives in EEPROM: torque has to be off to change it."""
        if self.read('torque_enable'):
            self.torque(False)
        self.write('operating_mode', int(mode))

    def hardware_error(self):
        """``(bits, description)`` from the Hardware Error Status register."""
        bits = self.read('hardware_error_status')
        return bits, describe_hardware_error(bits)

    def clear_hardware_error(self):
        """Reboot is the only way to clear a latched hardware error."""
        self.reboot()
        time.sleep(0.5)

    @staticmethod
    def ticks_to_deg(ticks):
        return ticks * 360.0 / TICKS_PER_REV

    @staticmethod
    def deg_to_ticks(deg):
        return int(round(deg * TICKS_PER_REV / 360.0))

    def present_current_ma(self):
        return self.read('present_current') * CURRENT_UNIT_MA

    def wait_until_stopped(self, timeout=3.0, poll=0.02):
        """Block until the ``moving`` flag clears. False on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.read('moving'):
                return True
            time.sleep(poll)
        return False
