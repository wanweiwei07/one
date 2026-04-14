import struct
import time

import can
try:
    import serial
except ImportError:
    serial = None


class CubeMarsNativeCAN:
    """CubeMars servo CAN packet builder (extended CAN frame)."""

    CAN_ID_BITS = {
        'Control_mode': slice(8, 16),
        'Source_node_ID': slice(0, 8),
    }

    CONTROL_MODES = {
        'Duty_cycle_mode': 0,
        'Current_loop_mode': 1,
        'Current_brake_mode': 2,
        'Velocity_mode': 3,
        'Position_mode': 4,
        'Set_origin_mode': 5,
        'Position_velocity_loop_mode': 6,
    }

    def __init__(self, node_id):
        self.node_id = int(node_id)

    def _build_arbitration_id(self, control_mode):
        shift = self.CAN_ID_BITS['Control_mode'].start
        return self.node_id | (self.CONTROL_MODES[control_mode] << shift)

    @staticmethod
    def _append_s32(buffer, value):
        buffer.extend(struct.pack('>i', int(value)))

    @staticmethod
    def _append_s16(buffer, value):
        buffer.extend(struct.pack('>h', int(value)))

    def set_duty(self, duty):
        payload = bytearray()
        self._append_s32(payload, duty * 100000.0)
        return can.Message(arbitration_id=self._build_arbitration_id('Duty_cycle_mode'),
                           data=payload,
                           is_extended_id=True)

    def set_current(self, current_a):
        payload = bytearray()
        self._append_s32(payload, current_a * 1000.0)
        return can.Message(arbitration_id=self._build_arbitration_id('Current_loop_mode'),
                           data=payload,
                           is_extended_id=True)

    def set_cb(self, current_a):
        payload = bytearray()
        self._append_s32(payload, current_a * 1000.0)
        return can.Message(arbitration_id=self._build_arbitration_id('Current_brake_mode'),
                           data=payload,
                           is_extended_id=True)

    def set_rpm(self, rpm_erpm):
        payload = bytearray()
        self._append_s32(payload, rpm_erpm)
        return can.Message(arbitration_id=self._build_arbitration_id('Velocity_mode'),
                           data=payload,
                           is_extended_id=True)

    def set_pos(self, pos_deg):
        payload = bytearray()
        self._append_s32(payload, pos_deg * 10000.0)  # manual: int32(pos*10000)
        return can.Message(arbitration_id=self._build_arbitration_id('Position_mode'),
                           data=payload,
                           is_extended_id=True)

    def set_origin(self, set_origin_mode):
        payload = bytearray([int(set_origin_mode) & 0xFF])  # manual: 1-byte mode
        return can.Message(arbitration_id=self._build_arbitration_id('Set_origin_mode'),
                           data=payload,
                           is_extended_id=True)

    def set_pos_spd(self, pos_deg, spd_erpm, acc_erpm_s2):
        payload = bytearray()
        self._append_s32(payload, pos_deg * 10000.0)
        self._append_s16(payload, spd_erpm / 10.0)
        self._append_s16(payload, acc_erpm_s2 / 10.0)
        return can.Message(arbitration_id=self._build_arbitration_id('Position_velocity_loop_mode'),
                           data=payload,
                           is_extended_id=True)

    def parse_feedback_8b(self, rx_message):
        """
        Parse 8-byte feedback payload:
        [pos_hi,pos_lo,spd_hi,spd_lo,cur_hi,cur_lo,temp,error]
        """
        if len(rx_message.data) < 8:
            raise ValueError(f'feedback length must be >=8, got {len(rx_message.data)}')
        data = rx_message.data
        pos_int = struct.unpack('>h', bytes(data[0:2]))[0]
        spd_int = struct.unpack('>h', bytes(data[2:4]))[0]
        cur_int = struct.unpack('>h', bytes(data[4:6]))[0]
        control_mode = (rx_message.arbitration_id >> self.CAN_ID_BITS['Control_mode'].start) & 0xFF
        return {
            'control_mode': int(control_mode),
            'motor_pos_deg': float(pos_int * 0.1),
            'motor_spd_erpm': float(spd_int * 10.0),
            'motor_cur_a': float(cur_int * 0.01),
            'motor_temp_c': int(data[6]),
            'motor_error': int(data[7]),
        }


class CubeMarsRLinkSerial:
    """R-Link serial protocol helper with optional serial session support."""

    COMM_PACKET_ID = {
        'COMM_GET_VALUES': 4,
        'COMM_SET_DUTY': 5,
        'COMM_SET_CURRENT': 6,
        'COMM_SET_CURRENT_BRAKE': 7,
        'COMM_SET_RPM': 8,
        'COMM_SET_POS': 9,
        'COMM_ROTOR_POSITION': 22,
        'COMM_GET_VALUES_SETUP': 50,
        'COMM_SET_POS_SPD': 91,
        'COMM_SET_POS_ORIGIN': 95,
    }

    def __init__(self, port='COM3', baudrate=921600, timeout=0.02):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self._ser = None

    @staticmethod
    def crc16(data):
        cksum = 0
        for b in data:
            c = ((cksum >> 8) ^ int(b)) & 0xFF
            cksum = ((cksum << 8) & 0xFFFF) ^ CubeMarsRLinkSerial._crc16_byte(c)
        return cksum

    @staticmethod
    def _crc16_byte(c):
        c <<= 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if (c & 0x8000) else ((c << 1) & 0xFFFF)
        return c

    @classmethod
    def build_packet(cls, cmd_id, payload=b''):
        body = bytes([int(cmd_id) & 0xFF]) + bytes(payload)
        if len(body) > 255:
            raise ValueError(f'packet too long: {len(body)}')
        crc = cls.crc16(body)
        return bytes([0x02, len(body)]) + body + bytes([(crc >> 8) & 0xFF, crc & 0xFF, 0x03])

    @classmethod
    def comm_get_values(cls):
        return cls.build_packet(cls.COMM_PACKET_ID['COMM_GET_VALUES'])

    @classmethod
    def comm_get_values_setup_all(cls):
        return cls.build_packet(cls.COMM_PACKET_ID['COMM_GET_VALUES_SETUP'], payload=b'\xFF\xFF\xFF\xFF')

    @classmethod
    def comm_get_rotor_position_stream(cls):
        # manual example: 02 02 0B 04 9C 7E 03
        return cls.build_packet(0x0B, payload=b'\x04')

    @classmethod
    def comm_set_pos(cls, pos_deg):
        # R-Link serial protocol uses int32(pos * 1e6) for COMM_SET_POS.
        payload = struct.pack('>i', int(pos_deg * 1000000.0))
        return cls.build_packet(cls.COMM_PACKET_ID['COMM_SET_POS'], payload=payload)

    @classmethod
    def comm_set_pos_spd(cls, pos_deg, spd_erpm, acc_erpm_s2):
        # R-Link serial variant: int32(pos*1e6), int32(spd), int32(acc).
        payload = struct.pack('>iii',
                              int(pos_deg * 1000000.0),
                              int(spd_erpm),
                              int(acc_erpm_s2))
        return cls.build_packet(cls.COMM_PACKET_ID['COMM_SET_POS_SPD'], payload=payload)

    @classmethod
    def comm_set_pos_origin(cls, mode):
        payload = bytes([int(mode) & 0xFF])
        return cls.build_packet(cls.COMM_PACKET_ID['COMM_SET_POS_ORIGIN'], payload=payload)

    @staticmethod
    def comm_power_on_raw():
        # Widely used R-Link startup sequence from vendor examples.
        return bytes([0x40, 0x80, 0x20, 0x02, 0x21, 0xC0])

    @classmethod
    def parse_packet(cls, frame):
        """
        Parse one full frame in format:
        0x02 + length + body + crc_hi + crc_lo + 0x03
        Returns body bytes (cmd + payload) on success, otherwise None.
        """
        if len(frame) < 6:
            return None
        if frame[0] != 0x02 or frame[-1] != 0x03:
            return None
        length = int(frame[1])
        if len(frame) != length + 5:
            return None
        body = bytes(frame[2:2 + length])
        crc_rx = (int(frame[2 + length]) << 8) | int(frame[3 + length])
        if cls.crc16(body) != crc_rx:
            return None
        return body

    def open(self):
        if serial is None:
            raise ImportError('pyserial is required for CubeMarsRLinkSerial session methods')
        self._ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        return self

    def close(self):
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def is_open(self):
        return self._ser is not None

    def send(self, frame):
        if self._ser is None:
            raise RuntimeError('CubeMarsRLinkSerial is not open')
        self._ser.write(frame)
        self._ser.flush()

    def read_available(self):
        if self._ser is None:
            raise RuntimeError('CubeMarsRLinkSerial is not open')
        n = self._ser.in_waiting
        if n <= 0:
            return b''
        return bytes(self._ser.read(n))

    def transact(self, frame, probe_time_s=0.2):
        self.send(frame)
        end_t = time.time() + float(probe_time_s)
        rx = bytearray()
        while time.time() < end_t:
            chunk = self.read_available()
            if chunk:
                rx.extend(chunk)
            time.sleep(0.01)
        return bytes(rx)

    def startup(self, probe_time_s=0.2):
        return self.transact(CubeMarsRLinkSerial.comm_power_on_raw(), probe_time_s=probe_time_s)

    def set_pos(self, pos_deg, probe_time_s=0.2):
        frame = CubeMarsRLinkSerial.comm_set_pos(pos_deg)
        return self.transact(frame, probe_time_s=probe_time_s)

    def set_pos_spd(self, pos_deg, spd_erpm, acc_erpm_s2, probe_time_s=0.2):
        frame = CubeMarsRLinkSerial.comm_set_pos_spd(pos_deg, spd_erpm=spd_erpm, acc_erpm_s2=acc_erpm_s2)
        return self.transact(frame, probe_time_s=probe_time_s)


# Backward compatible aliases
Motor = CubeMarsNativeCAN
RLink = CubeMarsRLinkSerial
    
