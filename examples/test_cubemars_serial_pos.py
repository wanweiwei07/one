import argparse
import struct
import time

import serial


CRC16_TAB = (
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
    0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
)


def crc16(buf):
    cksum = 0
    for b in buf:
        idx = ((cksum >> 8) ^ b) & 0xFF
        cksum = ((cksum << 8) & 0xFFFF) ^ _crc16_tab(idx)
    return cksum


def _crc16_tab(i):
    if i < 32:
        return CRC16_TAB[i]
    # Extend table on demand using polynomial 0x1021
    c = i << 8
    for _ in range(8):
        c = ((c << 1) ^ 0x1021) & 0xFFFF if (c & 0x8000) else ((c << 1) & 0xFFFF)
    return c


def build_frame(cmd_id, payload=b''):
    body = bytes([cmd_id]) + payload
    length = len(body)
    cksum = crc16(body)
    return bytes([0x02, length]) + body + bytes([(cksum >> 8) & 0xFF, cksum & 0xFF, 0x03])


def build_set_pos_frame(pos_deg):
    # Manual: COMM_SET_POS uses int32(pos * 10000.0)
    payload = struct.pack('>i', int(pos_deg * 10000.0))
    return build_frame(0x09, payload)


def build_set_pos_spd_frame(pos_deg, spd_elec_rpm=2000, acc_elec_rpm_s2=2000):
    # Manual (5.1.6): COMM_SET_POS_SPD=91, payload=int32(pos*10000) + int16(spd/10) + int16(acc/10)
    payload = struct.pack(
        '>iHH',
        int(pos_deg * 10000.0),
        int(spd_elec_rpm / 10.0) & 0xFFFF,
        int(acc_elec_rpm_s2 / 10.0) & 0xFFFF,
    )
    return build_frame(0x5B, payload)


def send_and_log(ser, frame, probe_s=0.15):
    ser.write(frame)
    ser.flush()
    print('TX', frame.hex(' '))
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < probe_s:
        n = ser.in_waiting
        if n:
            buf.extend(ser.read(n))
        time.sleep(0.01)
    if buf:
        print('RX', buf.hex(' '))
    else:
        print('RX none')


def run_test(port, baudrate, cycles, hold_s, use_pos_spd, spd, acc):
    print(f'Open serial: {port} @ {baudrate}')
    with serial.Serial(port, baudrate=baudrate, timeout=0.02) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.05)

        if use_pos_spd:
            cmd = lambda deg: build_set_pos_spd_frame(deg, spd_elec_rpm=spd, acc_elec_rpm_s2=acc)
        else:
            cmd = build_set_pos_frame

        send_and_log(ser, cmd(0.0))
        time.sleep(0.2)
        for i in range(cycles):
            print(f'Cycle {i + 1}/{cycles}: +30 deg')
            send_and_log(ser, cmd(30.0))
            time.sleep(hold_s)
            print(f'Cycle {i + 1}/{cycles}: -30 deg')
            send_and_log(ser, cmd(-30.0))
            time.sleep(hold_s)
        print('Back to 0 deg')
        send_and_log(ser, cmd(0.0))
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description='Cubemars serial COMM_SET_POS test (+30/-30 deg).')
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--baudrate', type=int, default=921600)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--hold', type=float, default=1.0)
    parser.add_argument('--use-pos-spd', action='store_true', help='Use COMM_SET_POS_SPD (91) instead of COMM_SET_POS (9)')
    parser.add_argument('--spd', type=int, default=3000, help='Electrical rpm for POS_SPD mode')
    parser.add_argument('--acc', type=int, default=3000, help='Electrical rpm/s^2 for POS_SPD mode')
    args = parser.parse_args()
    run_test(args.port, args.baudrate, args.cycles, args.hold, args.use_pos_spd, args.spd, args.acc)


if __name__ == '__main__':
    main()
