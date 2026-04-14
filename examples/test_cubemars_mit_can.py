import argparse
import time

import can


def _clip(val, lo, hi):
    return max(lo, min(hi, val))


def _float_to_uint(x, x_min, x_max, bits):
    span = float(x_max - x_min)
    if span <= 0.0:
        raise ValueError('invalid range')
    x = _clip(float(x), float(x_min), float(x_max))
    return int((x - x_min) * ((1 << bits) - 1) / span)


def _pack_mit_cmd(p, v, kp, kd, t_ff,
                  p_min=-12.5, p_max=12.5,
                  v_min=-45.0, v_max=45.0,
                  kp_min=0.0, kp_max=500.0,
                  kd_min=0.0, kd_max=5.0,
                  t_min=-18.0, t_max=18.0):
    p_u = _float_to_uint(p, p_min, p_max, 16)
    v_u = _float_to_uint(v, v_min, v_max, 12)
    kp_u = _float_to_uint(kp, kp_min, kp_max, 12)
    kd_u = _float_to_uint(kd, kd_min, kd_max, 12)
    t_u = _float_to_uint(t_ff, t_min, t_max, 12)
    data = bytearray(8)
    data[0] = (p_u >> 8) & 0xFF
    data[1] = p_u & 0xFF
    data[2] = (v_u >> 4) & 0xFF
    data[3] = ((v_u & 0x0F) << 4) | ((kp_u >> 8) & 0x0F)
    data[4] = kp_u & 0xFF
    data[5] = (kd_u >> 4) & 0xFF
    data[6] = ((kd_u & 0x0F) << 4) | ((t_u >> 8) & 0x0F)
    data[7] = t_u & 0xFF
    return data


def _fmt_data(data):
    return ' '.join(f'{b:02X}' for b in data)


def _send(bus, node_id, data, ext=False):
    msg = can.Message(arbitration_id=int(node_id), data=data, is_extended_id=ext)
    bus.send(msg)
    print(f'TX id=0x{msg.arbitration_id:X} ext={msg.is_extended_id} data=[{_fmt_data(msg.data)}]')


def _probe_rx(bus, timeout_s=0.3):
    t0 = time.time()
    n = 0
    while time.time() - t0 < timeout_s:
        rx = bus.recv(timeout=0.02)
        if rx is None:
            continue
        n += 1
        print(f'  RX id=0x{rx.arbitration_id:X} ext={rx.is_extended_id} data=[{_fmt_data(rx.data)}]')
    if n == 0:
        print('  RX none')


def run_test(port, bustype, bitrate, node_id, cycles, hold, probe):
    print(f'Opening CAN bus: bustype={bustype}, channel={port}, bitrate={bitrate}')
    with can.Bus(interface=bustype, channel=port, bitrate=bitrate) as bus:
        print('Enter MIT mode')
        _send(bus, node_id, bytes([0xFF] * 7 + [0xFC]), ext=False)
        _probe_rx(bus, timeout_s=probe)

        cmd_a = _pack_mit_cmd(p=0.5, v=0.0, kp=20.0, kd=1.0, t_ff=0.0)
        cmd_b = _pack_mit_cmd(p=-0.5, v=0.0, kp=20.0, kd=1.0, t_ff=0.0)
        cmd_0 = _pack_mit_cmd(p=0.0, v=0.0, kp=10.0, kd=1.0, t_ff=0.0)

        for i in range(cycles):
            print(f'Cycle {i + 1}/{cycles}: +0.5 rad')
            _send(bus, node_id, cmd_a, ext=False)
            _probe_rx(bus, timeout_s=probe)
            time.sleep(hold)

            print(f'Cycle {i + 1}/{cycles}: -0.5 rad')
            _send(bus, node_id, cmd_b, ext=False)
            _probe_rx(bus, timeout_s=probe)
            time.sleep(hold)

        print('Back to 0 rad')
        _send(bus, node_id, cmd_0, ext=False)
        _probe_rx(bus, timeout_s=probe)
        time.sleep(hold)

        print('Exit MIT mode')
        _send(bus, node_id, bytes([0xFF] * 7 + [0xFD]), ext=False)
        _probe_rx(bus, timeout_s=probe)


def main():
    parser = argparse.ArgumentParser(description='Cubemars MIT mode CAN probe on COM port.')
    parser.add_argument('--port', default='COM3')
    parser.add_argument('--bustype', default='slcan')
    parser.add_argument('--bitrate', type=int, default=1000000)
    parser.add_argument('--node-id', type=int, default=1)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--hold', type=float, default=0.6)
    parser.add_argument('--probe', type=float, default=0.3)
    args = parser.parse_args()
    run_test(args.port, args.bustype, args.bitrate, args.node_id, args.cycles, args.hold, args.probe)


if __name__ == '__main__':
    main()
