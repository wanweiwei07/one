import argparse
import time

import can

from one.devices.cubemars import Motor, RLink


def _hex(bs):
    return ' '.join(f'{b:02X}' for b in bs)


def _send_and_probe_serial(session, frame, probe_time_s=0.2):
    print(f'TX [{_hex(frame)}]')
    rx = session.transact(frame, probe_time_s=probe_time_s)
    if rx:
        print(f'  RX [{_hex(rx)}]')
    else:
        print('  RX none')


def _send_and_probe_can(bus, msg, probe_time_s=0.2):
    bus.send(msg)
    print(f'TX id=0x{msg.arbitration_id:X} dlc={len(msg.data)} data=[{_hex(msg.data)}]')
    t0 = time.time()
    got = 0
    while time.time() - t0 < probe_time_s:
        rx = bus.recv(timeout=0.02)
        if rx is None:
            continue
        got += 1
        print(f'  RX id=0x{rx.arbitration_id:X} dlc={len(rx.data)} data=[{_hex(rx.data)}]')
    if got == 0:
        print('  RX none')


def run_swing_test_serial(port, baudrate, cycles, hold_s, settle_s, probe_s, use_pos_spd, spd, acc, amplitude, forever):
    print(f'Opening R-Link serial: port={port}, baudrate={baudrate}')
    with RLink(port=port, baudrate=baudrate, timeout=0.02) as session:
        print('Serial opened, sending startup...')
        startup_frame = RLink.comm_power_on_raw()
        _send_and_probe_serial(session, startup_frame, probe_time_s=probe_s)
        time.sleep(0.1)
        print('Sending position commands...')

        def cmd(deg):
            if use_pos_spd:
                return RLink.comm_set_pos_spd(deg, spd_erpm=spd, acc_erpm_s2=acc)
            return RLink.comm_set_pos(deg)

        _send_and_probe_serial(session, cmd(0.0), probe_time_s=probe_s)
        time.sleep(settle_s)
        i = 0
        while forever or i < cycles:
            if forever:
                print(f'Cycle {i + 1}: +{amplitude:.1f} deg')
            else:
                print(f'Cycle {i + 1}/{cycles}: +{amplitude:.1f} deg')
            _send_and_probe_serial(session, cmd(amplitude), probe_time_s=probe_s)
            time.sleep(hold_s)
            if forever:
                print(f'Cycle {i + 1}: -{amplitude:.1f} deg')
            else:
                print(f'Cycle {i + 1}/{cycles}: -{amplitude:.1f} deg')
            _send_and_probe_serial(session, cmd(-amplitude), probe_time_s=probe_s)
            time.sleep(hold_s)
            i += 1
        print('Returning to 0 deg')
        _send_and_probe_serial(session, cmd(0.0), probe_time_s=probe_s)
        time.sleep(settle_s)
    print('Done.')


def run_swing_test_native_can(can_interface,
                              can_channel,
                              can_bitrate,
                              node_id,
                              cycles,
                              hold_s,
                              settle_s,
                              probe_s,
                              use_pos_spd,
                              spd,
                              acc,
                              amplitude,
                              forever):
    print(f'Opening native CAN: interface={can_interface}, channel={can_channel}, bitrate={can_bitrate}')
    motor = Motor(node_id=node_id)
    with can.Bus(interface=can_interface, channel=can_channel, bitrate=can_bitrate) as bus:
        print('CAN bus opened, sending commands...')

        def cmd(deg):
            if use_pos_spd:
                return motor.set_pos_spd(deg, spd_erpm=spd, acc_erpm_s2=acc)
            return motor.set_pos(deg)

        _send_and_probe_can(bus, cmd(0.0), probe_time_s=probe_s)
        time.sleep(settle_s)
        i = 0
        while forever or i < cycles:
            if forever:
                print(f'Cycle {i + 1}: +{amplitude:.1f} deg')
            else:
                print(f'Cycle {i + 1}/{cycles}: +{amplitude:.1f} deg')
            _send_and_probe_can(bus, cmd(amplitude), probe_time_s=probe_s)
            time.sleep(hold_s)
            if forever:
                print(f'Cycle {i + 1}: -{amplitude:.1f} deg')
            else:
                print(f'Cycle {i + 1}/{cycles}: -{amplitude:.1f} deg')
            _send_and_probe_can(bus, cmd(-amplitude), probe_time_s=probe_s)
            time.sleep(hold_s)
            i += 1
        print('Returning to 0 deg')
        _send_and_probe_can(bus, cmd(0.0), probe_time_s=probe_s)
        time.sleep(settle_s)
    print('Done.')


def main():
    parser = argparse.ArgumentParser(description='Cubemars swing test (R-Link serial or native CAN).')
    parser.add_argument('--transport',
                        default='native_can',
                        choices=['native_can', 'rlink_serial'],
                        help='Communication backend')
    parser.add_argument('--port', default='COM3', help='R-Link serial port, e.g. COM3')
    parser.add_argument('--baudrate', type=int, default=921600, help='R-Link serial baudrate')
    parser.add_argument('--can-interface',
                        default='gs_usb',
                        help='python-can interface, e.g. gs_usb/slcan/socketcan/pcan')
    parser.add_argument('--can-channel',
                        default='0',
                        help='CAN channel name/index used by selected interface')
    parser.add_argument('--can-bitrate', type=int, default=1000000, help='Native CAN bitrate')
    parser.add_argument('--node-id', type=int, default=1, help='CubeMars node ID')
    parser.add_argument('--cycles', type=int, default=5, help='Number of +30/-30 cycles')
    parser.add_argument('--hold', type=float, default=1.0, help='Hold time per target in seconds')
    parser.add_argument('--settle', type=float, default=0.5, help='Settle time at start/end in seconds')
    parser.add_argument('--probe', type=float, default=0.15, help='RX probe window after each TX (seconds)')
    parser.add_argument('--use-pos-spd', action='store_true', help='Use COMM_SET_POS_SPD (recommended)')
    parser.add_argument('--spd', type=int, default=3000, help='ERPM for COMM_SET_POS_SPD')
    parser.add_argument('--acc', type=int, default=3000, help='ERPM/s^2 for COMM_SET_POS_SPD')
    parser.add_argument('--amplitude', type=float, default=30.0, help='Swing amplitude in degrees')
    parser.add_argument('--forever', action='store_true', help='Loop forever until Ctrl+C')
    args = parser.parse_args()

    if args.transport == 'rlink_serial':
        run_swing_test_serial(
            port=args.port,
            baudrate=args.baudrate,
            cycles=args.cycles,
            hold_s=args.hold,
            settle_s=args.settle,
            probe_s=args.probe,
            use_pos_spd=args.use_pos_spd,
            spd=args.spd,
            acc=args.acc,
            amplitude=args.amplitude,
            forever=args.forever,
        )
    else:
        run_swing_test_native_can(
            can_interface=args.can_interface,
            can_channel=args.can_channel,
            can_bitrate=args.can_bitrate,
            node_id=args.node_id,
            cycles=args.cycles,
            hold_s=args.hold,
            settle_s=args.settle,
            probe_s=args.probe,
            use_pos_spd=args.use_pos_spd,
            spd=args.spd,
            acc=args.acc,
            amplitude=args.amplitude,
            forever=args.forever,
        )


if __name__ == '__main__':
    main()
