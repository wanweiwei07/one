"""Play back a recorded L1O6 joint-angle sequence in the viewer.

The motion is given as a small time-stamped table of joint angles (radians) for
a subset of the robot's joints -- the waist, both 6-DoF arms and the 2-DoF neck;
every other joint (hands) stays at home. We map each named column to its joint
index in the compiled mechanism, then drive the robot through the keyframes by
linearly interpolating in joint space against the wall clock (looping).

Run headless validation (no window):  set ONE_HEADLESS=1
"""
import os

import numpy as np

import one.viewer.world as ovw
import one.scene.scene_object_primitive as ossop
import one.robots.humanoids.linx.l1.l1 as l1

# Column order of the recorded table (joint angles in radians).
COLUMNS = ['waist2',
           'L1', 'L2', 'L3', 'L4', 'L5', 'L6',
           'R1', 'R2', 'R3', 'R4', 'R5', 'R6',
           'nk1', 'nk2']

# Recorded keyframes: (t_seconds, [angle per COLUMNS]).
KEYFRAMES = [
    (0.000, [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (0.129, [0.000, 0.000, 0.013, -0.017, 0.000, 0.000, 0.000, 0.000, -0.013, 0.017, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (0.257, [0.000, 0.000, 0.053, -0.066, 0.000, 0.000, 0.000, 0.000, -0.053, 0.066, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (0.386, [0.000, 0.000, 0.119, -0.149, 0.000, 0.000, 0.000, 0.000, -0.119, 0.149, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (0.514, [0.000, 0.000, 0.212, -0.264, 0.000, 0.000, 0.000, 0.000, -0.212, 0.264, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (0.643, [0.000, 0.000, 0.331, -0.413, 0.000, 0.000, 0.000, 0.000, -0.331, 0.413, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (0.771, [0.000, 0.000, 0.469, -0.587, 0.000, 0.000, 0.000, 0.000, -0.469, 0.587, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (0.900, [0.000, 0.000, 0.588, -0.736, 0.000, 0.000, 0.000, 0.000, -0.588, 0.736, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (1.029, [0.000, 0.000, 0.681, -0.851, 0.000, 0.000, 0.000, 0.000, -0.681, 0.851, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (1.157, [0.000, 0.000, 0.747, -0.934, 0.000, 0.000, 0.000, 0.000, -0.747, 0.934, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (1.286, [0.000, 0.000, 0.787, -0.983, 0.000, 0.000, 0.000, 0.000, -0.787, 0.983, 0.000, 0.000, 0.000, 0.000, 0.000]),
    (1.414, [0.000, 0.000, 0.800, -1.000, 0.000, 0.000, 0.000, 0.000, -0.800, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000]),
]


def column_to_jntidx(robot):
    """Map each table column name to its joint index in the compiled mechanism.

    The named chains expose their active joint ids in order, so the arms/neck map
    directly; the waist column is the lone proximal waist joint of an arm+waist
    chain (the joint that is in 'left_arm_waist' but not in 'left_arm')."""
    left = list(robot.chain('left_arm').active_jnt_ids)         # L1..L6
    right = list(robot.chain('right_arm').active_jnt_ids)       # R1..R6
    neck = list(robot.chain('neck').active_jnt_ids)             # nk1, nk2
    waist = [j for j in robot.chain('left_arm_waist').active_jnt_ids
             if j not in left][0]                               # waist2
    mapping = {'waist2': int(waist)}
    for i, j in enumerate(left):
        mapping[f'L{i + 1}'] = int(j)
    for i, j in enumerate(right):
        mapping[f'R{i + 1}'] = int(j)
    mapping['nk1'], mapping['nk2'] = int(neck[0]), int(neck[1])
    return mapping


def build_qs_keyframes(robot):
    """Turn the table into full-length qs vectors (home for untouched joints)."""
    col2idx = column_to_jntidx(robot)
    idxs = np.array([col2idx[c] for c in COLUMNS])
    home = robot.qs.astype(np.float64)
    times = np.array([t for t, _ in KEYFRAMES])
    qs = []
    for _t, row in KEYFRAMES:
        q = home.copy()
        q[idxs] = np.asarray(row, np.float64)
        qs.append(q)
    return times, np.array(qs)


def sample(times, qs, t):
    """Linear interpolation of the keyframe configurations at time ``t``."""
    if t <= times[0]:
        return qs[0]
    if t >= times[-1]:
        return qs[-1]
    k = int(np.searchsorted(times, t) - 1)
    a = (t - times[k]) / (times[k + 1] - times[k])
    return (1 - a) * qs[k] + a * qs[k + 1]


def main():
    robot = l1.L1O6()
    times, qs = build_qs_keyframes(robot)

    if os.environ.get('ONE_HEADLESS'):
        col2idx = column_to_jntidx(robot)
        print('column -> joint index:', col2idx)
        moved = np.where(np.abs(qs.max(0) - qs.min(0)) > 1e-6)[0]
        expected = sorted(col2idx[c] for c in COLUMNS
                          if abs(max(r[COLUMNS.index(c)] for _, r in KEYFRAMES)
                                 - min(r[COLUMNS.index(c)] for _, r in KEYFRAMES)) > 1e-6)
        print('joints that move:', sorted(moved.tolist()), '(expected', expected, ')')
        robot.fk(qs=qs[-1])   # final pose must be valid
        print(f'headless OK: {len(KEYFRAMES)} keyframes over {times[-1]:.3f}s')
        return

    base = ovw.World(cam_pos=(2.2, 1.4, 1.6), cam_lookat_pos=(0.0, 0.0, 0.9))
    ossop.frame().attach_to(base.scene)
    ossop.plane(pos=(0, 0, 0.0)).attach_to(base.scene)
    robot.attach_to(base.scene)
    robot.fk(qs=qs[0])

    period = float(times[-1])
    hold = 0.6   # pause at the end before looping
    state = {'t': 0.0}
    print(f'playing {len(KEYFRAMES)} keyframes ({period:.3f}s), looping')

    def tick(dt):
        state['t'] += dt
        if state['t'] > period + hold:
            state['t'] = 0.0
        robot.fk(qs=sample(times, qs, state['t']))

    base.schedule_interval(tick, interval=0.02)
    base.run()


if __name__ == '__main__':
    main()
