"""Interactive IK on the L1 left arm (6-DOF analytic S456X12, waist frozen).

A movable target frame (default orientation: tcp z pointing DOWN, x pointing
FORWARD) is driven by the keyboard; each frame we solve IK for the hand's
grasp-center tcp and snap the robot onto a solution. The robot's tcp frame
(MYC) overlays the target (RGB) when a solution is found.

Keys (hold to move continuously, world frame):
  translate: Q/A = +x/-x (forward/back)   W/S = +y/-y   E/D = +z/-z
  rotate   : Y/H = +/- about X   U/J = +/- about Y   I/K = +/- about Z
  N : cycle to the next IK branch   R : reset target   (Esc closes)
"""
import os
import sys
import builtins

import numpy as np
import pyglet.window.key as key

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.math as oum                                   # noqa: E402
import one.utils.constant as ouc                               # noqa: E402
import one.scene.scene_object_primitive as ossop               # noqa: E402
import one.viewer.world as ovw                                 # noqa: E402
import one.robots.humanoids.linx.l1.l1 as l1                   # noqa: E402

CHAIN = 'left_arm'
TCP_NAME = 'pinch_center'
# default target: z down (approach from above), x forward
DEFAULT_POS = np.array([0.15, 0.12, 1.00], dtype=np.float32)
DEFAULT_ROT = np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]], dtype=np.float32)
POS_STEP = 0.004              # m per tick while held
ROT_STEP = np.radians(1.5)    # rad per tick while held


def main():
    robot = l1.L1O6()
    tcp = robot.left_hand.tcp(TCP_NAME)
    home = robot.qs.copy()

    base = ovw.World(cam_pos=(1.6, -0.5, 1.4), cam_lookat_pos=(0.2, 0.1, 1.0))
    builtins.base = base
    ossop.frame().attach_to(base.scene)                 # world frame
    ossop.plane(pos=(0, 0, 0.0)).attach_to(base.scene)
    robot.attach_to(base.scene)

    tgt = {"pos": DEFAULT_POS.copy(), "rot": DEFAULT_ROT.copy(), "sol": 0}
    tgt_frame = ossop.frame(pos=tgt["pos"], rotmat=tgt["rot"], length_scale=0.5)
    tgt_frame.attach_to(base.scene)
    tcp_frame = ossop.frame(length_scale=0.5, color_mat=ouc.CoordColor.MYC)
    tcp_frame.attach_to(base.scene)

    def solve_and_show():
        sols = robot.ik(tgt["pos"], tgt["rot"], chain=CHAIN, tcp=tcp,
                        ref_qs=robot.chain(CHAIN).extract_active_qs(robot.qs),
                        max_solutions=8)
        tgt_frame.set_pos_rotmat(pos=tgt["pos"], rotmat=tgt["rot"])
        if sols:
            i = tgt["sol"] % len(sols)
            robot.fk(qs=np.asarray(sols[i], dtype=np.float32))
            tf = tcp.tf
            tcp_frame.set_pos_rotmat(pos=tf[:3, 3], rotmat=tf[:3, :3])
            status = f"IK: {len(sols)} sols, showing #{i}"
        else:
            tcp_frame.set_pos_rotmat(pos=(0, 0, -1), rotmat=np.eye(3))  # hide
            status = "IK: no solution"
        rpy = np.degrees(oum.euler_from_rotmat(tgt["rot"]))
        base.set_caption(
            f"pos=({tgt['pos'][0]:.2f},{tgt['pos'][1]:.2f},{tgt['pos'][2]:.2f}) "
            f"rpy=({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})   {status}")
        base.scene.dirty = True

    solve_and_show()

    def tick(dt):
        im = base.input_manager
        moved = False
        dp = np.zeros(3, dtype=np.float32)
        for k, ax, sgn in ((key.Q, 0, 1), (key.A, 0, -1),
                           (key.W, 1, 1), (key.S, 1, -1),
                           (key.E, 2, 1), (key.D, 2, -1)):
            if im.is_key_pressed(k):
                dp[ax] += sgn * POS_STEP
                moved = True
        if moved:
            tgt["pos"] = tgt["pos"] + dp
        for k, axis, sgn in ((key.Y, ouc.StandardAxis.X, 1),
                             (key.H, ouc.StandardAxis.X, -1),
                             (key.U, ouc.StandardAxis.Y, 1),
                             (key.J, ouc.StandardAxis.Y, -1),
                             (key.I, ouc.StandardAxis.Z, 1),
                             (key.K, ouc.StandardAxis.Z, -1)):
            if im.is_key_pressed(k):
                # rotate about WORLD axis -> pre-multiply
                tgt["rot"] = (oum.rotmat_from_axangle(axis, sgn * ROT_STEP)
                              @ tgt["rot"]).astype(np.float32)
                moved = True
        if im.is_key_pressed_edge(key.N):
            tgt["sol"] += 1
            moved = True
        if im.is_key_pressed_edge(key.R):
            tgt["pos"] = DEFAULT_POS.copy()
            tgt["rot"] = DEFAULT_ROT.copy()
            tgt["sol"] = 0
            robot.fk(qs=home)
            moved = True
        if moved:
            solve_and_show()

    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == "__main__":
    main()
