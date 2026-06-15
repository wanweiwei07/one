"""Draw the grasp TARGET frames and the robot's grasp-center TCP for the
cylinder at l1picking's current CYL_POS -- to eyeball why nothing is reachable.

    RGB small frames   = the grasp targets (where the hand's grasp-center must
                         land), one per loaded grasp; pre-grasp frames are faint.
    MYC big frame      = the robot's grasp-center tcp at HOME (the frame IK
                         tries to drive onto each target).

Robot stays at home; no planning. So the visual question is just: are the RGB
target frames anywhere near where the MYC tcp frame can sweep?
"""
import os
import sys
import builtins

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import one.utils.constant as ouc                               # noqa: E402
import one.scene.scene_object_primitive as ossop               # noqa: E402
import one.robots.base.tcp as orbt                             # noqa: E402
import one.viewer.world as ovw                                 # noqa: E402

from l1picking import build_scene, load_world_grasps           # noqa: E402


def main():
    robot, table, cyl, ground = build_scene()
    grasps = load_world_grasps(cyl)
    jaw = robot.left_hand.spawn_jaw('pinch')
    print(f"{len(grasps)} grasps; cylinder at {np.round(cyl.pos, 3)}")

    base = ovw.World(cam_pos=(1.5, -0.4, 1.4), cam_lookat_pos=(0.15, 0.2, 1.0))
    builtins.base = base
    ossop.frame().attach_to(base.scene)               # world frame
    for e in (robot, *table, cyl, ground):
        e.attach_to(base.scene)

    # grasp target frames (RGB), pre-grasp frames faint
    for pose, pre, jw, sc in grasps:
        ossop.frame(pos=pose[:3, 3], rotmat=pose[:3, :3],
                    length_scale=0.22).attach_to(base.scene)
        f = ossop.frame(pos=pre[:3, 3], rotmat=pre[:3, :3], length_scale=0.16)
        f.alpha = 0.3
        f.attach_to(base.scene)

    # robot's grasp-center tcp at HOME (the IK target frame), drawn in MYC
    jw0 = float(jaw.jaw_range[1]) * 0.5
    center_tcp = orbt.TCP(robot.left_hand.runtime_root_lnk,
                          jaw.eval_grasp_tcp(jw0).loc_tf)
    tf = center_tcp.tf
    print("robot grasp-center tcp (home) pos:", np.round(tf[:3, 3], 3))
    ossop.frame(pos=tf[:3, 3], rotmat=tf[:3, :3], length_scale=0.6,
                color_mat=ouc.CoordColor.MYC).attach_to(base.scene)

    base.run()


if __name__ == "__main__":
    main()
