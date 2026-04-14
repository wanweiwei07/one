import builtins
import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.scene.scene_object_primitive as ossop
import one.robots.manipulators.universal_robots.ur3.ur3 as ormuu3
import one.robots.end_effectors.onrobot.or_2fg7.or_2fg7 as oreo2fg7
import one.robots.end_effectors.openarm_gripper.oa_gripper as oreeogog


if __name__ == "__main__":
    base = ovw.World(cam_pos=(1.6, 1.0, 1.4), cam_lookat_pos=(0.0, 0.0, 0.45))
    builtins.base = base
    scene = base.scene
    ossop.frame().attach_to(scene)

    robot = ormuu3.UR3()
    robot.attach_to(scene)
    builtins.robot = robot

    # gripper = oreo2fg7.OR2FG7()
    gripper = oreeogog.OAGripper()
    gripper.set_jaw_width(0.03)
    gripper.attach_to(scene)

    # engage_tf is flange->ee_base transform
    engage_tf = oum.tf_from_rotmat_pos(
        rotmat=np.eye(3, dtype=np.float32),
        pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    robot.engage(gripper, engage_tf=engage_tf)

    tgt_pos = np.array([0.35, -0.20, 0.2], dtype=np.float32)
    tgt_rotmat = (
        oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 6.0)
        @ oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi)
    )
    ossop.frame(pos=tgt_pos, rotmat=tgt_rotmat, color_mat=ouc.CoordColor.DYO).attach_to(scene)

    qs = robot.ik_tcp_nearest(tgt_rotmat=tgt_rotmat, tgt_pos=tgt_pos)
    print("ik:", qs)
    if qs is not None:
        robot.fk(qs=qs)
        ossop.frame(
            pos=robot.gl_tcp_tf[:3, 3],
            rotmat=robot.gl_tcp_tf[:3, :3],
            color_mat=ouc.CoordColor.MYC,
        ).attach_to(scene)

    base.run()
