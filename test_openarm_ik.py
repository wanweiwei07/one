import builtins
from one import ouc, oum, ovw, ossop
import one.robots.end_effectors.openarm_gripper.oa_gripper as oreeogog
import one.robots.manipulators.openarm.openarm as ormoo

base = ovw.World(cam_pos=(1.2, .575, 1.2), cam_lookat_pos=(0, 0, .4),
                 toggle_auto_cam_orbit=True)
oframe = ossop.frame().attach_to(base.scene)
robot = ormoo.OpenArm()
robot.attach_to(base.scene)
builtins.robot = robot  # for debug access
builtins.base = base
robot.rgt_arm.fk((0, 0, 0, 0, 0, 0, 0))
robot.lft_arm.fk((0, 0, 0, 0, 0, 0, 0))
lft_gripper = oreeogog.OAGripper()
rgt_gripper = oreeogog.OAGripper()
lft_gripper.attach_to(base.scene)
rgt_gripper.attach_to(base.scene)
robot.rgt_arm.engage(rgt_gripper)
robot.lft_arm.engage(lft_gripper)
lft_tcp_tf = robot.lft_arm.gl_tcp_tf
lft_tcp_frame = ossop.frame(rotmat=lft_tcp_tf[:3, :3], pos=lft_tcp_tf[:3, 3],
                            color_mat=ouc.CoordColor.MYC)
lft_tcp_frame.attach_to(base.scene)
rgt_tcp_tf = robot.rgt_arm.gl_tcp_tf
rgt_tcp_frame = ossop.frame(rotmat=rgt_tcp_tf[:3, :3], pos=rgt_tcp_tf[:3, 3],
                            color_mat=ouc.CoordColor.MYC)
rgt_tcp_frame.attach_to(base.scene)
# robot.body.alpha=0.3
# base.run()

tgt_pos = (0.4, 0.1, 0.4)
# tgt_rotmat = (oum.rotmat_from_axangle(ouc.StandardAxis.X, oum.pi / 2) @
#               oum.rotmat_from_axangle(ouc.StandardAxis.Y, oum.pi / 2))
tgt_rotmat = (oum.rotmat_from_axangle(ouc.StandardAxis.Z, 0) @
              oum.rotmat_from_axangle(ouc.StandardAxis.Y, oum.pi))
ossop.frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base.scene)

prev_qs = robot.lft_arm.qs.copy()
for y in range(1, 5):
    for z in range(1, 5):
        tgt_pos = (0.2, y * 0.1, z * 0.1)
        qs = robot.lft_arm.ik_tcp_nearest(
            tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, ref_qs=prev_qs)
        if qs is not None:
            prev_qs = qs
            tmp_lft_arm = robot.lft_arm.clone()
            tmp_lft_arm.fk(qs=qs)
            tmp_lft_arm.attach_to(base.scene)
base.run()
