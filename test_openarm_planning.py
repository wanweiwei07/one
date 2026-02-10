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

lft_gripper = oreeogog.OAGripper()
lft_gripper.attach_to(base.scene)

rgt_gripper = oreeogog.OAGripper()
rgt_gripper.attach_to(base.scene)

robot.rgt_arm.engage(rgt_gripper)
robot.lft_arm.engage(lft_gripper)
robot.body.alpha = 0.3

tgt_pos = (0.4, 0.2, 0.4)
tgt_rotmat = (oum.rotmat_from_axangle(ouc.StandardAxis.X, oum.pi / 2) @
              oum.rotmat_from_axangle(ouc.StandardAxis.Y, oum.pi / 2))
# a circle from around tgt_pos and in the plane dtermined by ouc.StandardAxis.X
tgt_pos_list = []
radius = 0.1
num_points = 8
for i in range(num_points):
    angle = (2 * oum.pi / num_points) * i
    x = tgt_pos[0]
    y = tgt_pos[1] + radius * oum.cos(angle)
    z = tgt_pos[2] + radius * oum.sin(angle)
    tgt_pos_list.append((x, y, z))
    ossop.frame(pos=(x, y, z), rotmat=tgt_rotmat).attach_to(base.scene)

qs_list = robot.lft_arm.ik_tcp(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
tcp_tf = robot.lft_arm.gl_tcp_tf
tcp_frame = ossop.frame(rotmat=tcp_tf[:3, :3], pos=tcp_tf[:3, 3],
                        color_mat=ouc.CoordColor.MYC)
tcp_frame.attach_to(base.scene)
for qs in qs_list:
    tmp_lft_arm = robot.lft_arm.clone()
    tmp_lft_arm.fk(qs=qs)
    tmp_lft_arm.attach_to(base.scene)
base.run()
