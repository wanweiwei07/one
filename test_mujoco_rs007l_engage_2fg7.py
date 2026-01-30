import numpy as np
import one.physics.mj_env as opme
from one import oum, ouc, ovw, ossop, khi_rs007l, or_2fg7

base = ovw.World(cam_pos=(2, 1, 1.5), cam_lookat_pos=(0, 0, .75),
                 toggle_auto_cam_orbit=False)
ossop.gen_frame().attach_to(base.scene)
robot = khi_rs007l.RS007L(pos=(.5, 0, 0))
robot.attach_to(base.scene)

tgt_pos = np.array([0, .5, .3])
tgt_rotmat = oum.rotmat_from_euler(oum.pi, 0, 0)
ossop.gen_frame(rotmat=tgt_rotmat, pos=tgt_pos).attach_to(base.scene)
qs_list = robot.ik_tcp(tgt_rotmat=tgt_rotmat, tgt_pos=tgt_pos)
robot_ik = robot.clone()
robot_ik.rgb = ouc.BasicColor.LIME
robot_ik.fk(qs=qs_list[0])
robot_ik.attach_to(base.scene)
wd_tcp_rotmat = robot_ik.wd_tcp_tf[:3, :3]
wd_tcp_pos = robot_ik.wd_tcp_tf[:3, 3]
ossop.gen_frame(rotmat=wd_tcp_rotmat, pos=wd_tcp_pos,
                color_mat=ouc.CoordColor.MYC).attach_to(base.scene)

robot2 = robot.clone()
robot2.set_rotmat_pos(pos=(-.5, 0, 0))
gripper = or_2fg7.OR2FG7()
robot2.engage(gripper)
robot2.attach_to(base.scene)
qs2_list = robot2.ik_tcp(
    tgt_rotmat=tgt_rotmat, tgt_pos=tgt_pos)

box = ossop.gen_cylinder(
    spos=(-.3, 0, .3), epos=(.3, 0, .1),
    radius=.03, is_free=True)
box.attach_to(base.scene)
gripper.grasp(box)
# gripper.release(box)
# base.run()

robot2_ik = robot2.clone()
robot2_ik.rgb = ouc.BasicColor.YELLOW
robot2_ik.fk(qs=qs2_list[0])
robot2_ik.attach_to(base.scene)

mjenv = opme.MJEnv(scene=base.scene)
base.schedule_interval(mjenv.step)
base.run()

# # engage later
# robot.engage(gripper)
# robot.fk(qs=[0, -oum.pi / 4, 0, -oum.pi / 2, 0, oum.pi / 3])
# base.run()
