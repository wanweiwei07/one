import numpy as np
from one import oum, ovw, ossop, khi_rs007l

base = ovw.World(cam_pos=(1.6, .3, .7), cam_lookat_pos=(0, 0, .45),
                toggle_auto_cam_orbit=False)
# world origin
oframe = ossop.frame().attach_to(base.scene)
base_pos1 = np.array([0, 0.5, 0])
base_rotmat = oum.rotmat_from_euler(0, 0, -np.pi / 2)
# robot 1 (left robot)
robot1 = khi_rs007l.RS007L()
robot1.attach_to(base.scene)
robot1.set_rotmat_pos(rotmat=base_rotmat, pos=base_pos1)
robot1.toggle_render_collision = True
# robot 2 (right robot)
robot2 = robot1.clone()
base_pos2 = np.array([0, -0.5, 0])
robot2.attach_to(base.scene)
robot2.set_rotmat_pos(rotmat=base_rotmat, pos=base_pos2)
robot2.toggle_render_collision = False
# goal1
tgt1_rotmat = oum.rotmat_from_euler(-oum.pi / 2, 0, 0)
tgt1_pos = np.array([0.3, 0, 0.5])
g1frame = ossop.frame(rotmat=tgt1_rotmat, pos=tgt1_pos)
g1frame.attach_to(base.scene)
qs1_list = robot1.ik_tcp(tgt_rotmat=tgt1_rotmat, tgt_pos=tgt1_pos)
for qs in qs1_list:
    tmp_robot = robot1.clone()
    tmp_robot.fk(qs=qs)
    tmp_robot.attach_to(base.scene)
# goal2
tgt2_rotmat = oum.rotmat_from_euler(oum.pi / 2, 0, 0)
tgt2_pos = np.array([0.3, 0, 0.5])
g2frame = ossop.frame(rotmat=tgt2_rotmat, pos=tgt2_pos)
g2frame.attach_to(base.scene)
qs2_list = robot2.ik_tcp(tgt_rotmat=tgt2_rotmat, tgt_pos=tgt2_pos)
for qs in qs2_list:
    tmp_robot = robot2.clone()
    tmp_robot.fk(qs=qs)
    tmp_robot.attach_to(base.scene)
base.run()
