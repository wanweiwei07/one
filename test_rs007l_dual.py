import numpy as np
from one import rm, wd, prims, khi_rs007l

base = wd.World(cam_pos=(1.6, .3, .7), cam_lookat_pos=(0, 0, .45),
                toggle_auto_cam_orbit=False)
# world origin
oframe = prims.gen_frame().attach_to(base.scene)
base_pos1 = np.array([0, 0.5, 0])
base_rotmat = rm.rotmat_from_euler(0, 0, -np.pi / 2)
# robot 1 (left robot)
robot1 = khi_rs007l.RS007L()
robot1.attach_to(base.scene)
robot1.set_base_rotmat_pos(rotmat=base_rotmat, pos=base_pos1)
# robot1.toggle_render_collision = True
# robot 2 (right robot)
robot2 = robot1.clone()
base_pos2 = np.array([0, -0.5, 0])
robot2.attach_to(base.scene)
robot2.set_base_rotmat_pos(rotmat=base_rotmat, pos=base_pos2)
# robot2.toggle_render_collision = True
# goal1
tgt1_rotmat = rm.rotmat_from_euler(-rm.pi / 2, 0, 0)
tgt1_pos = np.array([0.3, 0, 0.5])
g1frame = prims.gen_frame(rotmat=tgt1_rotmat, pos=tgt1_pos)
g1frame.attach_to(base.scene)
qs1, _ = robot1.ik_tcp(tgt_rotmat=tgt1_rotmat, tgt_pos=tgt1_pos)
if qs1 is not None:
    robot1.fk(qs=qs1)
# goal2
tgt2_rotmat = rm.rotmat_from_euler(rm.pi / 2, 0, 0)
tgt2_pos = np.array([0.3, 0, 0.5])
g2frame = prims.gen_frame(rotmat=tgt2_rotmat, pos=tgt2_pos)
g2frame.attach_to(base.scene)
qs2, _ = robot2.ik_tcp(tgt_rotmat=tgt2_rotmat, tgt_pos=tgt2_pos)
if qs2 is not None:
    robot2.fk(qs=qs2)
base.run()
