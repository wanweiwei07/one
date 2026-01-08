import numpy as np
import one.physics.mj_env as mj
from one import oum, ovw, ossop, ouc, osso, khi_rs007l

base = ovw.World(cam_pos=(3.5, 1, 3.5),
                 cam_lookat_pos=(0, 0, .5))

robot = khi_rs007l.RS007L()
robot.attach_to(base.scene)
robot.toggle_render_collision = True
robot.fk(qs=[0, 0, -np.pi / 4, 0, 0, 0])

for i in range(1,15):
    tmp_robot = robot.clone()
    tmp_robot.base_pos=np.array([0,0,i*1.5])
    tmp_robot.attach_to(base.scene)

plane_bottom = ossop.gen_plane()
plane_bottom.toggle_render_collision = True
plane_bottom.attach_to(base.scene)

mjenv = mj.MJEnv(scene=base.scene)
# mjenv.sync_mechstates_to_mujoco()
mjenv.save("scene.xml")
base.schedule_interval(mjenv.step)
base.run()