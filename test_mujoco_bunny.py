import cProfile
import numpy as np
import one.physics.mujoco_env as mj
from one import wd, prims, const, sob

oframe = prims.gen_frame()
bunny = sob.SceneObject.from_file("bunny.stl", collision_type=const.CollisionType.MESH)
# bunny.toggle_render_collision = True
base = wd.World(cam_pos=(1, 2, 2), toggle_auto_cam_orbit=False)
oframe.attach_to(base.scene)
# bunny.attach_to(base.scene)
# bunny.pos = (0, 0, 1)
for i in np.linspace(.5, 10.5, 50):
    tmp_bunny = bunny.clone()
    tmp_bunny.pos = (0, 0, i)
    tmp_bunny.attach_to(base.scene)
plane_bottom = prims.gen_plane()
plane_bottom.toggle_render_collision = True
plane_bottom.attach_to(base.scene)

plane_left = prims.gen_plane(pos=(0, .5, 0), size=(.5, .5),
                             normal=-const.StandardAxis.Y,
                             alpha=const.ALPHA.TRANSPARENT)
plane_left.attach_to(base.scene)
plane_right = prims.gen_plane(pos=(0, -.5, 0), size=(.5, .5),
                              normal=const.StandardAxis.Y,
                              alpha=const.ALPHA.TRANSPARENT)
plane_right.attach_to(base.scene)
plane_front = prims.gen_plane(pos=(.5, 0, 0), size=(.5, .5),
                              normal=-const.StandardAxis.X,
                              alpha=const.ALPHA.TRANSPARENT)
plane_front.attach_to(base.scene)
plane_back = prims.gen_plane(pos=(-.5, 0, 0), size=(.5, .5),
                             normal=const.StandardAxis.X,
                             alpha=const.ALPHA.TRANSPARENT)
plane_back.attach_to(base.scene)

# base.run()
mjenv = mj.MuJoCoEnv(scene=base.scene)
base.schedule_interval(mjenv.step)
base.run()
