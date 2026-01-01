import builtins
import numpy as np
import one.physics.mujoco_env as mj
from one import rm, wd, prims, const, sob, khi_rs007l

oframe = prims.gen_frame()
bunny = sob.SceneObject.from_file("bunny.stl", collision_type=const.CollisionType.MESH)
bunny.rgb = const.ExtendedColor.PINK
# bunny.toggle_render_collision = True
base = wd.World(cam_pos=(3.5, 1, 3.5), cam_lookat_pos=(0, 0, .5), toggle_auto_cam_orbit=False)
builtins.base = base
oframe.attach_to(base.scene)
# bunny.attach_to(base.scene)
for i in np.linspace(.5, 10.5, 50):
    tmp_bunny = bunny.clone()
    tmp_bunny.pos = (.5, 0, i)
    tmp_bunny.attach_to(base.scene)
# container
plane_bottom = prims.gen_plane()
plane_bottom.toggle_render_collision = True
plane_bottom.attach_to(base.scene)
plane_left = prims.gen_plane(pos=(.5, .35, 0), size=(.7, .7),
                             normal=-const.StandardAxis.Y,
                             alpha=const.ALPHA.TRANSPARENT)
plane_left.attach_to(base.scene)
plane_right = prims.gen_plane(pos=(.5, -.35, 0), size=(.7, .7),
                              normal=const.StandardAxis.Y,
                              alpha=const.ALPHA.TRANSPARENT)
plane_right.attach_to(base.scene)
plane_front = prims.gen_plane(pos=(.85, 0, 0), size=(.7, .7),
                              normal=-const.StandardAxis.X,
                              alpha=const.ALPHA.TRANSPARENT)
plane_front.attach_to(base.scene)
plane_back = prims.gen_plane(pos=(.15, 0, 0), size=(.7, .7),
                             normal=const.StandardAxis.X,
                             alpha=const.ALPHA.TRANSPARENT)
plane_back.attach_to(base.scene)

# robot
base_rotmat = rm.rotmat_from_euler(0, 0, -np.pi / 2)
base_pos = np.array([0, 0.7, 0])
robot1 = khi_rs007l.RS007L()
robot1.attach_to(base.scene)
robot1.set_base_rotmat_pos(rotmat=base_rotmat, pos=base_pos)

mjenv = mj.MuJoCoEnv(scene=base.scene)
base.schedule_interval(mjenv.step)
base.run()
