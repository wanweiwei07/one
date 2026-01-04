import builtins
import numpy as np
import one.physics.mj_env as mj
from one import rm, wd, prims, const, sob, khi_rs007l

oframe = prims.gen_frame()
bunny = sob.SceneObject.from_file("bunny.stl", collision_type=const.CollisionType.MESH)
bunny.rgb = const.ExtendedColor.PINK
bunny.alpha=0.4
# bunny.toggle_render_collision = True
base = wd.World(cam_pos=(3.5, 1, 3.5), cam_lookat_pos=(0, 0, .5), toggle_auto_cam_orbit=False)
builtins.base = base
oframe.attach_to(base.scene)
bunny.attach_to(base.scene)
for i in np.linspace(1.5, 15.5, 50):
    tmp_bunny = bunny.clone()
    tmp_bunny.pos = (.5, 0, i)
    tmp_bunny.attach_to(base.scene)
# container
plane_bottom = prims.gen_plane()
plane_bottom.toggle_render_collision = True
plane_bottom.attach_to(base.scene)
wall_left = prims.gen_box(pos=(.5, .35, .2),
                          half_extents=(.355, .005, .2),
                          collision_type=const.CollisionType.AABB,
                          alpha=const.ALPHA.TRANSPARENT, is_fixed=True)
wall_left.attach_to(base.scene)
wall_right = prims.gen_box(pos=(.5, -.35, .2),
                           half_extents=(.355, .005, .2),
                           collision_type=const.CollisionType.AABB,
                           alpha=const.ALPHA.TRANSPARENT, is_fixed=True)
wall_right.attach_to(base.scene)
wall_front = prims.gen_box(pos=(.85, 0, .2),
                           half_extents=(.005, .355, .2),
                           collision_type=const.CollisionType.AABB,
                           alpha=const.ALPHA.TRANSPARENT, is_fixed=True)
wall_front.attach_to(base.scene)
wall_back = prims.gen_box(pos=(.15, 0, .2),
                          half_extents=(.005, .355, .2),
                          collision_type=const.CollisionType.AABB,
                          alpha=const.ALPHA.TRANSPARENT, is_fixed=True)
wall_back.attach_to(base.scene)

# robot
base_rotmat = rm.rotmat_from_euler(0, 0, -np.pi / 2)
base_pos = np.array([0, 0.7, 0])
robot1 = khi_rs007l.RS007L()
robot1.attach_to(base.scene)
robot1.set_base_rotmat_pos(rotmat=base_rotmat, pos=base_pos)
robot1.toggle_render_collision = True
robot1.fk(qs=[0, 0, -np.pi / 4, 0, 0, 0])
robot1.alpha=0.1

mjenv = mj.MjEnv(scene=base.scene)
mjenv.sync_mechstates_to_mujoco_dynamics()
mjenv.save_xml("scene.xml")
base.schedule_interval(mjenv.step)
base.run()