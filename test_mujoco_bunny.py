import cProfile
import builtins
import numpy as np
import one.physics.mj_env as opme
from one import ovw, ossop, ouc, osso

oframe = ossop.gen_frame()
bunny = osso.SceneObject.from_file(
    "bunny.stl", collision_type=ouc.CollisionType.MESH,
    is_free=True)
bunny.rgb = ouc.ExtendedColor.PINK
base = ovw.World(cam_pos=(.75, 1.5, 1.5),
                 toggle_auto_cam_orbit=False)
builtins.base = base
oframe.attach_to(base.scene)
for i in np.linspace(.5, 10.5, 50):
    tmp_bunny = bunny.clone()
    tmp_bunny.pos = (0, 0, i)
    tmp_bunny.attach_to(base.scene)
plane_ground = ossop.gen_plane()
plane_ground.attach_to(base.scene)
wall_left = ossop.gen_box(name="wall", pos=(.0, .35, .2),
                          half_extents=(.355, .005, .2),
                          collision_type=ouc.CollisionType.AABB,
                          alpha=ouc.ALPHA.TRANSPARENT, is_free=False)
wall_left.attach_to(base.scene)
wall_right = ossop.gen_box(name="wall", pos=(.0, -.35, .2),
                           half_extents=(.355, .005, .2),
                           collision_type=ouc.CollisionType.AABB,
                           alpha=ouc.ALPHA.TRANSPARENT, is_free=False)
wall_right.attach_to(base.scene)
wall_front = ossop.gen_box(name="wall", pos=(.35, 0, .2),
                           half_extents=(.005, .355, .2),
                           collision_type=ouc.CollisionType.AABB,
                           alpha=ouc.ALPHA.TRANSPARENT, is_free=False)
wall_front.attach_to(base.scene)
wall_back = ossop.gen_box(name="wall", pos=(-.35, 0, .2),
                          half_extents=(.005, .355, .2),
                          collision_type=ouc.CollisionType.AABB,
                          alpha=ouc.ALPHA.TRANSPARENT, is_free=False)
wall_back.attach_to(base.scene)
mjenv = opme.MJEnv(scene=base.scene)
base.schedule_interval(mjenv.step)
base.run()
