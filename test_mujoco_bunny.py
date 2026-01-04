import cProfile
import builtins
import numpy as np
import one.physics.mj_env as mj
from one import ovw, ossop, ouc, osso

oframe = ossop.gen_frame()
bunny = osso.SceneObject.from_file("bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny.rgb = ouc.ExtendedColor.PINK
# bunny.toggle_render_collision = True
base = ovw.World(cam_pos=(.75, 1.5, 1.5), toggle_auto_cam_orbit=False)
builtins.base = base
oframe.attach_to(base.scene)
# bunny.attach_to(base.scene)
# bunny.pos = (0, 0, 1)
for i in np.linspace(.5, 10.5, 50):
    tmp_bunny = bunny.clone()
    tmp_bunny.pos = (0, 0, i)
    tmp_bunny.attach_to(base.scene)
plane_bottom = ossop.gen_plane()
plane_bottom.toggle_render_collision = True
plane_bottom.attach_to(base.scene)

plane_left = ossop.gen_plane(pos=(0, .35, 0), size=(.7, .7),
                             normal=-ouc.StandardAxis.Y,
                             alpha=ouc.ALPHA.TRANSPARENT)
plane_left.attach_to(base.scene)
plane_right = ossop.gen_plane(pos=(0, -.35, 0), size=(.7, .7),
                              normal=ouc.StandardAxis.Y,
                              alpha=ouc.ALPHA.TRANSPARENT)
plane_right.attach_to(base.scene)
plane_front = ossop.gen_plane(pos=(.35, 0, 0), size=(.7, .7),
                              normal=-ouc.StandardAxis.X,
                              alpha=ouc.ALPHA.TRANSPARENT)
plane_front.attach_to(base.scene)
plane_back = ossop.gen_plane(pos=(-.35, 0, 0), size=(.7, .7),
                             normal=ouc.StandardAxis.X,
                             alpha=ouc.ALPHA.TRANSPARENT)
plane_back.attach_to(base.scene)

mjenv = mj.MjEnv(scene=base.scene)
base.schedule_interval(mjenv.step)
base.run()
