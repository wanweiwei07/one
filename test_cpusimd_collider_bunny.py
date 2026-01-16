import builtins
import numpy as np
from one import ovw, ouc, osso, ossop
from one.collider import cpu_simd

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .1),
                 toggle_auto_cam_orbit=True)
builtins.base = base
bunny1 = osso.SceneObject.from_file(
    "bunny.stl",
    collision_type=ouc.CollisionType.MESH)
bunny1.alpha = .5
bunny2 = osso.SceneObject.from_file(
    "bunny.stl",
    collision_type=ouc.CollisionType.MESH)
bunny2.alpha = .3
bunny1.set_rotmat_pos(rotmat=np.eye(3), pos=np.array([0.0, 0.0, 0.0]))
bunny2.set_rotmat_pos(rotmat=np.eye(3), pos=np.array([0.02, 0.0, 0.0]))
hit_points = cpu_simd.is_sobj_collided(bunny1, bunny2)
bunny1.attach_to(base.scene)
bunny2.attach_to(base.scene)
if hit_points is not None:
    for hit_point in hit_points:
        s = ossop.gen_sphere(
            pos=hit_point, radius=0.002,
            rgb=ouc.BasicColor.RED, alpha=ouc.ALPHA.SOLID,
            collision_type=None, is_free=False)
        s.attach_to(base.scene)
base.run()
