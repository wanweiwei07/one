import numpy as np
from one import ovw, ouc, ossop, oum
from one.collider import cpu_simd

box1 = ossop.gen_box(half_extents=(0.03, 0.03, 0.03),
                     rgb=ouc.BasicColor.ORANGE,
                     collision_type=ouc.CollisionType.AABB,
                     is_free=True)
box1.alpha=.5
box2 = ossop.gen_box(half_extents=(0.03, 0.03, 0.03),
                     rgb=ouc.BasicColor.GREEN,
                     collision_type=ouc.CollisionType.AABB,
                     is_free=True)

box1.set_rotmat_pos(rotmat=oum.rotmat_from_axangle(
    ouc.StandardAxis.X, oum.pi / 4), pos=np.array([0.0, 0.0, 0.01]))
box2.set_rotmat_pos(rotmat=np.eye(3), pos=np.array([0.02, 0.0, 0.0]))

import time
tic=time.time()
hit_points = cpu_simd.is_sobj_collided(box1, box2)
toc = time.time()
print(f"Collision check time: {toc - tic} seconds")

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .1),
                 toggle_auto_cam_orbit=True)
box1.attach_to(base.scene)
box2.attach_to(base.scene)

if hit_points is not None:
    for hit in hit_points:
        s = ossop.gen_sphere(
            pos=hit, radius=0.002,
            rgb=ouc.BasicColor.RED, alpha=ouc.ALPHA.SOLID,
            collision_type=None, is_free=False)
        s.attach_to(base.scene)

base.run()
