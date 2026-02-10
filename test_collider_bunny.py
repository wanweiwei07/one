import builtins, time
import numpy as np
from one import ovw, ouc, osso, ossop
import one.collider.gpu_simd_batch as ocgsb

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .1),
                 toggle_auto_cam_orbit=True)
builtins.base = base
bunny1 = osso.SceneObject.from_file("bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny1.alpha = .5
bunny2 = osso.SceneObject.from_file("bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny2.alpha = .3
bunny1.set_rotmat_pos(rotmat=np.eye(3), pos=np.array([0.0, 0.0, 0.0]))
bunny2.set_rotmat_pos(rotmat=np.eye(3), pos=np.array([0.03, 0.05, 0.0]))
bunny1.attach_to(base.scene)
bunny2.attach_to(base.scene)

print("Testing high-level collision detection (GPU with CPU fallback)...")
print("\nFirst run (cold start):")
gcd_detor = ocgsb.create_detector()
items = [bunny1, bunny2]
pairs = [(0,1)]
batch = ocgsb.build_batch(items, pairs)
tic = time.perf_counter()
results = gcd_detor.detect_collision_batch(batch)
toc = time.perf_counter()
print(f"  Time: {(toc - tic)*1000:.4f} ms")
if results is not None:
    print(f"Found {len(results[0])} collision points")
else:
    print("No collision detected")
if results is not None:
    hit_points, pair_ids = results
    for hit_point in hit_points:
        s = ossop.sphere(
            pos=hit_point, radius=0.002,
            rgb=ouc.BasicColor.RED, alpha=ouc.ALPHA.SOLID,
            collision_type=None, is_free=False)
        s.attach_to(base.scene)

print("\nRepeated runs (warm):")
times = []
for i in range(10):
    tic = time.perf_counter()
    results = gcd_detor.detect_collision_batch(batch)
    toc = time.perf_counter()
    elapsed = (toc - tic) * 1000
    times.append(elapsed)
    print(f"  Iteration {i+1:2d}: {elapsed:.4f} ms")

import numpy as np
print(f"\nStatistics:")
print(f"  Min:  {min(times):.4f} ms")
print(f"  Max:  {max(times):.4f} ms")
print(f"  Mean: {np.mean(times):.4f} ms")
print(f"  Std:  {np.std(times):.4f} ms")

base.run()
