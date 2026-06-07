import builtins
import time
import numpy as np
from one import ovw, ouc, osso, ossop
import one.robots.end_effectors.onrobot.or_sd.or_sd as or_sd
from one.grasp.monocontact import monocontact

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
builtins.base = base
ossop.frame().attach_to(base.scene)

tool = or_sd.ORSD()
bunny = osso.SceneObject.from_file("bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny.attach_to(base.scene)

print("Computing monocontact (single-contact / suction) grasps...")
print("  Target: bunny   tcp: tip")
print("  Density: 0.015   Roll step: 90 deg   approach_bias: world +z")
tic = time.perf_counter()
grasps = monocontact(tool=tool, target_sobj=bunny, tcp='tip',
                     density=0.015, roll_step_deg=90, max_grasps=50)
toc = time.perf_counter()
planning_time = toc - tic

print(f"\nPlanning completed in {planning_time:.3f} seconds")
print(f"Found {len(grasps)} collision-free single-contact grasps")
if len(grasps) > 0:
    print(f"Planning rate: {len(grasps) / planning_time:.1f} grasps/sec")
    print(f"Score range: [{min(g[2] for g in grasps):.4f}, "
          f"{max(g[2] for g in grasps):.4f}]")
    print("\nTop 10 grasps (high score = top-facing surface, suction from above):")
    for i, (pose, _, score) in enumerate(grasps[:10]):
        pos = pose[:3, 3]
        print(f"  {i + 1:2d}. score={score:.4f}, "
              f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    tip_loc_inv = np.linalg.inv(tool.tcp('tip').loc_tf)
    s_lo = min(g[2] for g in grasps)
    s_hi = max(g[2] for g in grasps)
    s_span = (s_hi - s_lo) + 1e-9
    print(f"\nVisualizing all {len(grasps)} grasps (green = high score, "
          f"red = low score)...")
    for pose, pre_pose, score in grasps:
        t = (score - s_lo) / s_span
        # contact pose, coloured by score
        ghost = tool.clone()
        base_tf = pose @ tip_loc_inv
        ghost.set_pos_rotmat(base_tf[:3, 3], base_tf[:3, :3])
        ghost.rgb = (1.0 - t, t, 0.0)
        ghost.alpha = 0.4
        ghost.attach_to(base.scene)
        # pre-contact (retreated) pose, faint
        ghost = tool.clone()
        pre_tf = pre_pose @ tip_loc_inv
        ghost.set_pos_rotmat(pre_tf[:3, 3], pre_tf[:3, :3])
        ghost.rgb = (1.0, 1.0, 0.0)
        ghost.alpha = 0.15
        ghost.attach_to(base.scene)
else:
    print("No collision-free grasps found")

base.run()
