import builtins
import time
import numpy as np
from one import ovw, ouc, osso, ossop, or_2fg7
from one.grasp.antipodal import antipodal

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
builtins.base = base
ossop.frame().attach_to(base.scene)
gripper = or_2fg7.OR2FG7()
bunny = osso.SceneObject.from_file("bunny.stl", collision_type=ouc.CollisionType.MESH)
bunny.attach_to(base.scene)
# bunny.toggle_render_collision = True

print("Computing antipodal grasps...")
print("  Target: bunny")
print("  Density: 0.02")
print("  Normal tolerance: 20 deg")
print("  Roll step: 30 deg")
tic = time.perf_counter()
grasps = antipodal(gripper=gripper, target_sobj=bunny,
                   density=0.01, normal_tol_deg=20, roll_step_deg=30,
                   max_grasps=50)
toc = time.perf_counter()
planning_time = toc - tic

print(f"\nPlanning completed in {planning_time:.3f} seconds")
print(f"Found {len(grasps)} collision-free grasps")
if len(grasps) > 0:
    print(f"Planning rate: {len(grasps) / planning_time:.1f} grasps/sec")
    print(f"Score range: [{min(g[2] for g in grasps):.4f}, {max(g[2] for g in grasps):.4f}]")
    print("\nTop 10 grasps:")
    for i, (pose, _, jaw_width, score) in enumerate(grasps[:10]):
        pos = pose[:3, 3]
        print(f"  {i + 1:2d}. score={score:.4f}, jaw={jaw_width:.4f}, "
              f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"\nVisualizing all {len(grasps)} grasps...")
    for i, (pose, pre_pose, jaw_width, score) in enumerate(grasps):
        # pose
        ghost = gripper.clone()
        ghost.grip_at(pose[:3, 3], pose[:3, :3], jaw_width)
        ghost.rgb = (0.0, 1.0, 0.0)
        ghost.alpha = 0.3
        ghost.attach_to(base.scene)
        # pre_pose
        ghost = gripper.clone()
        ghost.grip_at(pre_pose[:3, 3], pre_pose[:3, :3], jaw_width)
        ghost.rgb = (1.0, 1.0, 0.0)
        ghost.alpha = 0.5
        ghost.attach_to(base.scene)
    print("  Green = high score, Red = low score")
else:
    print("No collision-free grasps found")

base.run()
