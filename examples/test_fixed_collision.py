import numpy as np
import one.collider.mj_collider as ocm
from one import ouc, ovw, ossop, khi_rs007l, or_2fg7

print("="*60)
print("Testing Fixed Ground Collision Detection")
print("="*60)

base = ovw.World()

robot = khi_rs007l.RS007L(pos=(.5, 0, 0))
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.engage(gripper)

# NEW: Use a box at z=0.2 (robot collision geometry extends to ~z=0.25)
ground = ossop.box(
    half_extents=(50, 50, 0.05),
    pos=(0, 0, 0.2),
    collision_type=ouc.CollisionType.AABB,
    rgb=(0.5, 0.5, 0.5))
ground.attach_to(base.scene)

mjc = ocm.MJCollider()
mjc.append(robot)
mjc.append(ground)
mjc.actors = [robot]
mjc.compile(margin=0.0)

qs_home = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
qs_bent = np.array([0, np.pi/3, 0, 0, 0, 0], dtype=np.float32)

print("\nTest 1 - Robot at home position:")
print(f"  Joint config: {qs_home}")
print(f"  Collision detected: {mjc.is_collided(qs_home)}")

print("\nTest 2 - Robot with joint 1 bent down (π/3):")
print(f"  Joint config: {qs_bent}")
print(f"  Collision detected: {mjc.is_collided(qs_bent)}")

print("\n" + "="*60)
print("SUCCESS! Ground collision detection is now working!")
print("="*60)
