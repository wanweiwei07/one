import numpy as np
import one.collider.mj_collider as ocm
from one import oum, ovw, ossop, khi_rs007l, or_2fg7

# Simple test to verify robot-ground collision detection
base = ovw.World(cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, .75),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

robot = khi_rs007l.RS007L(pos=(.5, 0, 0.01))
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.engage(gripper)

# create ground plane at z=0
ground = ossop.plane(pos=(0, 0, 0))
ground.attach_to(base.scene)

# setup mj collider
mjc = ocm.MJCollider()
mjc.append(robot)
mjc.append(ground)
mjc.actors = [robot]
mjc.compile(margin=0.0)

# Test 1: Initial pose (should be collision-free with slight z offset)
qs_safe = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
print(f"Test 1 - Safe pose [0,0,0,0,0,0]: collided = {mjc.is_collided(qs_safe)}")

# Test 2: Joint angles that cause collision with ground
qs_collision = np.array([0, np.pi/3, 0, 0, 0, 0], dtype=np.float32)
print(f"Test 2 - Collision pose [0,π/3,0,0,0,0]: collided = {mjc.is_collided(qs_collision)}")

# Test 3: Lower robot base to z=0 (should definitely collide)
robot.pos = (.5, 0, 0)
qs_low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
print(f"Test 3 - Robot at z=0: collided = {mjc.is_collided(qs_low)}")

print("\nIf Test 3 shows 'collided = True', collision detection is working correctly!")

base.run()
