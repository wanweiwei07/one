import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

print("Finding q2's Zero Reference Direction")
print("="*80)

# When q1=0, q2=0, q3=0, where is the wrist?
robot.fk(qs=[0, 0, 0, 0, 0, 0])
tcp_zero = robot.gl_tcp_tf[:3, 3]
rot_zero = robot.gl_tcp_tf[:3, :3]
wrist_zero = tcp_zero - rot_zero @ solver.ow_6

print(f"Wrist at zero config: {wrist_zero}")
print(f"o2 (J2 origin): {solver.o2}")
print(f"Wrist - o2: {wrist_zero - solver.o2}")
print()

# This is the direction the arm points when q2=0
arm_dir_zero = wrist_zero - solver.o2
arm_dir_zero_normalized = oum.unit_vec(arm_dir_zero, return_length=False)

print(f"Arm direction at q2=0 (world frame): {arm_dir_zero_normalized}")
print()

# Now rotate q2 by 30°
robot.fk(qs=[0, np.radians(30), 0, 0, 0, 0])
tcp_q2_30 = robot.gl_tcp_tf[:3, 3]
rot_q2_30 = robot.gl_tcp_tf[:3, :3]
wrist_q2_30 = tcp_q2_30 - rot_q2_30 @ solver.ow_6

arm_dir_q2_30 = wrist_q2_30 - solver.o2
arm_dir_q2_30_normalized = oum.unit_vec(arm_dir_q2_30, return_length=False)

print(f"Wrist at q2=30°: {wrist_q2_30}")
print(f"Arm direction at q2=30°: {arm_dir_q2_30_normalized}")
print()

# What's the angle between them?
angle = np.arccos(np.clip(np.dot(arm_dir_zero_normalized, arm_dir_q2_30_normalized), -1, 1))
print(f"Angle between q2=0 and q2=30°: {np.degrees(angle):.2f}° (should be 30°)")
print()

# Now let's check: in J1's frame (with q1=0), what is the arm direction?
# Since q1=0, J1's frame = world frame

# The zero direction should be...
a2 = solver.a2
a2_unit = oum.unit_vec(a2, return_length=False)

print(f"a2 (J2 axis): {a2_unit}")
print()

# The arm swings in a plane perpendicular to a2
# Project arm_dir_zero onto that plane
arm_proj_zero = arm_dir_zero - np.dot(arm_dir_zero, a2_unit) * a2_unit
arm_proj_q2_30 = arm_dir_q2_30 - np.dot(arm_dir_q2_30, a2_unit) * a2_unit

print(f"Arm projection (q2=0): {arm_proj_zero}")
print(f"Arm projection (q2=30°): {arm_proj_q2_30}")
print()

# In the swing plane, what coordinate system should we use?
# Current code uses: dir_zero = project(+Z onto swing plane)

z_axis = np.array([0, 0, 1], dtype=np.float32)
z_proj = z_axis - np.dot(z_axis, a2_unit) * a2_unit
z_proj_normalized = oum.unit_vec(z_proj, return_length=False)

print(f"Projected +Z onto swing plane: {z_proj_normalized}")
print()

# But the actual zero direction is arm_proj_zero!
actual_zero_dir = oum.unit_vec(arm_proj_zero, return_length=False)
print(f"Actual zero direction: {actual_zero_dir}")
print()

# What's the angle between them?
angle_offset = np.arccos(np.clip(np.dot(z_proj_normalized, actual_zero_dir), -1, 1))
print(f"Offset angle: {np.degrees(angle_offset):.2f}°")
print()

# So the problem is: when we compute phi_wrist using projected +Z as reference,
# we need to add this offset!

print("="*80)
print("SOLUTION: Instead of using projected +Z, use actual zero direction from FK")
