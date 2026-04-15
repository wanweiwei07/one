import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

print("Debugging q1=45°, q2=30°, q3=0°")
print("="*80)

q1 = np.radians(45)
q2 = np.radians(30)
q3 = np.radians(0)

# Target wrist
robot.fk(qs=[q1, q2, q3, 0, 0, 0])
tcp = robot.gl_tcp_tf[:3, 3]
rot = robot.gl_tcp_tf[:3, :3]
wrist_target = tcp - rot @ solver.ow_6

print(f"Target wrist: {wrist_target}")
print()

# Transform to J1's frame
a1 = oum.unit_vec(solver.a1, return_length=False)
R_j1 = oum.rotmat_from_axangle(a1, q1)

v_target = R_j1 @ (wrist_target - solver.o2)
print(f"v_target in J1's frame: {v_target}")
print()

# What's the wrist position at q2=0?
robot.fk(qs=[q1, 0, 0, 0, 0, 0])
tcp_q2_0 = robot.gl_tcp_tf[:3, 3]
rot_q2_0 = robot.gl_tcp_tf[:3, :3]
wrist_q2_0 = tcp_q2_0 - rot_q2_0 @ solver.ow_6

v_q2_0 = R_j1 @ (wrist_q2_0 - solver.o2)
print(f"v at q2=0 in J1's frame: {v_q2_0}")
print()

# Project both onto swing plane
a2_in_j1 = R_j1 @ solver.a2
a2_unit = oum.unit_vec(a2_in_j1, return_length=False)

v_target_proj = v_target - np.dot(v_target, a2_unit) * a2_unit
v_q2_0_proj = v_q2_0 - np.dot(v_q2_0, a2_unit) * a2_unit

print(f"v_target projected: {v_target_proj}")
print(f"v_q2_0 projected: {v_q2_0_proj}")
print()

# Normalize to get directions
dir_target = oum.unit_vec(v_target_proj, return_length=False)
dir_q2_0 = oum.unit_vec(v_q2_0_proj, return_length=False)

print(f"Direction to target: {dir_target}")
print(f"Direction at q2=0: {dir_q2_0}")
print()

# Angle between them (this should be q2!)
# But we need to define the sign convention

# Using the perpendicular direction
dir_perp = oum.unit_vec(np.cross(a2_unit, dir_q2_0), return_length=False)

coord_zero = np.dot(dir_target, dir_q2_0)
coord_perp = np.dot(dir_target, dir_perp)

angle = np.arctan2(coord_perp, coord_zero)

print(f"Angle from q2=0 to target: {np.degrees(angle):.2f}°")
print(f"Expected: 30.00°")
print()

# So if we use the actual FK q2=0 direction, we should get the right answer!
# But in the IK code, we're using projected +Z as dir_zero

# Let's see what projected +Z gives us
z_axis = np.array([0, 0, 1], dtype=np.float32)
z_in_j1 = R_j1 @ z_axis
dir_z_proj = z_in_j1 - np.dot(z_in_j1, a2_unit) * a2_unit
dir_z_proj_norm = oum.unit_vec(dir_z_proj, return_length=False)

print(f"Projected +Z: {dir_z_proj_norm}")
print(f"Actual q2=0 direction: {dir_q2_0}")
print()

offset = np.arccos(np.clip(np.dot(dir_z_proj_norm, dir_q2_0), -1, 1))
print(f"Offset between them: {np.degrees(offset):.2f}°")

# Using projected +Z as reference:
dir_perp_z = oum.unit_vec(np.cross(a2_unit, dir_z_proj_norm), return_length=False)
coord_z = np.dot(dir_target, dir_z_proj_norm)
coord_perp_z = np.dot(dir_target, dir_perp_z)
angle_z = np.arctan2(coord_perp_z, coord_z)

print(f"Angle using projected +Z: {np.degrees(angle_z):.2f}°")
print()

print("="*80)
print("CONCLUSION:")
print(f"The offset is {np.degrees(offset):.2f}° and angle difference is {np.degrees(angle - angle_z):.2f}°")
