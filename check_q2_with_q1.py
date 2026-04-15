import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

print("Finding q2's Reference with q1=45°")
print("="*80)

q1 = np.radians(45)

# When q1=45°, q2=0, q3=0
robot.fk(qs=[q1, 0, 0, 0, 0, 0])
tcp = robot.gl_tcp_tf[:3, 3]
rot = robot.gl_tcp_tf[:3, :3]
wrist = tcp - rot @ solver.ow_6

print(f"Wrist at (q1=45°, q2=0, q3=0): {wrist}")
print()

# Transform to J1's frame
a1 = oum.unit_vec(solver.a1, return_length=False)
R_j1 = oum.rotmat_from_axangle(a1, q1)

v_in_j1 = R_j1 @ (wrist - solver.o2)
print(f"Wrist in J1's frame: {v_in_j1}")
print()

# a2 in J1's frame
a2_in_j1 = R_j1 @ solver.a2
a2_unit = oum.unit_vec(a2_in_j1, return_length=False)

print(f"a2 in J1's frame: {a2_unit}")
print()

# Project v onto swing plane
v_proj = v_in_j1 - np.dot(v_in_j1, a2_unit) * a2_unit
print(f"v projected: {v_proj}")
print()

# Now, what is dir_zero?
z_axis = np.array([0, 0, 1], dtype=np.float32)
z_in_j1 = R_j1 @ z_axis
dir_zero = z_in_j1 - np.dot(z_in_j1, a2_unit) * a2_unit
dir_zero_normalized = oum.unit_vec(dir_zero, return_length=False)

print(f"Z in J1's frame: {z_in_j1}")
print(f"dir_zero: {dir_zero_normalized}")
print()

# The angle from dir_zero to v_proj
dir_perp = oum.unit_vec(np.cross(a2_unit, dir_zero_normalized), return_length=False)

coord_zero = np.dot(v_proj, dir_zero_normalized)
coord_perp = np.dot(v_proj, dir_perp)

phi = np.arctan2(coord_perp, coord_zero)

print(f"coord_zero: {coord_zero:.4f}, coord_perp: {coord_perp:.4f}")
print(f"phi (angle from dir_zero to wrist): {np.degrees(phi):.2f}°")
print(f"Expected: 0° (since q2=0)")
print()

# But we get phi != 0! This means dir_zero is wrong!

# What SHOULD dir_zero be?
# It should be the direction of the arm when q2=0, projected onto swing plane

# The actual arm direction at q2=0 (in J1's frame) is v_in_j1
# Normalize it
actual_dir_zero = oum.unit_vec(v_proj, return_length=False)

print(f"Actual dir_zero (from FK): {actual_dir_zero}")
print(f"Current dir_zero (from +Z projection): {dir_zero_normalized}")
print()

angle_diff = np.arccos(np.clip(np.dot(actual_dir_zero, dir_zero_normalized), -1, 1))
print(f"Difference: {np.degrees(angle_diff):.2f}°")

print("\n" + "="*80)
print("AHA! The problem is that dir_zero changes with q1!")
print("We're computing it from +Z, but the actual zero direction")
print("depends on how q1 has rotated the mechanism.")
