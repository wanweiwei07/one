"""
Debug script to trace through _solve_first3 step by step.
"""
import numpy as np
import one.utils.math as oum
from one.robots.manipulators.kawasaki.rs007l import rs007l

# Test case: q1=45°, q2=30°, q3=0°
robot = rs007l.RS007L()
qs_input = np.array([np.pi/4, np.pi/6, 0, 0, 0, 0], dtype=np.float32)

robot.fk(qs_input)
target_tcp = robot.gl_tcp_tf[:3, 3].copy()
target_rotmat = robot.gl_tcp_tf[:3, :3].copy()
wrist_pos = robot.gl_lnk_tfarr[4][:3, 3].copy()

print("Input configuration: q1=45°, q2=30°, q3=0°")
print(f"Target TCP: {target_tcp}")
print(f"Wrist position: {wrist_pos}")

solver = robot.get_solver(robot._chain)
pw = target_tcp - target_rotmat @ solver.ow_6

print(f"\nComputed pw (wrist center): {pw}")
print(f"Match with actual wrist: {np.allclose(pw, wrist_pos)}")

# Now manually trace through _solve_first3
print("\n" + "=" * 80)
print("Tracing _solve_first3")
print("=" * 80)

a1 = oum.unit_vec(solver.a1, return_length=False)
v12 = solver.o2 - solver.o1
d12 = np.linalg.norm(v12)

print(f"\na1 (J1 axis): {a1}")
print(f"o1 (J1 origin): {solver.o1}")
print(f"o2 (J2 origin): {solver.o2}")
print(f"d12 = ||o2 - o1||: {d12}")

v_shoulder_to_wrist = pw - solver.o1
print(f"\nv_shoulder_to_wrist: {v_shoulder_to_wrist}")

# Spherical joint case
v_projected = v_shoulder_to_wrist - np.dot(v_shoulder_to_wrist, a1) * a1
print(f"v_projected (onto plane ⊥ a1): {v_projected}")
print(f"||v_projected||: {np.linalg.norm(v_projected)}")

# Calculate q1
if abs(a1[2]) > 0.9:
    x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
else:
    x_ref = np.cross(a1, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    x_ref = oum.unit_vec(x_ref, return_length=False)

y_ref = np.cross(a1, x_ref)
px = np.dot(v_projected, x_ref)
py = np.dot(v_projected, y_ref)

print(f"\nx_ref: {x_ref}")
print(f"y_ref: {y_ref}")
print(f"px: {px}")
print(f"py: {py}")

azimuth = np.arctan2(py, px)
q1 = float(azimuth + np.pi / 2)

print(f"\nazimuth: {np.degrees(azimuth):.2f}°")
print(f"q1 = azimuth + 90°: {np.degrees(q1):.2f}°")
print(f"Expected q1: {np.degrees(qs_input[0]):.2f}°")

# Transform to J1's rotated frame
rotmat1 = oum.rotmat_from_axangle(a1, -q1)
v_in_j1_frame = rotmat1 @ v_shoulder_to_wrist
a2_in_j1_frame = rotmat1 @ solver.a2

print(f"\nAfter rotating by -q1 around a1:")
print(f"v_in_j1_frame: {v_in_j1_frame}")
print(f"a2_in_j1_frame: {a2_in_j1_frame}")

# Project to swing plane
a2_unit = oum.unit_vec(a2_in_j1_frame, return_length=False)
v_in_swing_plane = v_in_j1_frame - np.dot(v_in_j1_frame, a2_unit) * a2_unit
d_swing = np.linalg.norm(v_in_swing_plane)

print(f"\na2_unit: {a2_unit}")
print(f"v_in_swing_plane: {v_in_swing_plane}")
print(f"d_swing: {d_swing}")

# Solve q3
l2, l3 = solver.l2, solver.l3
print(f"\nl2: {l2}, l3: {l3}")

c3 = oum.clamp((d_swing * d_swing - l2 * l2 - l3 * l3) / (2 * l2 * l3), lo=-1.0, hi=1.0)
s3_abs = np.sqrt(max(0.0, 1.0 - c3 * c3))

print(f"\nc3 (cosine of q3): {c3}")
print(f"s3_abs: {s3_abs}")

q3 = float(np.arctan2(s3_abs, c3))
print(f"q3: {np.degrees(q3):.2f}°")
print(f"Expected q3: {np.degrees(qs_input[2]):.2f}°")

# Build 2D coordinate system in swing plane
dir1 = np.cross(a2_unit, a1)
if np.linalg.norm(dir1) < 1e-9:
    dir1 = np.array([0, 1, 0], dtype=np.float32)
dir1 = oum.unit_vec(dir1, return_length=False)

dir2 = np.cross(a2_unit, dir1)
dir2 = oum.unit_vec(dir2, return_length=False)

print(f"\nSwing plane basis:")
print(f"dir1: {dir1}")
print(f"dir2: {dir2}")

# Wrist coordinates in swing plane 2D system
wrist_coord1 = np.dot(v_in_swing_plane, dir1)
wrist_coord2 = np.dot(v_in_swing_plane, dir2)

print(f"\nWrist in swing plane 2D:")
print(f"wrist_coord1: {wrist_coord1}")
print(f"wrist_coord2: {wrist_coord2}")

# Solve q2
phi_wrist = np.arctan2(wrist_coord2, wrist_coord1)
psi = np.arctan2(l3 * s3_abs, l2 + l3 * c3)
q2 = float(phi_wrist - psi)

print(f"\nphi_wrist (angle to wrist): {np.degrees(phi_wrist):.2f}°")
print(f"psi (elbow contribution): {np.degrees(psi):.2f}°")
print(f"q2 = phi_wrist - psi: {np.degrees(q2):.2f}°")
print(f"Expected q2: {np.degrees(qs_input[1]):.2f}°")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print(f"Computed: q1={np.degrees(q1):.2f}°, q2={np.degrees(q2):.2f}°, q3={np.degrees(q3):.2f}°")
print(f"Expected: q1={np.degrees(qs_input[0]):.2f}°, q2={np.degrees(qs_input[1]):.2f}°, q3={np.degrees(qs_input[2]):.2f}°")

# Verify with FK
qs_computed = np.array([q1, q2, q3, 0, 0, 0], dtype=np.float32)
robot.fk(qs_computed)
computed_wrist = robot.gl_lnk_tfarr[4][:3, 3]
error = np.linalg.norm(computed_wrist - wrist_pos)
print(f"\nFK verification:")
print(f"Target wrist: {wrist_pos}")
print(f"Computed wrist: {computed_wrist}")
print(f"Error: {error:.6f}m")
