import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

# Known good configuration
q1_target = np.radians(45)
q2_target = np.radians(30)
q3_target = np.radians(0)

robot.fk(qs=[q1_target, q2_target, q3_target, 0, 0, 0])
tcp_pos = robot.gl_tcp_tf[:3, 3]
tcp_rot = robot.gl_tcp_tf[:3, :3]

pw_target = tcp_pos - tcp_rot @ solver.ow_6

print(f"Target: q1={np.degrees(q1_target):.1f}°, q2={np.degrees(q2_target):.1f}°, q3={np.degrees(q3_target):.1f}°")
print(f"Wrist center: {pw_target}")
print()

# Manually run through the _solve_first3 logic
a1 = oum.unit_vec(solver.a1, return_length=False)
v_shoulder_to_wrist = pw_target - solver.o2

v_projected = v_shoulder_to_wrist - np.dot(v_shoulder_to_wrist, a1) * a1

x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
y_ref = np.cross(a1, x_ref)

px = np.dot(v_projected, x_ref)
py = np.dot(v_projected, y_ref)

azimuth = np.arctan2(py, px)

print(f"Azimuth: {np.degrees(azimuth):.2f}°")

# Try both q1 solutions
q1_solutions = [
    oum.wrap_to_pi(azimuth - np.pi / 2),
    oum.wrap_to_pi(azimuth + np.pi / 2)
]

print(f"\nq1 solutions: {[np.degrees(q) for q in q1_solutions]}")
print(f"Target q1: {np.degrees(q1_target):.2f}°")

# Pick the correct q1 (should be 45°)
q1 = q1_solutions[0] if abs(q1_solutions[0] - q1_target) < abs(q1_solutions[1] - q1_target) else q1_solutions[1]

print(f"\nUsing q1 = {np.degrees(q1):.2f}°")

# Transform to J1's frame
rotmat1 = oum.rotmat_from_axangle(a1, q1)
v_in_j1_frame = rotmat1 @ v_shoulder_to_wrist
a2_in_j1_frame = rotmat1 @ solver.a2

print(f"\nv_in_j1_frame: {v_in_j1_frame}")
print(f"a2_in_j1_frame: {a2_in_j1_frame}")

# Project to swing plane
a2_unit = oum.unit_vec(a2_in_j1_frame, return_length=False)
v_in_swing_plane = v_in_j1_frame - np.dot(v_in_j1_frame, a2_unit) * a2_unit
d_swing = np.linalg.norm(v_in_swing_plane)

print(f"\nSwing plane projection:")
print(f"  v_in_swing_plane: {v_in_swing_plane}")
print(f"  d_swing: {d_swing:.4f} m")

# Law of cosines for q3
l2, l3 = solver.l2, solver.l3
c3 = oum.clamp(
    (d_swing * d_swing - l2 * l2 - l3 * l3) / (2 * l2 * l3),
    lo=-1.0, hi=1.0)
s3_abs = np.sqrt(max(0.0, 1.0 - c3 * c3))

print(f"\nLaw of cosines:")
print(f"  c3 = {c3:.4f}")
print(f"  s3_abs = {s3_abs:.4f}")

# For q3=0, we expect c3=1, s3=0
print(f"  Expected (for q3=0): c3=1, s3=0")

# But we're getting c3 ≠ 1, which means d_swing is wrong!
# Let's check what d_swing should be for q3=0

# When q3=0, the arm is straight from J2 to wrist
# Distance should be l2 + l3
d_straight = l2 + l3
print(f"\nFor straight arm (q3=0), d should be: {d_straight:.4f} m")
print(f"Actual d_swing: {d_swing:.4f} m")
print(f"Difference: {abs(d_swing - d_straight)*1000:.2f} mm")

# This suggests the problem is in how we compute d_swing!
# d_swing should be the distance in the swing plane perpendicular to a2

print("\n" + "="*80)
print("DIAGNOSIS:")
print("-"*80)
print("The issue is that d_swing is computed incorrectly.")
print("We're projecting v_in_j1_frame onto the swing plane,")
print("but this doesn't account for the actual 2R mechanism geometry.")
print()
print("The 2R mechanism has:")
print(f"  - J2 at origin of J1's frame")
print(f"  - First link (l2) extends from J2 along some direction")
print(f"  - Second link (l3) extends from J3 to wrist")
print()
print("We need to know: In the swing plane, where is the '零点' for q2?")
