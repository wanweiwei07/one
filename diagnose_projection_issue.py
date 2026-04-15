import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

q1 = np.radians(45)
q2 = np.radians(30)
q3 = np.radians(0)

robot.fk(qs=[q1, q2, q3, 0, 0, 0])
tcp = robot.gl_tcp_tf[:3, 3]
rot = robot.gl_tcp_tf[:3, :3]
wrist_target = tcp - rot @ solver.ow_6

a1 = oum.unit_vec(solver.a1, return_length=False)
R_j1 = oum.rotmat_from_axangle(a1, q1)

v_target = R_j1 @ (wrist_target - solver.o2)

a2_in_j1 = R_j1 @ solver.a2
a2_unit = oum.unit_vec(a2_in_j1, return_length=False)

v_target_proj = v_target - np.dot(v_target, a2_unit) * a2_unit

print("Projection Issue Analysis")
print("="*80)
print(f"v_target (3D): {v_target}")
print(f"||v_target||: {np.linalg.norm(v_target):.4f} m")
print(f"Expected (l2+l3): {solver.l2 + solver.l3:.4f} m")
print()

print(f"v_target_proj (swing plane): {v_target_proj}")
print(f"||v_target_proj||: {np.linalg.norm(v_target_proj):.4f} m")
print()

# The issue: when we project onto the swing plane, we lose the component along a2
# So v_target_proj is SHORTER than the actual arm length!

# The arm sweeps out a cone as q2 varies
# The axis of the cone is a2
# The half-angle of the cone depends on q2

# When q2=0, the arm points along dir_zero
# When q2=30°, the arm has rotated 30° around a2

# The key insight: the arm length (l2+l3) stays constant in 3D
# But its projection onto the swing plane changes!

# At q2=0:
robot.fk(qs=[q1, 0, 0, 0, 0, 0])
tcp_q2_0 = robot.gl_tcp_tf[:3, 3]
rot_q2_0 = robot.gl_tcp_tf[:3, :3]
wrist_q2_0 = tcp_q2_0 - rot_q2_0 @ solver.ow_6

v_q2_0 = R_j1 @ (wrist_q2_0 - solver.o2)
v_q2_0_proj = v_q2_0 - np.dot(v_q2_0, a2_unit) * a2_unit

print(f"At q2=0:")
print(f"  v_q2_0_proj: {v_q2_0_proj}")
print(f"  ||v_q2_0_proj||: {np.linalg.norm(v_q2_0_proj):.4f} m")
print()

# So the projection length is the same! That's because both have q3=0

# The problem must be in how we interpret the angle...

# Let me think about this differently:
# q2 is the angle of rotation around a2
# When q2=0, the arm points in direction dir_zero
# When q2=θ, the arm has rotated θ degrees around a2

# To find the angle, we should use the Rodrigues rotation formula
# or simply project and measure angle in the plane perpendicular to a2

# Current approach: project both onto swing plane, measure angle
# This should work IF we're measuring the right thing

# Let me check: what if a2 is not perfectly perpendicular to the zero direction?

dot_a2_zero = np.dot(a2_unit, v_q2_0)
print(f"a2 · v_q2_0: {dot_a2_zero:.6f}")
print(f"Should be 0 if perpendicular")
print()

# Ah! It's NOT zero! This means v_q2_0 has a component along a2!
# So when we project, we're losing information

# The correct approach: don't project onto swing plane
# Instead, measure the rotation angle directly using the rotation axis formula

print("="*80)
print("The issue: v_q2_0 is NOT perpendicular to a2!")
print("This means the 2R mechanism doesn't work in a plane perpendicular to a2")
print("We need a different geometric model.")
