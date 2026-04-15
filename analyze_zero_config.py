import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

print("Analyzing RS007L Zero Configuration Geometry")
print("="*80)

# Zero config
robot.fk(qs=[0, 0, 0, 0, 0, 0])

print("\nJoint positions in zero config:")
for i in range(7):
    pos = robot.gl_lnk_tfarr[i][:3, 3]
    print(f"  Link {i}: {pos}")

print("\nKey distances:")
print(f"  o1 to o2: {np.linalg.norm(solver.o2 - solver.o1):.4f} m")
print(f"  o2 to o3: {np.linalg.norm(solver.o3 - solver.o2):.4f} m")
print(f"  o3 to wrist center: {np.linalg.norm(solver.ow - solver.o3):.4f} m")

print(f"\n  l2 (computed): {solver.l2:.4f} m")
print(f"  l3 (computed): {solver.l3:.4f} m")

# When q2=0, where does J3 go?
print("\n" + "="*80)
print("Testing q2 and q3 geometry:")
print("-"*80)

configs = [
    ("q2=0, q3=0", [0, 0, 0, 0, 0, 0]),
    ("q2=30°, q3=0", [0, np.radians(30), 0, 0, 0, 0]),
    ("q2=0, q3=30°", [0, 0, np.radians(30), 0, 0, 0]),
]

for name, config in configs:
    robot.fk(qs=config)
    j2_pos = robot.gl_lnk_tfarr[2][:3, 3]  # After J2
    j3_pos = robot.gl_lnk_tfarr[3][:3, 3]  # After J3
    wrist_pos = robot.gl_lnk_tfarr[4][:3, 3]  # J5 position (close to wrist center)
    
    print(f"\n{name}:")
    print(f"  J2 position: {j2_pos}")
    print(f"  J3 position: {j3_pos}")
    print(f"  J2→J3 direction: {j3_pos - j2_pos}")
    print(f"  J3→wrist direction: {wrist_pos - j3_pos}")

# Check: what is the "natural" 2R plane?
print("\n" + "="*80)
print("2R Mechanism Analysis:")
print("-"*80)

# In zero config, with q1=0, q2=0, q3=0
# The arm is pointing straight up in +Z direction
# When we rotate q2, the arm should swing in a plane perpendicular to a2

robot.fk(qs=[0, 0, 0, 0, 0, 0])
o2_zero = robot.gl_lnk_tfarr[2][:3, 3]
o3_zero = robot.gl_lnk_tfarr[3][:3, 3]

robot.fk(qs=[0, np.radians(30), 0, 0, 0, 0])
o3_q2_30 = robot.gl_lnk_tfarr[3][:3, 3]

robot.fk(qs=[0, 0, np.radians(30), 0, 0, 0])
o3_q3_30 = robot.gl_lnk_tfarr[3][:3, 3]

print(f"o3 at q2=0, q3=0: {o3_zero}")
print(f"o3 at q2=30°, q3=0: {o3_q2_30}")
print(f"o3 at q2=0, q3=30°: {o3_q3_30}")

# The 2R plane should be defined by the motion of o3 as q2 and q3 vary
v1 = o3_q2_30 - o3_zero
v2 = o3_q3_30 - o3_zero

print(f"\nMotion vectors:")
print(f"  Δo3 when q2 changes: {v1}")
print(f"  Δo3 when q3 changes: {v2}")

# The normal to the 2R plane
normal = np.cross(v1, v2)
normal_unit = oum.unit_vec(normal, return_length=False)

print(f"\n2R plane normal: {normal_unit}")
print(f"a2 (J2 axis): {solver.a2}")
print(f"a2 normalized: {oum.unit_vec(solver.a2, return_length=False)}")

# They should be parallel (or antiparallel)
dot_product = np.dot(normal_unit, oum.unit_vec(solver.a2, return_length=False))
print(f"\nDot product (should be ±1): {dot_product:.4f}")
