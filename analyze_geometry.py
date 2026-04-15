import numpy as np
import one.utils.math as oum

# RS007L geometry in zero config
a1 = np.array([0, 0, -1], dtype=np.float32)  # J1 axis (points down)
a2 = np.array([-1, 0, 0], dtype=np.float32)  # J2 axis in zero config

o1 = np.array([0, 0, 0.36], dtype=np.float32)
o2 = np.array([0, 0, 0.36], dtype=np.float32)  # J1 and J2 origins coincide
o3 = np.array([0.0925, 0, 0.36], dtype=np.float32)  # J3 origin in zero config
ow = np.array([0, 0, 1.29], dtype=np.float32)  # Wrist center

print("RS007L Zero Configuration Geometry")
print("="*80)
print(f"a1 (J1 axis): {a1}")
print(f"a2 (J2 axis): {a2}")
print(f"o1 (J1 origin): {o1}")
print(f"o2 (J2 origin): {o2}")
print(f"o3 (J3 origin): {o3}")
print(f"ow (wrist center): {ow}")
print()

# Test case: q1=45°, q2=30°, q3=0°
q1 = np.radians(45)
q2 = np.radians(30)
q3 = np.radians(0)

print(f"Test: q1={np.degrees(q1)}°, q2={np.degrees(q2)}°, q3={np.degrees(q3)}°")
print("="*80)

# After rotating by q1 around a1
R1 = oum.rotmat_from_axangle(a1, q1)
print(f"\nR1 (rotation by q1={np.degrees(q1)}° around a1):")
print(R1)

# In J1's frame (rotated by -q1 to "undo" the rotation)
rotmat1_inv = oum.rotmat_from_axangle(a1, -q1)
a2_in_j1 = rotmat1_inv @ a2
print(f"\na2 in J1's rotated frame: {a2_in_j1}")

# Check: what does a1 look like in J1's frame?
a1_in_j1 = rotmat1_inv @ a1
print(f"a1 in J1's rotated frame: {a1_in_j1}")
print("(Should still be [0,0,-1] since rotation is around a1 itself)")

# The swing plane should be perpendicular to a2_in_j1
# Build coordinate system in swing plane
a2_unit = oum.unit_vec(a2_in_j1, return_length=False)

# Current code does: dir1 = cross(a2_unit, a1)
# But a1 is in world frame, not J1 frame!
dir1_wrong = np.cross(a2_unit, a1)  # MIXING COORDINATE SYSTEMS!
print(f"\ndir1 (current, WRONG - mixing frames): {dir1_wrong}")

# Correct: use a1_in_j1
dir1_correct = np.cross(a2_unit, a1_in_j1)
print(f"dir1 (correct - both in J1 frame): {dir1_correct}")

# But since a1_in_j1 == a1 for rotation around a1, they should be the same!
print(f"\nAre they equal? {np.allclose(dir1_wrong, dir1_correct)}")

print("\n" + "="*80)
print("CONCLUSION:")
print("The issue is NOT that we're mixing coordinate systems for dir1.")
print("Since we rotate around a1, and a1 is the Z-axis, a1 doesn't change.")
print()
print("The REAL issue must be in how we define the 2R mechanism's origin point")
print("or how we account for the zero_tf rotations in J2.")
