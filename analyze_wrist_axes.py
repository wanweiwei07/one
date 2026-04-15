import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

print("RS007L Wrist Joint Axes Analysis")
print("="*80)

# Get the wrist joint axes in zero configuration
print("\nWrist joint axes in zero configuration (world frame):")
print(f"a4 (J4 axis): {solver.a4}")
print(f"a5 (J5 axis): {solver.a5}")
print(f"a6 (J6 axis): {solver.a6}")

# Normalize
a4_unit = oum.unit_vec(solver.a4, return_length=False)
a5_unit = oum.unit_vec(solver.a5, return_length=False)
a6_unit = oum.unit_vec(solver.a6, return_length=False)

print(f"\nNormalized:")
print(f"a4_unit: {a4_unit}")
print(f"a5_unit: {a5_unit}")
print(f"a6_unit: {a6_unit}")

# Check if it's ZYZ configuration
# ZYZ means: Z-axis, Y-axis, Z-axis
# Or in general: first and third axes should be parallel

dot_4_6 = np.dot(a4_unit, a6_unit)
dot_4_5 = np.dot(a4_unit, a5_unit)
dot_5_6 = np.dot(a5_unit, a6_unit)

print(f"\nDot products:")
print(f"a4·a6 = {dot_4_6:.4f}  (should be ±1 for ZYZ)")
print(f"a4·a5 = {dot_4_5:.4f}  (should be ≈0 for ZYZ)")
print(f"a5·a6 = {dot_5_6:.4f}  (should be ≈0 for ZYZ)")

if abs(dot_4_6) > 0.9 and abs(dot_4_5) < 0.1 and abs(dot_5_6) < 0.1:
    if dot_4_6 > 0:
        print("\n✓ This is a ZYZ (or ZXZ) spherical wrist with parallel outer axes")
    else:
        print("\n✓ This is a ZYZ-like spherical wrist with antiparallel outer axes")
else:
    print("\n✗ This is NOT a standard ZYZ spherical wrist!")
    print("  The wrist IK formulation may need adjustment.")

# Check the actual configuration
print("\n" + "="*80)
print("Checking specific axis directions:")
print("-"*80)

# Z axis is typically [0,0,1] or [0,0,-1]
# X axis is typically [1,0,0] or [-1,0,0]
# Y axis is typically [0,1,0] or [0,-1,0]

def identify_axis(vec):
    vec_norm = oum.unit_vec(vec, return_length=False)
    axes = {
        '+Z': np.array([0, 0, 1], dtype=np.float32),
        '-Z': np.array([0, 0, -1], dtype=np.float32),
        '+X': np.array([1, 0, 0], dtype=np.float32),
        '-X': np.array([-1, 0, 0], dtype=np.float32),
        '+Y': np.array([0, 1, 0], dtype=np.float32),
        '-Y': np.array([0, -1, 0], dtype=np.float32),
    }
    
    for name, axis in axes.items():
        if np.allclose(vec_norm, axis, atol=0.01):
            return name
    return "Unknown"

print(f"J4 axis: {identify_axis(a4_unit)}")
print(f"J5 axis: {identify_axis(a5_unit)}")
print(f"J6 axis: {identify_axis(a6_unit)}")

# Actual configuration
print(f"\nWrist configuration: {identify_axis(a4_unit)}-{identify_axis(a5_unit)}-{identify_axis(a6_unit)}")
