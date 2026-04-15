import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

print("Checking R3_6 Convention for RS007L")
print("="*80)

# At zero config, what is R3_6?
robot.fk(qs=[0, 0, 0, 0, 0, 0])
rotmat0_3 = solver.get_rotmat_from_fk([0, 0, 0], k=3)
rotmat0_6 = robot.gl_tcp_tf[:3, :3]
R36_zero = rotmat0_3.T @ rotmat0_6

print("R3_6 at zero configuration:")
print(R36_zero)
print()

# This means there's a fixed offset rotation!
# The actual rotation is: R3_6_measured = R_offset * R_ZXZ(q4, q5, q6)
# So: R_ZXZ(q4, q5, q6) = R_offset^T * R3_6_measured

# Let's identify R_offset
R_offset = R36_zero
print("R_offset (pre-rotation):")
print(R_offset)
print()

# Now test: does R_offset^T * R3_6 give us the pure ZXZ rotation?

def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)

def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,  0, 0],
                     [0,  c, -s],
                     [0,  s,  c]], dtype=np.float32)

test_cases = [
    ("Zero", [0, 0, 0, 0, 0, 0]),
    ("q5=30°", [0, 0, 0, 0, np.radians(30), 0]),
    ("Combined", [0, 0, 0, np.radians(20), np.radians(25), np.radians(30)]),
]

print("="*80)
print("Testing if R_offset^T * R3_6 = Rz(q4) * Rx(q5) * Rz(q6):")
print("-"*80)

for name, config in test_cases:
    q4, q5, q6 = config[3], config[4], config[5]
    
    # FK
    robot.fk(qs=config)
    rotmat0_3 = solver.get_rotmat_from_fk(config[:3], k=3)
    rotmat0_6 = robot.gl_tcp_tf[:3, :3]
    R36_measured = rotmat0_3.T @ rotmat0_6
    
    # Remove offset
    R_pure = R_offset.T @ R36_measured
    
    # Expected
    R_expected = Rz(q4) @ Rx(q5) @ Rz(q6)
    
    error = np.linalg.norm(R_pure - R_expected, 'fro')
    
    print(f"\n{name}: q4={np.degrees(q4):.1f}°, q5={np.degrees(q5):.1f}°, q6={np.degrees(q6):.1f}°")
    print(f"  Error: {error:.6f}")
    
    if error < 1e-5:
        print(f"  ✓ Offset formula works!")
    else:
        print(f"  ✗ Offset formula doesn't work")
        print("  R_pure:")
        print(R_pure)
        print("  R_expected:")
        print(R_expected)

print("\n" + "="*80)
print("Conclusion: The IK formula should be:")
print("  1. Compute R_pure = R_offset^T * R3_6_measured")
print("  2. Apply ZXZ formulas to R_pure")
print("where R_offset is the R3_6 at zero configuration")
