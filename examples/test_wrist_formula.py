import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()

print("Testing Wrist FK vs IK Formula")
print("="*80)

# Test: Given known q4, q5, q6, compute R3_6 via FK, then verify IK formula

test_cases = [
    ("Zero", [0, 0, 0]),
    ("q5=30°", [0, np.radians(30), 0]),
    ("q4=20°, q5=25°, q6=30°", [np.radians(20), np.radians(25), np.radians(30)]),
]

solver = robot._solver

for name, wrist_angles in test_cases:
    q4, q5, q6 = wrist_angles
    
    print(f"\n{name}: q4={np.degrees(q4):.1f}°, q5={np.degrees(q5):.1f}°, q6={np.degrees(q6):.1f}°")
    print("-"*80)
    
    # FK: compute R3_6 for these wrist angles
    # Set first 3 joints to zero for simplicity
    full_config = [0, 0, 0, q4, q5, q6]
    robot.fk(qs=full_config)
    
    # Get R0_3 and R0_6
    rotmat0_3 = solver.get_rotmat_from_fk([0, 0, 0], k=3)
    rotmat0_6 = robot.gl_tcp_tf[:3, :3]
    
    # Compute R3_6
    rotmat3_6 = rotmat0_3.T @ rotmat0_6
    
    print(f"R3_6 from FK:")
    print(rotmat3_6)
    
    # Now apply the IK formula
    print(f"\nApplying IK formula:")
    
    # Current formula (line 210): c5 = R3_6[2,2]
    c5 = oum.clamp(rotmat3_6[2, 2], lo=-1.0, hi=1.0)
    q5_computed = float(np.arccos(c5))
    
    print(f"  c5 = R3_6[2,2] = {c5:.4f}")
    print(f"  q5_computed = arccos(c5) = {np.degrees(q5_computed):.2f}°")
    print(f"  q5_expected = {np.degrees(q5):.2f}°")
    
    s5 = np.sin(q5_computed)
    
    if abs(s5) > 1e-8:
        q4_computed = float(np.arctan2(
            rotmat3_6[1, 2] / s5,
            rotmat3_6[0, 2] / s5))
        q6_computed = float(np.arctan2(
            rotmat3_6[2, 1] / s5,
            -rotmat3_6[2, 0] / s5))
        
        print(f"  q4_computed = atan2(R[1,2]/s5, R[0,2]/s5) = {np.degrees(q4_computed):.2f}°")
        print(f"  q6_computed = atan2(R[2,1]/s5, -R[2,0]/s5) = {np.degrees(q6_computed):.2f}°")
        print(f"  q4_expected = {np.degrees(q4):.2f}°")
        print(f"  q6_expected = {np.degrees(q6):.2f}°")
        
        # Check errors
        q4_error = abs(oum.wrap_to_pi(q4_computed - q4))
        q5_error = abs(q5_computed - q5)
        q6_error = abs(oum.wrap_to_pi(q6_computed - q6))
        
        print(f"\nErrors:")
        print(f"  Δq4 = {np.degrees(q4_error):.2f}°")
        print(f"  Δq5 = {np.degrees(q5_error):.2f}°")
        print(f"  Δq6 = {np.degrees(q6_error):.2f}°")
        
        if q4_error < 0.01 and q5_error < 0.01 and q6_error < 0.01:
            print("  ✓ Formula is CORRECT")
        else:
            print("  ✗ Formula is WRONG")
    else:
        print("  Singularity: s5 ≈ 0")

print("\n" + "="*80)
print("Analysis: Check if the ZYZ formula works for Z-X-Z configuration")
