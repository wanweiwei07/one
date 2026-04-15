import numpy as np
import one.utils.math as oum

print("Deriving Z-X-Z Euler Angle IK Formula")
print("="*80)

# Z-X-Z Euler angles: R = Rz(q4) * Rx(q5) * Rz(q6)
#
# Where:
#   Rz(θ) = [[cos(θ), -sin(θ), 0],
#            [sin(θ),  cos(θ), 0],
#            [0,       0,      1]]
#
#   Rx(θ) = [[1,      0,       0],
#            [0, cos(θ), -sin(θ)],
#            [0, sin(θ),  cos(θ)]]

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

print("\nZ-X-Z Euler Angles: R = Rz(q4) * Rx(q5) * Rz(q6)")
print()

# Compute symbolic result
# R = Rz(q4) * Rx(q5) * Rz(q6)
#   = [[c4, -s4, 0],   [[1,  0,   0 ],   [[c6, -s6, 0],
#      [s4,  c4, 0], *  [0,  c5, -s5], *  [s6,  c6, 0],
#      [0,   0,  1]]    [0,  s5,  c5]]    [0,   0,  1]]

# Result:
# R = [[c4*c6 - s4*c5*s6,  -c4*s6 - s4*c5*c6,   s4*s5],
#      [s4*c6 + c4*c5*s6,  -s4*s6 + c4*c5*c6,  -c4*s5],
#      [s5*s6,              s5*c6,               c5   ]]

print("Matrix elements:")
print("R[0,0] = cos(q4)*cos(q6) - sin(q4)*cos(q5)*sin(q6)")
print("R[0,1] = -cos(q4)*sin(q6) - sin(q4)*cos(q5)*cos(q6)")
print("R[0,2] = sin(q4)*sin(q5)")
print()
print("R[1,0] = sin(q4)*cos(q6) + cos(q4)*cos(q5)*sin(q6)")
print("R[1,1] = -sin(q4)*sin(q6) + cos(q4)*cos(q5)*cos(q6)")
print("R[1,2] = -cos(q4)*sin(q5)")
print()
print("R[2,0] = sin(q5)*sin(q6)")
print("R[2,1] = sin(q5)*cos(q6)")
print("R[2,2] = cos(q5)")
print()

print("="*80)
print("IK Formulas:")
print("-"*80)
print()
print("From R[2,2]:")
print("  c5 = R[2,2]")
print("  q5 = ±arccos(R[2,2])")
print()
print("If sin(q5) ≠ 0:")
print("  From R[0,2] and R[1,2]:")
print("    sin(q4)*sin(q5) = R[0,2]")
print("    -cos(q4)*sin(q5) = R[1,2]")
print("  Therefore:")
print("    q4 = atan2(R[0,2], -R[1,2])")
print()
print("  From R[2,0] and R[2,1]:")
print("    sin(q5)*sin(q6) = R[2,0]")
print("    sin(q5)*cos(q6) = R[2,1]")
print("  Therefore:")
print("    q6 = atan2(R[2,0], R[2,1])")
print()
print("If sin(q5) = 0 (singularity):")
print("  q4 and q6 are not uniquely determined")
print("  Set q4 = 0, then:")
print("    q6 = atan2(-R[0,1], R[0,0])  (if c5 = 1)")
print("    q6 = atan2(R[0,1], -R[0,0])  (if c5 = -1)")

print()
print("="*80)
print("Testing derived formulas:")
print("-"*80)

test_cases = [
    (0, 0, 0),
    (0, np.radians(30), 0),
    (np.radians(20), np.radians(25), np.radians(30)),
    (np.radians(45), np.radians(60), np.radians(-30)),
]

for q4, q5, q6 in test_cases:
    print(f"\nInput: q4={np.degrees(q4):.1f}°, q5={np.degrees(q5):.1f}°, q6={np.degrees(q6):.1f}°")
    
    # FK
    R = Rz(q4) @ Rx(q5) @ Rz(q6)
    
    # IK
    c5 = R[2, 2]
    q5_ik = np.arccos(np.clip(c5, -1, 1))
    s5 = np.sin(q5_ik)
    
    if abs(s5) > 1e-8:
        q4_ik = np.arctan2(R[0, 2], -R[1, 2])
        q6_ik = np.arctan2(R[2, 0], R[2, 1])
    else:
        q4_ik = 0
        if c5 > 0:  # c5 = 1
            q6_ik = np.arctan2(-R[0, 1], R[0, 0])
        else:  # c5 = -1
            q6_ik = np.arctan2(R[0, 1], -R[0, 0])
    
    print(f"  IK: q4={np.degrees(q4_ik):.1f}°, q5={np.degrees(q5_ik):.1f}°, q6={np.degrees(q6_ik):.1f}°")
    
    # Verify
    R_verify = Rz(q4_ik) @ Rx(q5_ik) @ Rz(q6_ik)
    error = np.linalg.norm(R - R_verify, 'fro')
    
    if error < 1e-6:
        print(f"  ✓ Verification: error = {error:.2e}")
    else:
        print(f"  ✗ Verification: error = {error:.2e}")
