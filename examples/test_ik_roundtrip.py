"""
Simple test to compare analytical IK with known good configurations.
"""
import numpy as np
import one.utils.math as oum
from one.robots.manipulators.kawasaki.rs007l import rs007l

def test_fk_to_ik(robot, qs_input, description):
    """Test: FK(qs) → IK(TCP) → should return qs"""
    print("\n" + "=" * 80)
    print(f"Test: {description}")
    print("=" * 80)
    
    # Forward kinematics
    robot.fk(qs_input)
    target_tcp = robot.gl_tcp_tf[:3, 3].copy()
    target_rotmat = robot.gl_tcp_tf[:3, :3].copy()
    wrist_pos = robot.gl_lnk_tfarr[4][:3, 3].copy()
    
    print(f"\nInput angles:")
    print(f"  q1={np.degrees(qs_input[0]):7.2f}°, q2={np.degrees(qs_input[1]):7.2f}°, q3={np.degrees(qs_input[2]):7.2f}°")
    print(f"  q4={np.degrees(qs_input[3]):7.2f}°, q5={np.degrees(qs_input[4]):7.2f}°, q6={np.degrees(qs_input[5]):7.2f}°")
    print(f"\nFK result:")
    print(f"  TCP: [{target_tcp[0]:.4f}, {target_tcp[1]:.4f}, {target_tcp[2]:.4f}]")
    print(f"  Wrist: [{wrist_pos[0]:.4f}, {wrist_pos[1]:.4f}, {wrist_pos[2]:.4f}]")
    
    # Inverse kinematics
    print(f"\nAnalytical IK:")
    qs_list = robot.ik_tcp(tgt_pos=target_tcp, tgt_rotmat=target_rotmat)
    
    if qs_list:
        print(f"  Found {len(qs_list)} solution(s)")
        
        # Check if any solution matches the input
        best_match = None
        best_error = float('inf')
        
        for idx, qs in enumerate(qs_list):
            angular_error = np.linalg.norm(qs - qs_input)
            
            # Also check FK error
            robot.fk(qs)
            actual_tcp = robot.gl_tcp_tf[:3, 3]
            actual_wrist = robot.gl_lnk_tfarr[4][:3, 3]
            tcp_error = np.linalg.norm(actual_tcp - target_tcp)
            wrist_error = np.linalg.norm(actual_wrist - wrist_pos)
            
            print(f"\n  Solution {idx+1}:")
            print(f"    q1={np.degrees(qs[0]):7.2f}°, q2={np.degrees(qs[1]):7.2f}°, q3={np.degrees(qs[2]):7.2f}°")
            print(f"    q4={np.degrees(qs[3]):7.2f}°, q5={np.degrees(qs[4]):7.2f}°, q6={np.degrees(qs[5]):7.2f}°")
            print(f"    FK TCP error: {tcp_error:.6f}m, Wrist error: {wrist_error:.6f}m")
            print(f"    Angular distance from input: {angular_error:.6f} rad")
            
            if tcp_error < best_error:
                best_error = tcp_error
                best_match = qs
        
        if best_error < 1e-3:
            print(f"\n  ✓ PASS: Best solution has TCP error {best_error:.6f}m")
        else:
            print(f"\n  ✗ FAIL: Best TCP error {best_error:.6f}m (should be < 0.001m)")
    else:
        print(f"  ✗ FAIL: No solution found")

def main():
    robot = rs007l.RS007L()
    
    # Test 1: Zero configuration
    test_fk_to_ik(
        robot,
        np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
        "Zero configuration"
    )
    
    # Test 2: Simple q1 rotation
    test_fk_to_ik(
        robot,
        np.array([np.pi/4, 0, 0, 0, 0, 0], dtype=np.float32),
        "q1=45° only"
    )
    
    # Test 3: q1 and q2
    test_fk_to_ik(
        robot,
        np.array([np.pi/4, np.pi/6, 0, 0, 0, 0], dtype=np.float32),
        "q1=45°, q2=30°"
    )
    
    # Test 4: All first 3 joints
    test_fk_to_ik(
        robot,
        np.array([np.pi/4, np.pi/6, -np.pi/4, 0, 0, 0], dtype=np.float32),
        "q1=45°, q2=30°, q3=-45°"
    )
    
    # Test 5: With wrist joints
    test_fk_to_ik(
        robot,
        np.array([np.pi/4, np.pi/6, -np.pi/4, np.pi/3, np.pi/4, np.pi/6], dtype=np.float32),
        "Full configuration"
    )

if __name__ == '__main__':
    main()
