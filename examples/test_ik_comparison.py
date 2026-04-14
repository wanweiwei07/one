"""
Test to compare analytical vs numerical IK and understand the mismatch.
"""
import numpy as np
import one.utils.math as oum
from one.robots.manipulators.kawasaki.rs007l import rs007l
from one.robots.base.kinematics.numik_sel import SELIKSolver

def test_target(robot, target_wrist, target_orientation_euler=(np.pi, 0, 0)):
    """Test IK for a specific target."""
    print("\n" + "=" * 80)
    print(f"Target wrist center: [{target_wrist[0]:.3f}, {target_wrist[1]:.3f}, {target_wrist[2]:.3f}]")
    print("=" * 80)
    
    # Construct target TCP (wrist + offset)
    tgt_rotmat = oum.rotmat_from_euler(*target_orientation_euler)
    
    # Get TCP position from wrist position
    # Need to account for wrist-to-TCP offset
    # From anaik.py: self.ow_6 = self.jnt_zero_tfs[5][:3, 3] - self.ow
    solver = robot.get_solver(robot._chain)
    ow = solver.ow
    ow_6 = solver.ow_6
    
    print(f"\nWrist center in zero config: {ow}")
    print(f"TCP offset from wrist: {ow_6}")
    
    # TCP = wrist + rotmat @ ow_6
    target_tcp = target_wrist + tgt_rotmat @ ow_6
    print(f"Target TCP: [{target_tcp[0]:.3f}, {target_tcp[1]:.3f}, {target_tcp[2]:.3f}]")
    
    # Get analytical IK solution
    print("\n--- Analytical IK ---")
    qs_list = robot.ik_tcp(tgt_pos=target_tcp, tgt_rotmat=tgt_rotmat)
    
    if qs_list:
        print(f"Found {len(qs_list)} solution(s)")
        for idx, qs in enumerate(qs_list):
            print(f"\nSolution {idx+1}:")
            print(f"  q1={np.degrees(qs[0]):7.2f}°, q2={np.degrees(qs[1]):7.2f}°, q3={np.degrees(qs[2]):7.2f}°")
            print(f"  q4={np.degrees(qs[3]):7.2f}°, q5={np.degrees(qs[4]):7.2f}°, q6={np.degrees(qs[5]):7.2f}°")
            
            # Verify with FK
            robot.fk(qs)
            actual_tcp = robot.gl_tcp_tf[:3, 3]
            actual_wrist = robot.gl_lnk_tfarr[4][:3, 3]  # Link 4 is wrist center
            
            tcp_error = np.linalg.norm(actual_tcp - target_tcp)
            wrist_error = np.linalg.norm(actual_wrist - target_wrist)
            
            print(f"  Actual TCP: [{actual_tcp[0]:.4f}, {actual_tcp[1]:.4f}, {actual_tcp[2]:.4f}], error={tcp_error:.6f}m")
            print(f"  Actual wrist: [{actual_wrist[0]:.4f}, {actual_wrist[1]:.4f}, {actual_wrist[2]:.4f}], error={wrist_error:.6f}m")
    else:
        print("No solution found")
    
    # Get numerical IK solution
    print("\n--- Numerical IK ---")
    numerical_solver = SELIKSolver(robot._chain)
    
    tgt_tf = np.eye(4, dtype=np.float32)
    tgt_tf[:3, :3] = tgt_rotmat
    tgt_tf[:3, 3] = target_tcp
    
    qs_numerical = numerical_solver.solve_tgt_tf_exhaustive(tgt_tf)
    
    if qs_numerical is not None:
        print(f"Solution:")
        print(f"  q1={np.degrees(qs_numerical[0]):7.2f}°, q2={np.degrees(qs_numerical[1]):7.2f}°, q3={np.degrees(qs_numerical[2]):7.2f}°")
        print(f"  q4={np.degrees(qs_numerical[3]):7.2f}°, q5={np.degrees(qs_numerical[4]):7.2f}°, q6={np.degrees(qs_numerical[5]):7.2f}°")
        
        robot.fk(qs_numerical)
        actual_tcp = robot.gl_tcp_tf[:3, 3]
        actual_wrist = robot.gl_lnk_tfarr[4][:3, 3]
        
        tcp_error = np.linalg.norm(actual_tcp - target_tcp)
        wrist_error = np.linalg.norm(actual_wrist - target_wrist)
        
        print(f"  Actual TCP: [{actual_tcp[0]:.4f}, {actual_tcp[1]:.4f}, {actual_tcp[2]:.4f}], error={tcp_error:.6f}m")
        print(f"  Actual wrist: [{actual_wrist[0]:.4f}, {actual_wrist[1]:.4f}, {actual_wrist[2]:.4f}], error={wrist_error:.6f}m")
    else:
        print("No solution found")

def main():
    robot = rs007l.RS007L()
    
    # Test several simple targets
    test_targets = [
        np.array([0.4, 0.0, 0.5], dtype=np.float32),   # Forward in Y-Z plane
        np.array([0.3, 0.2, 0.5], dtype=np.float32),   # Off to the side
        np.array([0.0, 0.3, 0.6], dtype=np.float32),   # Pure Y direction
    ]
    
    for target in test_targets:
        test_target(robot, target)

if __name__ == '__main__':
    main()
