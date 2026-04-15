"""
Debug script to understand RS007L joint geometry and verify IK calculations.
"""
import numpy as np
import one.utils.math as oum
from one.robots.manipulators.kawasaki.rs007l import rs007l

def analyze_joint_geometry():
    """Analyze the joint structure and transformations."""
    robot = rs007l.RS007L()
    
    print("=" * 80)
    print("RS007L Joint Geometry Analysis")
    print("=" * 80)
    
    # Get joints from the chain
    jnts = robot._chain.jnts
    
    for i in range(6):
        jnt = jnts[i]
        pos = jnt.gl_pos0_pos
        axis = jnt.gl_motion_ax
        print(f"\nJ{i+1}:")
        print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"  Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
        print(f"  gl_flange_tf:\n{jnt.gl_flange_tf}")
    
    # Calculate key distances
    o1 = jnts[0].gl_pos0_pos
    o2 = jnts[1].gl_pos0_pos
    o3 = jnts[2].gl_pos0_pos
    o4 = jnts[3].gl_pos0_pos
    
    print("\n" + "=" * 80)
    print("Key Distances:")
    print("=" * 80)
    print(f"||o2 - o1|| = {np.linalg.norm(o2 - o1):.6f} m (should be ~0 for spherical joint)")
    print(f"||o3 - o2|| = {np.linalg.norm(o3 - o2):.6f} m (l2, shoulder to elbow)")
    print(f"||o4 - o3|| = {np.linalg.norm(o4 - o3):.6f} m (l3, elbow to wrist)")
    
    return robot

def test_specific_configuration(robot, q1, q2, q3):
    """Test a specific joint configuration and print detailed results."""
    print("\n" + "=" * 80)
    print(f"Testing Configuration: q1={np.degrees(q1):.2f}°, q2={np.degrees(q2):.2f}°, q3={np.degrees(q3):.2f}°")
    print("=" * 80)
    
    # Set joint angles (q4, q5, q6 = 0 for now)
    qs = np.array([q1, q2, q3, 0, 0, 0], dtype=np.float32)
    robot.fk(qs)
    
    # Get positions after FK
    jnts = robot._chain.jnts
    for i in range(4):
        pos = jnts[i].gl_pos0_pos
        print(f"J{i+1} position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    
    # Get wrist center (J4) position
    wrist_pos = jnts[3].gl_pos0_pos
    tcp_pos = robot.gl_tcp_tf[:3, 3]
    
    print(f"\nWrist center (J4): [{wrist_pos[0]:.4f}, {wrist_pos[1]:.4f}, {wrist_pos[2]:.4f}]")
    print(f"TCP position: [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}]")
    
    return wrist_pos, tcp_pos

def verify_2r_geometry(robot):
    """Verify the 2R planar mechanism geometry for different q1 angles."""
    print("\n" + "=" * 80)
    print("2R Planar Mechanism Verification")
    print("=" * 80)
    
    # Get link lengths from zero configuration
    robot.fk(np.zeros(6, dtype=np.float32))
    jnts = robot._chain.jnts
    o2 = jnts[1].gl_pos0_pos
    o3 = jnts[2].gl_pos0_pos
    o4 = jnts[3].gl_pos0_pos
    
    l2 = np.linalg.norm(o3 - o2)
    l3 = np.linalg.norm(o4 - o3)
    
    print(f"l2 (shoulder to elbow) = {l2:.6f} m")
    print(f"l3 (elbow to wrist) = {l3:.6f} m")
    print(f"Max reach = {l2 + l3:.6f} m")
    print(f"Min reach = {abs(l2 - l3):.6f} m")
    
    # Test q1=0, vary q2 and q3
    print("\n--- Test Case 1: q1=0° (mechanism in YZ plane) ---")
    test_specific_configuration(robot, 0, 0, 0)
    test_specific_configuration(robot, 0, np.pi/4, 0)
    test_specific_configuration(robot, 0, np.pi/2, 0)
    
    # Test q1=90°, vary q2 and q3
    print("\n--- Test Case 2: q1=90° (mechanism in XZ plane) ---")
    test_specific_configuration(robot, np.pi/2, 0, 0)
    test_specific_configuration(robot, np.pi/2, np.pi/4, 0)
    test_specific_configuration(robot, np.pi/2, np.pi/2, 0)
    
    # Test with q3 variations
    print("\n--- Test Case 3: q3 variations ---")
    test_specific_configuration(robot, 0, 0, np.pi/4)
    test_specific_configuration(robot, 0, 0, np.pi/2)
    test_specific_configuration(robot, 0, 0, -np.pi/2)

def compare_analytical_numerical(robot, target_tcp):
    """Compare analytical IK with numerical solution."""
    print("\n" + "=" * 80)
    print(f"IK Comparison for Target TCP: [{target_tcp[0]:.3f}, {target_tcp[1]:.3f}, {target_tcp[2]:.3f}]")
    print("=" * 80)
    
    # Target orientation (pointing down)
    tgt_rotmat = oum.rotmat_from_euler(oum.pi, 0, 0)
    
    # Get analytical IK solution
    qs_list = robot.ik_tcp(tgt_pos=target_tcp, tgt_rotmat=tgt_rotmat)
    
    if qs_list:
        print(f"\nAnalytical IK found {len(qs_list)} solution(s)")
        for idx, qs in enumerate(qs_list):
            print(f"\nSolution {idx+1}:")
            print(f"  q1={np.degrees(qs[0]):.2f}°, q2={np.degrees(qs[1]):.2f}°, q3={np.degrees(qs[2]):.2f}°")
            print(f"  q4={np.degrees(qs[3]):.2f}°, q5={np.degrees(qs[4]):.2f}°, q6={np.degrees(qs[5]):.2f}°")
            
            # Verify with FK
            robot.fk(qs)
            actual_tcp = robot.gl_tcp_tf[:3, 3]
            error = np.linalg.norm(actual_tcp - target_tcp)
            print(f"  FK result: [{actual_tcp[0]:.4f}, {actual_tcp[1]:.4f}, {actual_tcp[2]:.4f}]")
            print(f"  Error: {error:.6f} m")
    else:
        print("\nAnalytical IK: No solution found")
    
    # Try numerical solver for comparison
    from one.robots.base.kinematics.ik_numba_solver import SELIKSolver
    jnt_values = np.array([jnt.motion_value for jnt in robot.manipulator_dict['rgt_arm'].jnts], dtype=np.float32)
    numerical_solver = SELIKSolver(robot.manipulator_dict['rgt_arm'])
    
    tgt_tf = np.eye(4, dtype=np.float32)
    tgt_tf[:3, :3] = tgt_rotmat
    tgt_tf[:3, 3] = target_tcp
    
    qs_numerical = numerical_solver.solve_tgt_tf_exhaustive(tgt_tf)
    
    if qs_numerical is not None:
        print(f"\nNumerical IK solution:")
        print(f"  q1={np.degrees(qs_numerical[0]):.2f}°, q2={np.degrees(qs_numerical[1]):.2f}°, q3={np.degrees(qs_numerical[2]):.2f}°")
        print(f"  q4={np.degrees(qs_numerical[3]):.2f}°, q5={np.degrees(qs_numerical[4]):.2f}°, q6={np.degrees(qs_numerical[5]):.2f}°")
        
        robot.fk(qs_numerical)
        actual_tcp = robot.gl_tcp_tf[:3, 3]
        error = np.linalg.norm(actual_tcp - target_tcp)
        print(f"  FK result: [{actual_tcp[0]:.4f}, {actual_tcp[1]:.4f}, {actual_tcp[2]:.4f}]")
        print(f"  Error: {error:.6f} m")
    else:
        print("\nNumerical IK: No solution found")

def main():
    """Main debug routine."""
    robot = analyze_joint_geometry()
    verify_2r_geometry(robot)
    
    # Test specific targets
    test_targets = [
        np.array([0.3, 0.2, 0.5], dtype=np.float32),
        np.array([0.4, 0.0, 0.4], dtype=np.float32),
        np.array([0.0, 0.3, 0.6], dtype=np.float32),
    ]
    
    for target in test_targets:
        compare_analytical_numerical(robot, target)

if __name__ == '__main__':
    main()
