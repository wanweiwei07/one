"""
Test q2 calculation in isolation (with q1=0, q3=0).
This isolates q2 geometry from q1/q3 complications.
"""
import numpy as np
from one.robots.manipulators.kawasaki.rs007l.rs007l import RS007L

def test_q2_only():
    """Test FK->IK with only q2 varying, q1=q3=0"""
    robot = RS007L()
    
    # Test multiple q2 values
    test_angles = [0, 15, 30, 45, 60, -15, -30]
    
    print("Testing q2 calculation (q1=0, q3=0):")
    print("=" * 70)
    
    for q2_deg in test_angles:
        q2_rad = np.deg2rad(q2_deg)
        
        # FK with q = [0, q2, 0, 0, 0, 0]
        q_input = np.array([0, q2_rad, 0, 0, 0, 0], dtype=np.float32)
        robot.fk(q_input)
        target_pos = robot.gl_tcp_tf[:3, 3].copy()
        target_rot = robot.gl_tcp_tf[:3, :3].copy()
        
        # IK
        ik_solutions = robot.ik_tcp(tgt_pos=target_pos, tgt_rotmat=target_rot)
        
        if not ik_solutions:
            print(f"\nq2 = {q2_deg:6.1f}°: NO SOLUTION FOUND")
            continue
        
        # Find closest solution
        best_sol = None
        best_error = float('inf')
        
        for sol in ik_solutions:
            error = np.linalg.norm(sol - q_input)
            if error < best_error:
                best_error = error
                best_sol = sol
        
        # Convert to degrees for display
        sol_deg = np.rad2deg(best_sol)
        input_deg = np.rad2deg(q_input)
        
        # Verify FK of IK solution
        robot.fk(best_sol)
        actual_pos = robot.gl_tcp_tf[:3, 3].copy()
        pos_error = np.linalg.norm(actual_pos - target_pos)
        
        print(f"\nq2 = {q2_deg:6.1f}°:")
        print(f"  Target pos: {target_pos}")
        print(f"  Input:  q=[{input_deg[0]:6.1f}, {input_deg[1]:6.1f}, {input_deg[2]:6.1f}, ...]")
        print(f"  IK sol: q=[{sol_deg[0]:6.1f}, {sol_deg[1]:6.1f}, {sol_deg[2]:6.1f}, ...]")
        print(f"  q2 error: {sol_deg[1] - input_deg[1]:+7.2f}°")
        print(f"  Position error: {pos_error*1000:.2f} mm")
        
        if abs(sol_deg[1] - input_deg[1]) > 1.0:
            print(f"  ❌ FAILED - q2 error too large")
        else:
            print(f"  ✓ PASSED")

if __name__ == '__main__':
    test_q2_only()
