import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()

print("Testing Wrist (q4, q5, q6) IK")
print("="*80)

# Test case 1: Known configuration
test_configs = [
    ("All zeros", [0, 0, 0, 0, 0, 0]),
    ("q4=45°", [0, 0, 0, np.radians(45), 0, 0]),
    ("q5=30°", [0, 0, 0, 0, np.radians(30), 0]),
    ("q6=60°", [0, 0, 0, 0, 0, np.radians(60)]),
    ("Combined", [np.radians(45), np.radians(30), 0, np.radians(20), np.radians(25), np.radians(30)]),
]

solver = robot._solver

for name, qs_input in test_configs:
    print(f"\n{name}: {np.degrees(qs_input)}")
    print("-"*80)
    
    # FK to get target pose
    robot.fk(qs=qs_input)
    target_tcp_tf = robot.gl_tcp_tf.copy()
    target_pos = target_tcp_tf[:3, 3]
    target_rot = target_tcp_tf[:3, :3]
    
    # Try IK
    try:
        solutions = solver.ik_all(target_tcp_tf)
        
        if solutions and len(solutions) > 0:
            print(f"Found {len(solutions)} solution(s)")
            
            # Check each solution
            for i, sol in enumerate(solutions):
                robot.fk(qs=sol)
                fk_tcp_tf = robot.gl_tcp_tf
                fk_pos = fk_tcp_tf[:3, 3]
                fk_rot = fk_tcp_tf[:3, :3]
                
                pos_error = np.linalg.norm(fk_pos - target_pos)
                
                # Rotation error using Frobenius norm of difference
                rot_error = np.linalg.norm(fk_rot - target_rot, 'fro')
                
                q_deg = np.degrees(sol)
                print(f"  Sol {i+1}: q=[{q_deg[0]:6.1f}, {q_deg[1]:6.1f}, {q_deg[2]:6.1f}, {q_deg[3]:6.1f}, {q_deg[4]:6.1f}, {q_deg[5]:6.1f}]")
                print(f"         Pos error: {pos_error*1000:.2f} mm, Rot error: {rot_error:.4f}")
                
                if pos_error < 0.001 and rot_error < 0.01:
                    print(f"         ✓ GOOD")
                else:
                    print(f"         ✗ BAD (wrist orientation wrong)")
                    
                    # Show the wrist angles specifically
                    print(f"         Input q4,q5,q6: {np.degrees(qs_input[3:6])}")
                    print(f"         IK    q4,q5,q6: {q_deg[3:6]}")
        else:
            print("  No solutions found")
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("If rotation errors are large, the wrist IK solver needs fixing.")
