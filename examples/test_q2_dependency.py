import numpy as np
import one.robots.manipulators.kawasaki.rs007l.rs007l as rs007l
import one.utils.math as oum

robot = rs007l.RS007L()

# Test: For the same target TCP position, check solutions
target_pos = np.array([0.3, 0.3, 1.2], dtype=np.float32)
target_tf = np.eye(4, dtype=np.float32)
target_tf[:3, 3] = target_pos

print("Testing q2/q3 dependence on q1 for same TCP target:")
print(f"Target position: {target_pos}")
print("="*80)

ik_solver = robot._solver

try:
    solutions = ik_solver.ik_all(target_tf)
    
    if solutions:
        print(f"Found {len(solutions)} solutions:")
        for i, sol in enumerate(solutions):
            q_deg = np.degrees(sol)
            print(f"Sol {i+1}: q1={q_deg[0]:6.2f}°, q2={q_deg[1]:6.2f}°, q3={q_deg[2]:6.2f}°")
            
            # Verify with FK
            robot.set_jnt_values(sol)
            fk_tcp = robot.get_gl_tcp()[:3, 3]
            error = np.linalg.norm(fk_tcp - target_pos)
            print(f"        FK TCP: {fk_tcp}, error: {error*1000:.2f}mm")
    else:
        print("No IK solutions found")
except Exception as e:
    print(f"Error during IK: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Expected behavior:")
print("- Solutions with q1 ≈ 45° and q1 ≈ 225° should exist (two arm configurations)")
print("- For each q1, q2 and q3 should produce the correct reach and height")
print("- If q2/q3 are wildly different between q1 solutions, there's a coupling bug")
