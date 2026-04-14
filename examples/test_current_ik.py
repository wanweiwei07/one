import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l

robot = rs007l.RS007L()
solver = robot._solver

print("Testing Current IK Implementation")
print("="*80)

test_cases = [
    ("Zero config", [0, 0, 0, 0, 0, 0]),
    ("q1=45° only", [np.radians(45), 0, 0, 0, 0, 0]),
    ("q1=45°, q2=30°", [np.radians(45), np.radians(30), 0, 0, 0, 0]),
]

for name, config in test_cases:
    print(f"\n{name}: {np.degrees(config)}")
    print("-"*80)
    
    # FK
    robot.fk(qs=config)
    target_tf = robot.gl_tcp_tf.copy()
    target_pos = target_tf[:3, 3]
    
    # IK
    solutions = solver.ik_all(target_tf)
    
    if solutions:
        print(f"Found {len(solutions)} solution(s)")
        
        best_error = float('inf')
        for i, sol in enumerate(solutions):
            robot.fk(qs=sol)
            fk_pos = robot.gl_tcp_tf[:3, 3]
            error = np.linalg.norm(fk_pos - target_pos)
            
            if error < best_error:
                best_error = error
            
            if error < 0.01:  # < 10mm
                print(f"  Sol {i+1}: q=[{np.degrees(sol[0]):6.1f}, {np.degrees(sol[1]):6.1f}, {np.degrees(sol[2]):6.1f}, {np.degrees(sol[3]):6.1f}, {np.degrees(sol[4]):6.1f}, {np.degrees(sol[5]):6.1f}]")
                print(f"          Error: {error*1000:.2f} mm ✓")
        
        if best_error >= 0.01:
            print(f"  ✗ Best error: {best_error*1000:.2f} mm (too large)")
    else:
        print("  No solutions found")

print("\n" + "="*80)
print("If most tests pass, the issue is only with腕部姿态. Otherwise, first3 also needs fix.")
