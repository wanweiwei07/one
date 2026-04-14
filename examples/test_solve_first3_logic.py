import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

print("Testing _solve_first3 Logic")
print("="*80)

# Test: q1=45°, q2=30°, q3=0°
config = [np.radians(45), np.radians(30), 0, 0, 0, 0]

robot.fk(qs=config)
tcp_pos = robot.gl_tcp_tf[:3, 3]
tcp_rot = robot.gl_tcp_tf[:3, :3]

# Compute wrist center
pw_target = tcp_pos - tcp_rot @ solver.ow_6

print(f"Target config: q1={np.degrees(config[0]):.1f}°, q2={np.degrees(config[1]):.1f}°, q3={np.degrees(config[2]):.1f}°")
print(f"Target wrist center: {pw_target}")
print()

# Call _solve_first3
solutions = solver._solve_first3(pw_target)

print(f"_solve_first3 returned {len(solutions)} solution(s):")
for i, (q1, q2, q3) in enumerate(solutions):
    print(f"  Sol {i+1}: q1={np.degrees(q1):7.2f}°, q2={np.degrees(q2):7.2f}°, q3={np.degrees(q3):7.2f}°")
    
    # Verify
    robot.fk(qs=[q1, q2, q3, 0, 0, 0])
    wrist_computed = robot.gl_tcp_tf[:3, 3] - robot.gl_tcp_tf[:3, :3] @ solver.ow_6
    
    error = np.linalg.norm(wrist_computed - pw_target)
    print(f"         Wrist error: {error*1000:.2f} mm")
    
    if error < 0.001:
        print(f"         ✓ CORRECT")
    else:
        print(f"         ✗ WRONG")

print()
print("="*80)
print("Expected at least one solution close to: q1=45°, q2=30°, q3=0°")
