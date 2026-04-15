"""
Debug q2 calculation step-by-step to understand the geometric issue.
"""
import numpy as np
from one.robots.manipulators.kawasaki.rs007l.rs007l import RS007L
import one.utils.math as oum

def debug_q2_case(q2_deg):
    """Debug a specific q2 angle"""
    robot = RS007L()
    q2_rad = np.deg2rad(q2_deg)
    
    # FK with q = [0, q2, 0, 0, 0, 0]
    q_input = np.array([0, q2_rad, 0, 0, 0, 0], dtype=np.float32)
    robot.fk(q_input)
    target_tcp = robot.gl_tcp_tf[:3, 3].copy()
    
    print(f"\n{'='*80}")
    print(f"Debugging q2 = {q2_deg}°")
    print(f"{'='*80}")
    print(f"\nFK result:")
    print(f"  TCP position: {target_tcp}")
    print(f"  Expected q: [0, {q2_deg}°, 0, 0, 0, 0]")
    
    # Now manually compute IK step by step
    # Get robot geometry from anaik instance
    solver = robot.get_solver(robot._chain)
    j1_origin = solver.o1
    j2_origin = solver.o2
    wrist_center_zero = solver.ow
    l2 = solver.l2
    l3 = solver.l3
    
    print(f"\n Robot geometry:")
    print(f"  J1 origin: {j1_origin}")
    print(f"  J2 origin: {j2_origin}")
    print(f"  Wrist center (zero): {wrist_center_zero}")
    print(f"  l2 = {l2:.4f}, l3 = {l3:.4f}")
    print(f"  l2 + l3 = {l2 + l3:.4f}")
    
    # Wrist center in world frame (assuming zero wrist angles)
    # For zero wrist angles, wrist center should be at wrist_center_zero rotated by q1, q2, q3
    robot_zeros = robot.fk(np.zeros(6, dtype=np.float32))
    j1_zero_frame = robot.gl_lnk_tfarr[0]
    j2_zero_frame = robot.gl_lnk_tfarr[1]
    j3_zero_frame = robot.gl_lnk_tfarr[2]
    j4_zero_frame = robot.gl_lnk_tfarr[3]
    
    robot.fk(q_input)
    j1_frame = robot.gl_lnk_tfarr[0]
    j2_frame = robot.gl_lnk_tfarr[1]
    j3_frame = robot.gl_lnk_tfarr[2]
    j4_frame = robot.gl_lnk_tfarr[3]  # This is wrist center
    
    wrist_center_actual = j4_frame[:3, 3]
    
    print(f"\n FK joint frames:")
    print(f"  J1 position: {j1_frame[:3, 3]}")
    print(f"  J2 position: {j2_frame[:3, 3]}")
    print(f"  J3 position: {j3_frame[:3, 3]}")
    print(f"  J4 position (wrist): {wrist_center_actual}")
    
    # Step 1: Compute wrist center from TCP
    # For IK, we'd subtract TCP offset from target
    tcp_offset = solver.ow_6
    print(f"\n TCP offset: {tcp_offset}")
    
    # The wrist center should be TCP - tcp_offset * R_target
    # But for now, let's just use the actual wrist center
    wrist_target = wrist_center_actual
    
    print(f"\n Wrist center target: {wrist_target}")
    
    # Step 2: Vector from J2 to wrist
    v_j2_to_wrist = wrist_target - j2_origin
    d_total = np.linalg.norm(v_j2_to_wrist)
    
    print(f"\n Position IK:")
    print(f"  Vector J2→wrist: {v_j2_to_wrist}")
    print(f"  Distance: {d_total:.4f}")
    
    # Step 3: Compute q1 (azimuth)
    q1_calc = np.arctan2(v_j2_to_wrist[1], v_j2_to_wrist[0])
    print(f"  q1 (azimuth): {np.rad2deg(q1_calc):.2f}°  (expected: 0°)")
    
    # Step 4: Compute q3 (elbow)
    c3 = (d_total**2 - l2**2 - l3**2) / (2 * l2 * l3)
    print(f"  c3 = {c3:.4f}")
    if abs(c3) > 1.0:
        print(f"  ❌ c3 out of range! No solution exists.")
        return
    
    s3 = np.sqrt(1 - c3**2)
    q3_calc = np.arctan2(s3, c3)
    print(f"  q3 (elbow): {np.rad2deg(q3_calc):.2f}°  (expected: 0°)")
    
    # Step 5: Compute q2
    # This is where the problem is!
    # Get J1's frame axes
    a1 = j1_frame[:3, 2]  # Z-axis (J1 rotation axis)
    a2 = j2_frame[:3, 2]  # Z-axis (J2 rotation axis)
    
    print(f"\n q2 calculation:")
    print(f"  J1 Z-axis (a1): {a1}")
    print(f"  J2 Z-axis (a2): {a2}")
    
    # In J1 frame, what direction should the arm point at q2=0?
    # Let's check what direction it points at q2=0 in reality
    robot.fk(np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))
    j2_at_zero = robot.gl_lnk_tfarr[1]
    j4_at_zero = robot.gl_lnk_tfarr[3]
    arm_zero_vec = j4_at_zero[:3, 3] - j2_at_zero[:3, 3]
    arm_zero_dir = arm_zero_vec / np.linalg.norm(arm_zero_vec)
    
    print(f"  Arm direction at q2=0: {arm_zero_dir}")
    
    # Current arm direction
    arm_current_vec = wrist_target - j2_origin
    arm_current_dir = arm_current_vec / np.linalg.norm(arm_current_vec)
    
    print(f"  Arm direction at q2={q2_deg}°: {arm_current_dir}")
    
    # The angle between these should be q2
    dot_product = np.dot(arm_zero_dir, arm_current_dir)
    angle_between = np.arccos(np.clip(dot_product, -1, 1))
    
    print(f"  Angle between: {np.rad2deg(angle_between):.2f}°  (expected: {abs(q2_deg)}°)")
    
    # Now figure out the sign
    # Cross product tells us rotation direction
    cross = np.cross(arm_zero_dir, arm_current_dir)
    print(f"  Cross product: {cross}")
    print(f"  Cross · a2: {np.dot(cross, a2):.4f}")
    
    # If cross · a2 > 0, rotation is positive around a2
    if np.dot(cross, a2) > 0:
        q2_calc = angle_between
    else:
        q2_calc = -angle_between
    
    print(f"  q2 calculated: {np.rad2deg(q2_calc):.2f}°  (expected: {q2_deg}°)")
    print(f"  Error: {np.rad2deg(q2_calc - q2_rad):.2f}°")

if __name__ == '__main__':
    debug_q2_case(30)
    debug_q2_case(-30)
    debug_q2_case(45)
