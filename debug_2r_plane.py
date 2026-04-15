"""
调试2R平面机构的q2, q3计算
"""
import numpy as np
from one.robots.manipulators.kawasaki.rs007l.rs007l import RS007L
import one.utils.math as oum

robot = RS007L()
solver = robot.get_solver(robot._chain)

# 测试用例: q=[45°, 30°, -45°, 0, 0, 0]
q_target = np.array([np.deg2rad(45), np.deg2rad(30), np.deg2rad(-45), 0, 0, 0], dtype=np.float32)
robot.fk(q_target)

# 获取腕部位置
tcp_pos = robot.gl_tcp_tf[:3, 3].copy()
tcp_rot = robot.gl_tcp_tf[:3, :3].copy()
tgt_tcp_tf = oum.tf_from_rotmat_pos(tcp_rot, tcp_pos)
tgt_flange_tf = tgt_tcp_tf @ np.linalg.inv(robot._loc_tcp_tf)
root_tf = oum.tf_from_rotmat_pos(robot.rotmat, robot.pos)
tgt_tf_in_root = np.linalg.inv(root_tf) @ tgt_flange_tf
pw = tgt_tf_in_root[:3, 3] - tgt_tf_in_root[:3, :3] @ solver.ow_6

print('目标: q=[45°, 30°, -45°, 0, 0, 0]')
print(f'腕部位置: {pw}')
print('='*70)

# 手动执行_solve_first3的逻辑
a1 = oum.unit_vec(solver.a1, return_length=False)
l2, l3 = solver.l2, solver.l3

v = pw - solver.o2
d_total = np.linalg.norm(v)
print(f'\nv (J2→腕部): {v}')
print(f'd_total: {d_total:.4f}m')
print(f'l2+l3: {l2+l3:.4f}m')

# q1 = 45
q1 = np.deg2rad(45)
R_j1 = oum.rotmat_from_axangle(a1, q1)
v_j1 = R_j1 @ v
a2_j1 = R_j1 @ solver.a2
a2_unit = oum.unit_vec(a2_j1, return_length=False)

print(f'\nJ1框架 (q1={np.rad2deg(q1):.0f}°):')
print(f'  v_j1: {v_j1}')
print(f'  a2_j1: {a2_j1}')

# 分解v_j1
v_parallel = np.dot(v_j1, a2_unit)
v_perp_vec = v_j1 - v_parallel * a2_unit
v_perp = np.linalg.norm(v_perp_vec)

print(f'\n分解到a2坐标:')
print(f'  v_parallel (沿a2): {v_parallel:.4f}')
print(f'  v_perp (垂直a2): {v_perp:.4f}')
print(f'  v_perp_vec: {v_perp_vec}')

# 在垂直平面求解2R
d_2d = v_perp
print(f'\n2R平面距离: {d_2d:.4f}m')

# q3
c3 = (d_2d**2 - l2**2 - l3**2) / (2 * l2 * l3)
print(f'\nc3 = (d_2d²-l2²-l3²)/(2*l2*l3) = ({d_2d**2:.6f}-{l2**2:.6f}-{l3**2:.6f})/(2*{l2}*{l3})')
print(f'   = {c3:.4f}')
s3 = np.sqrt(max(0, 1 - c3**2))
q3_pos = np.arctan2(s3, c3)
q3_neg = np.arctan2(-s3, c3)
print(f's3 = {s3:.4f}')
print(f'q3_pos = {np.rad2deg(q3_pos):.2f}°')
print(f'q3_neg = {np.rad2deg(q3_neg):.2f}°')
print(f'期望: -45°')

# 对两个q3解都计算q2
for q3 in [q3_pos, q3_neg]:
    print(f'\n{"="*70}')
    print(f'测试 q3 = {np.rad2deg(q3):.2f}°')
    print(f'{"="*70}')
    
    # q2计算
    alpha = np.arctan2(l3 * np.sin(q3), l2 + l3 * np.cos(q3))
    print(f'\nalpha = arctan2(l3*sin(q3), l2+l3*cos(q3))')
    print(f'      = arctan2({l3*np.sin(q3):.4f}, {l2 + l3*np.cos(q3):.4f})')
    print(f'      = {np.rad2deg(alpha):.2f}°')
    
    # 零位方向
    z_world = np.array([0, 0, 1], dtype=np.float32)
    arm_zero_in_j1 = R_j1 @ z_world
    arm_zero_perp = arm_zero_in_j1 - np.dot(arm_zero_in_j1, a2_unit) * a2_unit
    arm_zero_perp_unit = oum.unit_vec(arm_zero_perp, return_length=False)
    
    print(f'\n零位方向 (在J1框架中):')
    print(f'  arm_zero_perp_unit: {arm_zero_perp_unit}')
    
    # 目标方向
    v_perp_unit = oum.unit_vec(v_perp_vec, return_length=False)
    print(f'\n目标方向:')
    print(f'  v_perp_unit: {v_perp_unit}')
    
    # beta角
    cos_beta = np.dot(arm_zero_perp_unit, v_perp_unit)
    sin_beta_cross = np.cross(arm_zero_perp_unit, v_perp_unit)
    cross_dot_a2 = np.dot(sin_beta_cross, a2_unit)
    sin_beta = np.linalg.norm(sin_beta_cross) * np.sign(cross_dot_a2) if abs(cross_dot_a2) > 1e-9 else 0.0
    beta = np.arctan2(sin_beta, cos_beta)
    
    print(f'\nbeta角计算:')
    print(f'  cos_beta = {cos_beta:.4f}')
    print(f'  sin_beta = {sin_beta:.4f}')
    print(f'  beta = {np.rad2deg(beta):.2f}°')
    
    # q2
    q2 = beta - alpha
    print(f'\nq2 = beta - alpha = {np.rad2deg(beta):.2f}° - {np.rad2deg(alpha):.2f}° = {np.rad2deg(q2):.2f}°')
    print(f'期望: 30°')
    
    # 验证FK
    print(f'\n验证FK:')
    robot.fk(np.array([q1, q2, q3, 0, 0, 0], dtype=np.float32))
    wrist_check = robot.gl_lnk_tfarr[5][:3, 3]
    wrist_actual = robot.gl_lnk_tfarr[5][:3, 3]
    robot.fk(q_target)
    wrist_target = robot.gl_lnk_tfarr[5][:3, 3]
    error = np.linalg.norm(wrist_check - wrist_target)
    print(f'  腕部目标: {wrist_target}')
    print(f'  腕部实际: {wrist_check}')
    print(f'  误差: {error*1000:.2f}mm')

