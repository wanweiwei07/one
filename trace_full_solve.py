import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

# Manually copy the _solve_first3 logic with print statements

q1_target = np.radians(45)
q2_target = np.radians(30)
q3_target = np.radians(0)

robot.fk(qs=[q1_target, q2_target, q3_target, 0, 0, 0])
tcp_pos = robot.gl_tcp_tf[:3, 3]
tcp_rot = robot.gl_tcp_tf[:3, :3]
pw = tcp_pos - tcp_rot @ solver.ow_6

a1 = oum.unit_vec(solver.a1, return_length=False)
l2, l3 = solver.l2, solver.l3

v = pw - solver.o2
d_total = np.linalg.norm(v)

v_projected = v - np.dot(v, a1) * a1
r_xy = np.linalg.norm(v_projected)

x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
y_ref = np.cross(a1, x_ref)
azimuth = np.arctan2(np.dot(v_projected, y_ref), np.dot(v_projected, x_ref))

q1_solutions = [
    float(oum.wrap_to_pi(azimuth - np.pi / 2)),
    float(oum.wrap_to_pi(azimuth + np.pi / 2))
]

print(f"q1_solutions: {[np.degrees(q) for q in q1_solutions]}")
print()

sols = []
for idx, q1 in enumerate(q1_solutions):
    print(f"Processing q1 = {np.degrees(q1):.2f}°:")
    
    R_j1 = oum.rotmat_from_axangle(a1, q1)
    v_j1 = R_j1 @ v
    a2_j1 = R_j1 @ solver.a2
    a2_unit = oum.unit_vec(a2_j1, return_length=False)
    
    print(f"  v_j1: {v_j1}")
    print(f"  a2_unit: {a2_unit}")
    
    # Law of cosines
    c3 = oum.clamp((d_total**2 - l2**2 - l3**2) / (2 * l2 * l3), lo=-1.0, hi=1.0)
    s3_abs = np.sqrt(max(0.0, 1.0 - c3**2))
    
    print(f"  c3: {c3:.4f}, s3_abs: {s3_abs:.4f}")
    
    for s3_sign in (+1, -1):
        s3 = s3_sign * s3_abs
        if abs(s3) < 1e-9 and sols:
            print(f"    Skipping s3={s3:.4f} (duplicate)")
            continue
        
        q3 = float(np.arctan2(s3, c3))
        print(f"    s3={s3:.4f}, q3={np.degrees(q3):.2f}°")
        
        # Solve for q2
        z_axis = np.array([0, 0, 1], dtype=np.float32)
        z_in_j1 = R_j1 @ z_axis
        dir_zero = z_in_j1 - np.dot(z_in_j1, a2_unit) * a2_unit
        
        if np.linalg.norm(dir_zero) < 1e-9:
            print(f"      ERROR: dir_zero is zero!")
            continue
        
        dir_zero = oum.unit_vec(dir_zero, return_length=False)
        dir_perp = oum.unit_vec(np.cross(a2_unit, dir_zero), return_length=False)
        
        print(f"      dir_zero: {dir_zero}")
        print(f"      dir_perp: {dir_perp}")
        
        coord_zero = np.dot(v_j1, dir_zero)
        coord_perp = np.dot(v_j1, dir_perp)
        
        phi_wrist = np.arctan2(coord_perp, coord_zero)
        psi = np.arctan2(l3 * s3, l2 + l3 * c3)
        
        q2 = float(phi_wrist - psi)
        
        print(f"      coord_zero: {coord_zero:.4f}, coord_perp: {coord_perp:.4f}")
        print(f"      phi_wrist: {np.degrees(phi_wrist):.2f}°, psi: {np.degrees(psi):.2f}°")
        print(f"      q2: {np.degrees(q2):.2f}°")
        
        sols.append((q1, oum.wrap_to_pi(q2), oum.wrap_to_pi(q3)))

print(f"\n Total solutions: {len(sols)}")
for i, (q1, q2, q3) in enumerate(sols):
    print(f"  Sol {i+1}: q1={np.degrees(q1):.2f}°, q2={np.degrees(q2):.2f}°, q3={np.degrees(q3):.2f}°")
