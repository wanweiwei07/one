import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

# Target
q1_target = np.radians(45)
q2_target = np.radians(30)
q3_target = np.radians(0)

robot.fk(qs=[q1_target, q2_target, q3_target, 0, 0, 0])
tcp_pos = robot.gl_tcp_tf[:3, 3]
tcp_rot = robot.gl_tcp_tf[:3, :3]
pw_target = tcp_pos - tcp_rot @ solver.ow_6

print(f"Target: q1={np.degrees(q1_target):.1f}°, q2={np.degrees(q2_target):.1f}°, q3={np.degrees(q3_target):.1f}°")
print(f"Wrist center: {pw_target}")
print()

# Manually trace through new logic
a1 = oum.unit_vec(solver.a1, return_length=False)
v = pw_target - solver.o2
d_total = np.linalg.norm(v)

print(f"v (J2 to wrist): {v}")
print(f"d_total: {d_total:.4f} m")
print(f"l2 + l3: {solver.l2 + solver.l3:.4f} m")
print()

# Project for q1
v_projected = v - np.dot(v, a1) * a1
r_xy = np.linalg.norm(v_projected)

print(f"v_projected: {v_projected}")
print(f"r_xy: {r_xy:.4f} m")
print()

# Azimuth
x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
y_ref = np.cross(a1, x_ref)
azimuth = np.arctan2(np.dot(v_projected, y_ref), np.dot(v_projected, x_ref))

print(f"azimuth: {np.degrees(azimuth):.2f}°")

q1_sols = [
    float(oum.wrap_to_pi(azimuth - np.pi / 2)),
    float(oum.wrap_to_pi(azimuth + np.pi / 2))
]

print(f"q1 solutions: {[np.degrees(q) for q in q1_sols]}")
print(f"Target q1: {np.degrees(q1_target):.2f}°")
print()

# Check which one is correct
for i, q1 in enumerate(q1_sols):
    print(f"\nTrying q1 = {np.degrees(q1):.2f}°:")
    
    R_j1 = oum.rotmat_from_axangle(a1, q1)
    v_j1 = R_j1 @ v
    
    print(f"  v_j1: {v_j1}")
    
    # For q1=45°, q2=30°, q3=0°, what should v_j1 be?
    # We can compute this from FK
    if abs(q1 - q1_target) < 0.01:
        # This is the correct q1
        robot.fk(qs=[q1_target, 0, 0, 0, 0, 0])
        wrist_q1_only = robot.gl_tcp_tf[:3, 3] - robot.gl_tcp_tf[:3, :3] @ solver.ow_6
        
        R_j1_correct = oum.rotmat_from_axangle(a1, q1_target)
        v_j1_expected_base = R_j1_correct @ (wrist_q1_only - solver.o2)
        
        print(f"  Expected v_j1 if q2=q3=0: {v_j1_expected_base}")
        
        # Now if q2=30°, q3=0°, where should the wrist be in J1's frame?
        robot.fk(qs=[q1_target, q2_target, q3_target, 0, 0, 0])
        wrist_actual = robot.gl_tcp_tf[:3, 3] - robot.gl_tcp_tf[:3, :3] @ solver.ow_6
        v_j1_expected = R_j1_correct @ (wrist_actual - solver.o2)
        
        print(f"  Expected v_j1 for target config: {v_j1_expected}")

print("\n" + "="*80)
print("Analysis: Are both q1 solutions being generated?")
