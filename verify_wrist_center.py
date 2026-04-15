import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l
import one.utils.math as oum

robot = rs007l.RS007L()
solver = robot._solver

print("Verifying Wrist Center Calculation")
print("="*80)

# In any configuration, the wrist center should be at the intersection
# of J4, J5, J6 axes

configs = [
    ("Zero", [0, 0, 0, 0, 0, 0]),
    ("q4=45°", [0, 0, 0, np.radians(45), 0, 0]),
    ("q1=45°, q2=30°", [np.radians(45), np.radians(30), 0, 0, 0, 0]),
]

for name, config in configs:
    print(f"\n{name}: {np.degrees(config)}")
    print("-"*80)
    
    robot.fk(qs=config)
    
    # Get joint transformations
    j4_tf = robot.gl_lnk_tfarr[4][:3, :]  # Link 4 (after J4)
    j5_tf = robot.gl_lnk_tfarr[5][:3, :]  # Link 5 (after J5)
    j6_tf = robot.gl_lnk_tfarr[6][:3, :]  # Link 6 (after J6)
    
    # Get joint axes in world frame
    # axis is transformed by the current rotation
    a4_world = j4_tf[:, :3] @ solver.chain.jnts[3].ax
    a5_world = j5_tf[:, :3] @ solver.chain.jnts[4].ax
    a6_world = j6_tf[:, :3] @ solver.chain.jnts[5].ax
    
    # Get joint origins
    o4 = j4_tf[:, 3]
    o5 = j5_tf[:, 3]
    o6 = j6_tf[:, 3]
    
    print(f"J4 origin: {o4}, axis: {a4_world}")
    print(f"J5 origin: {o5}, axis: {a5_world}")
    print(f"J6 origin: {o6}, axis: {a6_world}")
    
    # Compute intersection of J4, J5, J6 axes
    # Same method as in __init__
    A = np.zeros((3, 3), dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    for a, o in [(a4_world, o4), (a5_world, o5), (a6_world, o6)]:
        a_unit = oum.unit_vec(a, return_length=False)
        I_minus_aaT = np.eye(3, dtype=np.float32) - np.outer(a_unit, a_unit)
        A += I_minus_aaT
        b += I_minus_aaT @ o
    
    wrist_center_actual = np.linalg.solve(A, b)
    
    print(f"Actual wrist center (intersection): {wrist_center_actual}")
    
    # Compare with what IK assumes
    tcp_pos = robot.gl_tcp_tf[:3, 3]
    R06 = robot.gl_tcp_tf[:3, :3]
    pw_computed = tcp_pos - R06 @ solver.ow_6
    
    print(f"IK computed wrist center: {pw_computed}")
    print(f"Error: {np.linalg.norm(wrist_center_actual - pw_computed)*1000:.2f} mm")

print("\n" + "="*80)
print("Analysis:")
print("If error is large, then the wrist center moves with q1,q2,q3,")
print("which means ow_6 is not just in the TCP frame but needs adjustment.")
