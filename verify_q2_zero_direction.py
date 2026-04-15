import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l

robot = rs007l.RS007L()

print("Analyzing q2=0 reference direction for RS007L")
print("="*80)

# Test 1: What direction does link-2 point when q2=0?
configs_to_test = [
    ("All zeros", [0, 0, 0, 0, 0, 0]),
    ("q1=45°", [np.radians(45), 0, 0, 0, 0, 0]),
    ("q1=90°", [np.radians(90), 0, 0, 0, 0, 0]),
    ("q1=-90°", [np.radians(-90), 0, 0, 0, 0, 0]),
]

print("\nTest: Where is J3 (elbow) when q2=0?")
print("-"*80)

results = []
for name, config in configs_to_test:
    robot.fk(qs=config)
    
    # Get joint positions from the link transformation arrays
    # lnk0=base, lnk1=link1 (after J1), lnk2=link2 (after J2), lnk3=link3 (after J3)
    j1_tf = robot.gl_lnk_tfarr[1]  # Link1's global tf (after J1 rotation)
    j2_tf = robot.gl_lnk_tfarr[2]  # Link2's global tf (after J2 rotation)
    j3_tf = robot.gl_lnk_tfarr[3]  # Link3's global tf (after J3 rotation)
    
    j1_pos = j1_tf[:3, 3]
    j2_pos = j2_tf[:3, 3]
    j3_pos = j3_tf[:3, 3]
    
    # Vector from J2's origin to J3's origin
    # (This represents where link-2 points)
    v_j2_to_j3 = j3_pos - j1_pos  # Use j1_pos as reference since J1 and J2 origins coincide
    
    q1_deg = np.degrees(config[0])
    
    print(f"\n{name}:")
    print(f"  q1 = {q1_deg:.1f}°")
    print(f"  J1 position: {j1_pos}")
    print(f"  J2 position: {j2_pos}")
    print(f"  J3 position: {j3_pos}")
    print(f"  J1→J3 vector: {v_j2_to_j3}")
    print(f"  |J1→J3|: {np.linalg.norm(v_j2_to_j3):.4f} m")
    
    # Azimuthal angle
    theta = np.arctan2(v_j2_to_j3[1], v_j2_to_j3[0])
    theta_deg = np.degrees(theta)
    
    print(f"  Azimuthal angle θ: {theta_deg:.2f}°")
    
    results.append((q1_deg, theta_deg))

print("\n" + "="*80)
print("Analysis: θ should equal (q1 + fixed_offset)")
print("-"*80)

# Calculate offsets
offsets = [(q1, theta - q1) for q1, theta in results]
print("\nq1 (deg)  |  θ (deg)  |  offset = θ - q1")
print("-"*50)
for (q1, theta), (_, offset) in zip(results, offsets):
    print(f"{q1:8.1f}  |  {theta:8.2f}  |  {offset:8.2f}")

# Check if offset is consistent
offset_values = [offset for _, offset in offsets]
offset_std = np.std(offset_values)

print(f"\nOffset std deviation: {offset_std:.4f}°")

if offset_std < 1.0:
    print("✓ Offset is CONSISTENT - The 2R mechanism is independent of q1")
    print(f"  Fixed offset: {np.mean(offset_values):.2f}°")
else:
    print("✗ Offset VARIES - There's a bug in how the 2R mechanism relates to q1")

# Test 2: Check J2's local frame orientation
print("\n" + "="*80)
print("Test: J2's local Z-axis (rotation axis) at q2=0")
print("-"*80)

for name, config in configs_to_test:
    robot.fk(qs=config)
    
    j2_tf = robot.gl_lnk_tfarr[2]
    j2_rotmat = j2_tf[:3, :3]
    
    # J2's axis should be the local Z-axis after applying zero_tf
    local_z = j2_rotmat @ np.array([0, 0, 1], dtype=np.float32)
    
    q1_deg = np.degrees(config[0])
    print(f"\n{name}: q1={q1_deg:.1f}°")
    print(f"  J2 local Z-axis: {local_z}")
    print(f"  |Z|: {np.linalg.norm(local_z):.4f}")
