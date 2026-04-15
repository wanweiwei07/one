import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l

robot = rs007l.RS007L()
solver = robot._solver

print("Checking ow_6 (TCP offset from wrist center)")
print("="*80)

# Zero configuration
robot.fk(qs=[0, 0, 0, 0, 0, 0])

# Get positions
tcp_pos = robot.gl_tcp_tf[:3, 3]
wrist_pos = robot.gl_lnk_tfarr[4][:3, 3]  # Link 5 is after J5 (wrist center area)

print(f"TCP position (zero config): {tcp_pos}")
print(f"Link 5 position (zero config): {wrist_pos}")
print()

# The wrist center we computed
print(f"Computed wrist center (ow): {solver.ow}")
print(f"Computed ow_6: {solver.ow_6}")
print()

# What should ow_6 be?
# ow_6 should be the offset from wrist center to TCP in the wrist frame
# In zero config, the wrist frame orientation is...
tcp_to_wrist_offset = tcp_pos - solver.ow

print(f"Actual TCP - ow offset (world frame): {tcp_to_wrist_offset}")
print()

# Check: does tcp_pos = ow + R06 @ ow_6?
R06_zero = robot.gl_tcp_tf[:3, :3]
tcp_reconstructed = solver.ow + R06_zero @ solver.ow_6

print(f"Reconstructed TCP: solver.ow + R06 @ ow_6 = {tcp_reconstructed}")
print(f"Actual TCP: {tcp_pos}")
print(f"Error: {np.linalg.norm(tcp_reconstructed - tcp_pos)*1000:.2f} mm")
print()

# Test with a different configuration
print("="*80)
print("Testing with q1=45°, q2=30°:")
print("-"*80)

robot.fk(qs=[np.radians(45), np.radians(30), 0, 0, 0, 0])
tcp_pos_test = robot.gl_tcp_tf[:3, 3]
R06_test = robot.gl_tcp_tf[:3, :3]

# Compute pw (wrist center in world frame)
pw_computed = tcp_pos_test - R06_test @ solver.ow_6

print(f"TCP: {tcp_pos_test}")
print(f"Computed wrist center (pw): {pw_computed}")

# What's the actual wrist position from FK?
# J5 is at link index 5
wrist_actual = robot.gl_lnk_tfarr[4][:3, 3]
print(f"Actual J5 position: {wrist_actual}")
print(f"Error: {np.linalg.norm(pw_computed - wrist_actual)*1000:.2f} mm")
