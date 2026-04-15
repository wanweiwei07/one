import numpy as np
from one.robots.manipulators.kawasaki.rs007l import rs007l

robot = rs007l.RS007L()

print("Checking Link Structure")
print("="*80)

# There are 7 links (base + 6 movable links) and 6 joints
print(f"Number of links: {len(robot.gl_lnk_tfarr)}")
print(f"Number of joints in chain: {len(robot._chain.jnts)}")

print("\n Links and their parent joints:")
print("  Link 0: base (no parent joint)")
print("  Link 1: after J1 (joint 0)")
print("  Link 2: after J2 (joint 1)")
print("  Link 3: after J3 (joint 2)")
print("  Link 4: after J4 (joint 3)")
print("  Link 5: after J5 (joint 4)")
print("  Link 6: after J6 (joint 5)")

print("\n" + "="*80)
print("Correct interpretation:")
print("-"*80)

# o2 should be J2's origin (which is Link1's position + J2's pos offset)
# o3 should be J3's origin

solver = robot._solver

print(f"\no1 (J1 origin from solver): {solver.o1}")
print(f"o2 (J2 origin from solver): {solver.o2}")
print(f"o3 (J3 origin from solver): {solver.o3}")

# Actually, let me check jnt_zero_tfs
print("\n" + "="*80)
print("Joint zero_tfs (origins in world frame at zero config):")
print("-"*80)

for i in range(6):
    origin = solver.jnt_zero_tfs[i][:3, 3]
    print(f"  J{i+1}: {origin}")

# Now let's track what happens when we change q2 and q3
print("\n" + "="*80)
print("FK tracking for q2 and q3:")
print("-"*80)

configs = [
    ("q2=0, q3=0", [0, 0, 0, 0, 0, 0]),
    ("q2=30°", [0, np.radians(30), 0, 0, 0, 0]),
    ("q3=30°", [0, 0, np.radians(30), 0, 0, 0]),
]

for name, config in configs:
    robot.fk(qs=config)
    
    print(f"\n{name}:")
    # J3's position should be at link2's origin + J3's offset (transformed)
    # But since J3 is revolute, we need to look at link3's position
    
    # Link 2 is the parent of J3
    # Link 3 is after J3 rotation
    link2_tf = robot.gl_lnk_tfarr[2]
    link3_tf = robot.gl_lnk_tfarr[3]
    
    # J3's origin is at link2's position (since J3 connects link2 to link3)
    # Actually, J3's origin is embedded in the joint definition
    
    # Let me just print all link positions
    for i in range(4):
        pos = robot.gl_lnk_tfarr[i][:3, 3]
        print(f"  Link {i}: {pos}")
