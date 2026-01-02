import os
import one.robot_sim.base.urdf_loader as rld

base_dir = os.path.dirname(__file__)
mappings = {}
robot = rld.load_robot_from_xacro("./g1_29dof_with_hand.urdf",
                                  base_dir=base_dir,
                                  mappings=mappings)
print("\n=== LINKS ===")
for link in robot.lnks:
    print(f" {link.name}")
print("\n=== JOINTS ===")
for joint in robot.jnts:
    print(f" {joint.name} (type={joint.type})")
    print("\n=== PARENT → CHILD RELATIONSHIPS ===")
for joint in robot.jnts:
    print(f" {joint.parent} ──[{joint.name} : {joint.type}]──> {joint.child}")
