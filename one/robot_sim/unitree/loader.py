import os
import one.robot_sim.robot_loader as rld

base_dir = os.path.dirname(__file__)
mappings = {"model": "cobotta"}
robot = rld.load_robot_from_xacro("./denso_robot_descriptions/urdf/denso_robot.urdf.xacro",
                                  mappings=mappings)
print("\n=== LINKS ===")
for link in robot.links:
    print(f" {link.name}")
print("\n=== JOINTS ===")
for joint in robot.joints:
    print(f" {joint.name} (type={joint.type})")
    print("\n=== PARENT → CHILD RELATIONSHIPS ===")
for joint in robot.joints:
    print(f" {joint.parent} ──[{joint.name} : {joint.type}]──> {joint.child}")