import os
import one.robot_sim.base.robot_loader as rld

base_dir = os.path.dirname(__file__)
mappings = {"model": "cobotta"}
robot = rld.load_robot_from_xacro("./denso_robot_descriptions/urdf/denso_robot.urdf.xacro",
                                  base_dir=base_dir,
                                  mappings=mappings)
# print("\n=== LINKS ===")
# for link in robot.links:
#     print(f" {link.name}")
#     print("URDF geom paths")
#     for geom in link.visuals + link.collisions:
#         print(f" - {geom.geometry.filename}")
# print("\n=== JOINTS ===")
# for joint in robot.joints:
#     print(f" {joint.name} (type={joint.type})")
#     print("\n=== PARENT → CHILD RELATIONSHIPS ===")
# for joint in robot.joints:
#     print(f" {joint.parent} ──[{joint.name} : {joint.type}]──> {joint.child}")

import one.scene.geometry_loader as geomld

for link in robot.links:
    for geom in link.visuals + link.collisions:
        geom_path = geom.geometry.filename

        if geom_path.startswith("package://denso_robot_descriptions/meshes/"):
            relative_path = geom_path.replace("package://denso_robot_descriptions/meshes/", "")
            full_path = os.path.join(base_dir, "denso_robot_descriptions", "meshes", relative_path)
            if os.path.exists(full_path):
                mesh = geomld.load_dae(full_path)
                print(f"Loaded mesh for {link.name} from {full_path}, number of vertices: {len(mesh.vertices)}")
            else:
                print(f"Mesh file not found: {full_path}")