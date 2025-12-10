import os
import numpy as np
import one.utils.constant as const
import one.utils.math as rm
import one.robot_sim.base.robot_state as rstate
import one.robot_sim.base.robot_structure as rstruct
import one.utils.decorator as deco

# ==============================
# define robot structure
# ==============================
structure = rstruct.RobotStructure()
mesh_dir = os.path.join(os.path.dirname(__file__), "meshes")
# 7 links
base_link = rstruct.Link.from_file(os.path.join(mesh_dir, "base_link.stl"),
                                   name="base_link", rgb=const.ExtendedColor.BEIGE)
link1 = rstruct.Link.from_file(os.path.join(mesh_dir, "link1.stl"),
                               name="link1", rgb=const.ExtendedColor.BEIGE)
link2 = rstruct.Link.from_file(os.path.join(mesh_dir, "link2.stl"),
                               name="link2", rgb=const.ExtendedColor.BEIGE)
link3 = rstruct.Link.from_file(os.path.join(mesh_dir, "link3.stl"),
                               name="link3", rgb=const.ExtendedColor.BEIGE)
link4 = rstruct.Link.from_file(os.path.join(mesh_dir, "link4.stl"),
                               name="link4", rgb=const.ExtendedColor.BEIGE)
link5 = rstruct.Link.from_file(os.path.join(mesh_dir, "link5.stl"),
                               name="link5", rgb=const.ExtendedColor.BEIGE)
link6 = rstruct.Link.from_file(os.path.join(mesh_dir, "link6.stl"),
                               name="link6", rgb=const.ExtendedColor.BEIGE)
# 6 joints
joint_bl_l1 = rstruct.Joint("joint_bl_l1", joint_type=const.JointType.REVOLUTE,
                            parent_link=base_link, child_link=link1,
                            axis=-const.StandardAxis.Z,
                            origin_pos=np.array([0.0, 0.0, 0.36], dtype=np.float32),
                            limit_lower=-np.pi, limit_upper=np.pi)
joint_l1_l2 = rstruct.Joint("joint_l1_12", joint_type=const.JointType.REVOLUTE,
                            parent_link=link1, child_link=link2,
                            axis=const.StandardAxis.Z,
                            origin_rotmat=rm.rotmat_from_euler(0, -np.pi / 2, 0),
                            limit_lower=-3 / 4 * np.pi, limit_upper=3 / 4 * np.pi)
joint_l2_l3 = rstruct.Joint("joint_l2_l3", joint_type=const.JointType.REVOLUTE,
                            parent_link=link2, child_link=link3,
                            axis=-const.StandardAxis.Z,
                            origin_pos=np.array([0.455, 0.0, 0.0], dtype=np.float32),
                            limit_lower=-7 / 8 * np.pi, limit_upper=7 / 8 * np.pi)
joint_l3_l4 = rstruct.Joint("joint_l3_l4", joint_type=const.JointType.REVOLUTE,
                            parent_link=link3, child_link=link4,
                            axis=const.StandardAxis.Z,
                            origin_pos=np.array([0.0925, 0.0, 0.0], dtype=np.float32),
                            origin_rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                            limit_lower=-10 / 9 * np.pi, limit_upper=10 / 9 * np.pi)
joint_l4_l5 = rstruct.Joint("joint_l4_l5", joint_type=const.JointType.REVOLUTE,
                            parent_link=link4, child_link=link5,
                            axis=-const.StandardAxis.Z,
                            origin_pos=np.array([0.0, 0.0, 0.3825], dtype=np.float32),
                            origin_rotmat=rm.rotmat_from_euler(0, -np.pi / 2, 0),
                            limit_lower=-25 / 36 * np.pi, limit_upper=25 / 36 * np.pi)
joint_l5_l6 = rstruct.Joint("joint_l5_l6", joint_type=const.JointType.REVOLUTE,
                            parent_link=link5, child_link=link6,
                            axis=const.StandardAxis.Z,
                            origin_pos=np.array([0.078, 0.0, 0.0], dtype=np.float32),
                            origin_rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                            limit_lower=-2 * np.pi, limit_upper=2 * np.pi)
# add links
link2.visuals[0].rotmat = rm.rotmat_from_euler(0, np.pi / 2, 0)
link3.visuals[0].rotmat = rm.rotmat_from_euler(0, np.pi / 2, 0)
link4.visuals[0].pos = np.array([0.0, 0.0, 0.3852], dtype=np.float32)
link5.visuals[0].rotmat = rm.rotmat_from_euler(0, np.pi / 2, 0)
structure.add_link(base_link)
structure.add_link(link1)
structure.add_link(link2)
structure.add_link(link3)
structure.add_link(link4)
structure.add_link(link5)
structure.add_link(link6)
# add joints
structure.add_joint(joint_bl_l1)
structure.add_joint(joint_l1_l2)
structure.add_joint(joint_l2_l3)
structure.add_joint(joint_l3_l4)
structure.add_joint(joint_l4_l5)
structure.add_joint(joint_l5_l6)
# order joints for quick access
structure.finalize()


# =============================
# define robot class
# =============================
class RS007L:

    def __init__(self, base_pos=None, base_rotmat=None):
        self.state = rstate.RobotState(structure)
        self.default_q = np.zeros(len(structure.joint_order), dtype=np.float32)
        self.state.fk(qs=self.default_q)
        self.state.update()

    @property
    @deco.readonly_view
    def structure(self):  # protect structure access
        return RS007L.structure

    def fk(self, qs=None, root_tfmat=None):
        return self.state.fk(qs, root_tfmat)

    def update(self):
        return self.state.update()

    def attach_to(self, scene):
        return self.state.attach_to(scene)

    def remove_from(self, scene):
        return self.state.remove_from(scene)

    def get_link_wd_tfmat(self, link):
        return self.state.get_link_wd_tfmat(link)

    def clone(self):
        new_state = self.state.clone()
        new_robot = RS007L()
        new_robot.state = new_state
        return new_robot
