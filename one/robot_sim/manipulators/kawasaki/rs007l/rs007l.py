import os
import one.robot_sim.base.robot_structure as rstruct
import one.robot_sim.base.robot_base as rbase
import one.utils.constant as const
import one.utils.math as rm


def get_robot_structure():
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
                                origin_pos=rm.np.array([0.0, 0.0, 0.36], dtype=rm.np.float32),
                                limit_lower=-rm.np.pi, limit_upper=rm.np.pi)
    joint_l1_l2 = rstruct.Joint("joint_l1_12", joint_type=const.JointType.REVOLUTE,
                                parent_link=link1, child_link=link2,
                                axis=const.StandardAxis.Z,
                                origin_rotmat=rm.rotmat_from_euler(0, -rm.np.pi / 2, 0),
                                limit_lower=-3 / 4 * rm.np.pi, limit_upper=3 / 4 * rm.np.pi)
    joint_l2_l3 = rstruct.Joint("joint_l2_l3", joint_type=const.JointType.REVOLUTE,
                                parent_link=link2, child_link=link3,
                                axis=-const.StandardAxis.Z,
                                origin_pos=rm.np.array([0.455, 0.0, 0.0], dtype=rm.np.float32),
                                limit_lower=-7 / 8 * rm.np.pi, limit_upper=7 / 8 * rm.np.pi)
    joint_l3_l4 = rstruct.Joint("joint_l3_l4", joint_type=const.JointType.REVOLUTE,
                                parent_link=link3, child_link=link4,
                                axis=const.StandardAxis.Z,
                                origin_pos=rm.np.array([0.0925, 0.0, 0.0], dtype=rm.np.float32),
                                origin_rotmat=rm.rotmat_from_euler(0, rm.np.pi / 2, 0),
                                limit_lower=-10 / 9 * rm.np.pi, limit_upper=10 / 9 * rm.np.pi)
    joint_l4_l5 = rstruct.Joint("joint_l4_l5", joint_type=const.JointType.REVOLUTE,
                                parent_link=link4, child_link=link5,
                                axis=-const.StandardAxis.Z,
                                origin_pos=rm.np.array([0.0, 0.0, 0.3825], dtype=rm.np.float32),
                                origin_rotmat=rm.rotmat_from_euler(0, -rm.np.pi / 2, 0),
                                limit_lower=-25 / 36 * rm.np.pi, limit_upper=25 / 36 * rm.np.pi)
    joint_l5_l6 = rstruct.Joint("joint_l5_l6", joint_type=const.JointType.REVOLUTE,
                                parent_link=link5, child_link=link6,
                                axis=const.StandardAxis.Z,
                                origin_pos=rm.np.array([0.078, 0.0, 0.0], dtype=rm.np.float32),
                                origin_rotmat=rm.rotmat_from_euler(0, rm.np.pi / 2, 0),
                                limit_lower=-2 * rm.np.pi, limit_upper=2 * rm.np.pi)
    # add links
    link2.visuals[0].rotmat = rm.rotmat_from_euler(0, rm.np.pi / 2, 0)
    link3.visuals[0].rotmat = rm.rotmat_from_euler(0, rm.np.pi / 2, 0)
    link4.visuals[0].pos = rm.np.array([0.0, 0.0, 0.3852], dtype=rm.np.float32)
    link5.visuals[0].rotmat = rm.rotmat_from_euler(0, rm.np.pi / 2, 0)
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
    return structure


class RS007L(rbase.RobotBase):

    @classmethod
    def _build_structure(cls):
        return get_robot_structure()

    def __init__(self, base_pos=None, base_rotmat=None):
        super().__init__()