import os
import numpy as np
import one.utils.math as rm
import one.utils.constant as const
import one.robot_sim.base.robot_structure as rstruct
import one.robot_sim.end_effectors.ee_base as eebase


def get_robot_structure():
    structure = rstruct.RobotStructure()
    mesh_dir = os.path.join(os.path.dirname(__file__), "meshes")
    # 3 links
    base_link = rstruct.Link.from_file(os.path.join(mesh_dir, "base_link.stl"),
                                       local_rotmat=rm.rotmat_from_euler(0, 0, np.pi / 2),
                                       rgb=const.ExtendedColor.ALUMINUM_ANODIZED,
                                       collision_type=const.CollisionType.AABB)
    left_finger_link = rstruct.Link.from_file(os.path.join(mesh_dir, "inward_left_finger_link.stl"),
                                              local_rotmat=rm.rotmat_from_euler(0, 0, np.pi / 2),
                                              rgb=const.ExtendedColor.DIM_GRAY,
                                              collision_type=const.CollisionType.AABB)
    right_finger_link = rstruct.Link.from_file(os.path.join(mesh_dir, "inward_right_finger_link.stl"),
                                              local_rotmat=rm.rotmat_from_euler(0, 0, np.pi / 2),
                                               rgb=const.ExtendedColor.DIM_GRAY,
                                               collision_type=const.CollisionType.AABB)
    # 1 joint
    joint_bl_lf = rstruct.Joint("joint_bl_lf", joint_type=const.JointType.PRISMATIC,
                                parent_link=base_link, child_link=left_finger_link,
                                axis=const.StandardAxis.Y,
                                pos=np.array([0, -0.019, 0], dtype=np.float32),
                                limit_lower=0.0, limit_upper=0.019)
    joint_bl_rf = rstruct.Joint("joint_bl_rf", joint_type=const.JointType.PRISMATIC,
                                parent_link=base_link, child_link=right_finger_link,
                                axis=-const.StandardAxis.Y,
                                pos=np.array([0, 0.019, 0], dtype=np.float32),
                                mimic=(joint_bl_lf, -1.0, 0.0),
                                limit_lower=0.0, limit_upper=0.019)
    # add links
    structure.add_link(base_link)
    structure.add_link(left_finger_link)
    structure.add_link(right_finger_link)
    # add joints
    structure.add_joint(joint_bl_lf)
    structure.add_joint(joint_bl_rf)
    # order joints for quick access
    structure.finalize()
    return structure


class OR2FG7(eebase.EndEffectorBase, eebase.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return get_robot_structure()

    def __init__(self, base_pos=None, base_rotmat=None):
        super().__init__()
        self.jaw_range = np.array([0.0, 0.038], dtype=np.float32)  # min, max

    def set_jaw_width(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f"jaw_width {jaw_width} out of range {self.jaw_range}")
        self.fk(qs=[jaw_width * 0.5, jaw_width * 0.5])
