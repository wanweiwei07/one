import os
import numpy as np
import one.utils.math as rm
import one.utils.constant as const
import one.robot_sim.base.mech_structure as rstruct
import one.robot_sim.end_effectors.ee_base as eebase


def get_robot_structure():
    structure = rstruct.MechaBase()
    mesh_dir = os.path.join(os.path.dirname(__file__), "meshes")
    # 3 links
    base_link = rstruct.Link.from_file(os.path.join(mesh_dir, "base_link.stl"),
                                       local_rotmat=rm.rotmat_from_euler(0, 0, np.pi / 2),
                                       collision_type=const.CollisionType.AABB,
                                       name="base",
                                       rgb=const.ExtendedColor.ALUMINUM_ANODIZED)
    left_finger_link = rstruct.Link.from_file(os.path.join(mesh_dir, "inward_left_finger_link.stl"),
                                              local_rotmat=rm.rotmat_from_euler(0, 0, np.pi / 2),
                                              collision_type=const.CollisionType.AABB,
                                              name="inward_left_finger",
                                              rgb=const.ExtendedColor.DIM_GRAY)
    right_finger_link = rstruct.Link.from_file(os.path.join(mesh_dir, "inward_right_finger_link.stl"),
                                               local_rotmat=rm.rotmat_from_euler(0, 0, np.pi / 2),
                                               collision_type=const.CollisionType.AABB,
                                               name="inward_right_finger",
                                               rgb=const.ExtendedColor.DIM_GRAY)
    # 1 joint
    joint_bl_lf = rstruct.Joint("joint_bl_lf", jnt_type=const.JntType.PRISMATIC,
                                parent_lnk=base_link, child_lnk=left_finger_link,
                                axis=const.StandardAxis.Y,
                                pos=np.array([0, -0.019, 0], dtype=np.float32),
                                lmt_low=0.0, lmt_up=0.019)
    joint_bl_rf = rstruct.Joint("joint_bl_rf", jnt_type=const.JntType.PRISMATIC,
                                parent_lnk=base_link, child_lnk=right_finger_link,
                                axis=-const.StandardAxis.Y,
                                pos=np.array([0, 0.019, 0], dtype=np.float32),
                                mmc=(joint_bl_lf, -1.0, 0.0),
                                lmt_low=0.0, lmt_up=0.019)
    # add links
    structure.add_lnk(base_link)
    structure.add_lnk(left_finger_link)
    structure.add_lnk(right_finger_link)
    # add joints
    structure.add_jnt(joint_bl_lf)
    structure.add_jnt(joint_bl_rf)
    # order joints for quick access
    structure.compile()
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
