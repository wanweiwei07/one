import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb


def get_structure():
    structure = orbms.MechStruct()
    mesh_dir=structure.default_mesh_dir
    # 3 links
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "base_link.stl"),
        local_rotmat=oum.rotmat_from_euler(0, 0, np.pi / 2),
        collision_type=ouc.CollisionType.AABB,
        name="base", rgb=ouc.ExtendedColor.SILVER)
    lft_fgr_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "inward_left_finger_link.stl"),
        local_rotmat=oum.rotmat_from_euler(0, 0, np.pi / 2),
        collision_type=ouc.CollisionType.AABB,
        name="inward_left_finger",
        rgb=ouc.ExtendedColor.STEEL_BLUE)
    rgt_fgr_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "inward_right_finger_link.stl"),
        local_rotmat=oum.rotmat_from_euler(0, 0, np.pi / 2),
        collision_type=ouc.CollisionType.AABB,
        name="inward_right_finger",
        rgb=ouc.ExtendedColor.SALMON_PINK)
    # 1 joint
    jnt_bl_lf = orbms.Joint(
        name="jnt_bl_lf", jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=lft_fgr_lnk,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0, -0.019, 0], dtype=np.float32),
        lmt_low=0.0, lmt_up=0.019)
    jnt_bl_rf = orbms.Joint(
        name="jnt_bl_rf", jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=rgt_fgr_lnk,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0, 0.019, 0], dtype=np.float32),
        mmc=(jnt_bl_lf, 1.0, 0.0),
        lmt_low=0.0, lmt_up=0.019)
    # add links
    structure.add_lnk(base_lnk)
    structure.add_lnk(lft_fgr_lnk)
    structure.add_lnk(rgt_fgr_lnk)
    # add joints
    structure.add_jnt(jnt_bl_lf)
    structure.add_jnt(jnt_bl_rf)
    # order joints for quick access
    structure.compile()
    return structure


class OR2FG7(oreb.EndEffectorBase, oreb.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return get_structure()

    def __init__(self):
        super().__init__(
            tcp_tfmat=oum.tf_from_rotmat_pos(pos=(0, 0, 0.15)))
        self.jaw_range = np.array([0.0, 0.038], dtype=np.float32)  # min, max

    def set_jaw_width(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f"jaw_width {jaw_width} out of range {self.jaw_range}")
        self.fk(qs=[jaw_width * 0.5, jaw_width * 0.5])
