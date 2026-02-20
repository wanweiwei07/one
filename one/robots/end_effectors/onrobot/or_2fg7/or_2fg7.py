import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir=structure.default_mesh_dir
    # 3 links
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "base_link.stl"),
        loc_rotmat=oum.rotmat_from_euler(0, 0, np.pi / 2),
        loc_pos=None,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER)
    lf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "inward_left_finger_link.stl"),
        loc_rotmat=oum.rotmat_from_euler(0, 0, np.pi / 2),
        loc_pos=None,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE)
    rf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "inward_right_finger_link.stl"),
        loc_rotmat=oum.rotmat_from_euler(0, 0, np.pi / 2),
        loc_pos=None,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SALMON_PINK)
    # 1 joint
    jnt_lf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=lf_lnk,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0, -0.019, 0], dtype=np.float32),
        lmt_lo=0.0, lmt_up=0.019)
    jnt_rf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=rf_lnk,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0, 0.019, 0], dtype=np.float32),
        mmc=(jnt_lf, 1.0, 0.0),
        lmt_lo=0.0, lmt_up=0.019)
    # add lnks
    structure.add_lnk(base_lnk)
    structure.add_lnk(lf_lnk)
    structure.add_lnk(rf_lnk)
    # add jnts
    structure.add_jnt(jnt_lf)
    structure.add_jnt(jnt_rf)
    # ignore collision between fingers
    structure.ignore_collision(lf_lnk, rf_lnk)
    # order joints for quick access
    structure.compile()
    return structure


class OR2FG7(oreb.EndEffectorBase, oreb.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        super().__init__(
            loc_tcp_tf=oum.tf_from_rotmat_pos(pos=(0, 0, 0.15)))
        self.jaw_range = np.array([0.005, 0.038], dtype=np.float32)  # min, max
        self.open_dir = ouc.StandardAxis.Y
        self.set_jaw_width(0.005)

    def set_jaw_width(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f"jaw_width {jaw_width} out of range {self.jaw_range}")
        self.fk(qs=[jaw_width * 0.5, jaw_width * 0.5])

    def clone(self):
        new = super().clone()
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir
        new.set_jaw_width(self.qs[0]*2)
        return new
