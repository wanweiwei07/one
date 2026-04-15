import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    fr_white = ouc.BasicColor.WHITE
    fr_dark = ouc.ExtendedColor.DEEP_GRAY
    # 3 links: hand + 2 fingers
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "hand.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    lf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "finger.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_dark,
    )
    # right finger is the same mesh mirrored via a 180-deg rotation about Z
    rf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "finger.stl"),
        loc_rotmat=oum.rotmat_from_euler(0, 0, np.pi),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_dark,
    )
    # 2 prismatic joints (right mimics left)
    jnt_lf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk,
        child_lnk=lf_lnk,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0, 0, 0.0584], dtype=np.float32),
        lmt_lo=0.0,
        lmt_up=0.04,
    )
    jnt_rf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk,
        child_lnk=rf_lnk,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0, 0, 0.0584], dtype=np.float32),
        mmc=(jnt_lf, 1.0, 0.0),
        lmt_lo=0.0,
        lmt_up=0.04,
    )
    structure.add_lnk(base_lnk)
    structure.add_lnk(lf_lnk)
    structure.add_lnk(rf_lnk)
    structure.add_jnt(jnt_lf)
    structure.add_jnt(jnt_rf)
    structure.ignore_collision(lf_lnk, rf_lnk)
    structure.compile()
    return structure


class FR3Gripper(oreb.EndEffectorBase, oreb.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        # TCP at the midpoint between the fingertips (panda_grasptarget).
        super().__init__(
            loc_tcp_tf=oum.tf_from_rotmat_pos(pos=(0, 0, 0.1034))
        )
        self.jaw_range = np.array([0.0, 0.08], dtype=np.float32)  # min, max
        self.open_dir = ouc.StandardAxis.Y
        self.set_jaw_width(self.jaw_range[1])

    def set_jaw_width(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(
                f"jaw_width {jaw_width} out of range {self.jaw_range}"
            )
        self.fk(qs=[jaw_width * 0.5, jaw_width * 0.5])

    def clone(self):
        new = super().clone()
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir
        new.set_jaw_width(self.qs[0] * 2)
        return new


if __name__ == "__main__":
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=[0.4, 0.2, 0.3], cam_lookat_pos=[0, 0, 0.06])
    gripper = FR3Gripper()
    gripper.attach_to(base.scene)
    ossop.frame().attach_to(base.scene)
    builtins.base = base
    builtins.gripper = gripper
    base.run()
