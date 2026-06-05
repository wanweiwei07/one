import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.base.mech_base as orbmb
import one.robots.end_effectors.ee_base as oreb


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    # local pos
    hand_pos = np.array([0.0, 0.0, -0.5584], dtype=np.float32)
    left_pos = np.array([0.0, -0.05, -0.572901], dtype=np.float32)
    right_pos = np.array([0.0, 0.05, -0.572901], dtype=np.float32)
    # links
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "hand.stl"),
        loc_rotmat=None, loc_pos=hand_pos,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER)
    lf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "finger.stl"),
        loc_rotmat=None, loc_pos=left_pos,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE)
    rf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "finger_mirrorY.stl"),
        loc_rotmat=None, loc_pos=right_pos,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SALMON_PINK)
    # joints
    jnt_rf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=rf_lnk,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0.0, -0.006, 0.015],
                     dtype=np.float32),
        lmt_lo=0.0, lmt_up=0.044)
    jnt_lf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=lf_lnk,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0.0, 0.006, 0.015],
                     dtype=np.float32),
        mmc=(jnt_rf, 1.0, 0.0),
        lmt_lo=0.0, lmt_up=0.044)
    # add lnks
    structure.add_lnk(base_lnk)
    structure.add_lnk(lf_lnk)
    structure.add_lnk(rf_lnk)
    # add jnts
    structure.add_jnt(jnt_rf)
    structure.add_jnt(jnt_lf)
    # ignore collisions between fingers
    structure.ignore_collision(lf_lnk, rf_lnk)
    # order joints for quick access
    structure.compile()
    return structure


class OAGripper(orbmb.MechBase, oreb.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        super().__init__()
        self.add_tcp('grasp_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(pos=(0.0, 0.0, 0.1801)))
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self.jaw_range = np.array(
            [0.0, 0.088], dtype=np.float32)
        self.set_jaw_width(0.0)

    def set_jaw_width(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f"jaw_width {jaw_width} out of range {self.jaw_range}")
        self.fk(qs=[jaw_width * 0.5, jaw_width * 0.5])

    def clone(self):
        new = super().clone()
        new.contact_pattern = self.contact_pattern.copy()
        new.jaw_range = self.jaw_range.copy()
        new.set_jaw_width(self.qs[0] * 2.0)
        return new


if __name__ == '__main__':
    import one.viewer.world as wd
    import one.scene.scene_object_primitive as ossop

    base = wd.World(cam_pos=[.5, .5, .3], cam_lookat_pos=[0, 0, 0])
    oframe = ossop.frame()
    oframe.attach_to(base.scene)
    gripper = OAGripper()
    gripper.attach_to(base.scene)
    _tf = gripper.tcp('grasp_center').tf
    tcp_frame = ossop.frame(rotmat=_tf[:3, :3],
                            pos=_tf[:3, 3],
                            color_mat=ouc.CoordColor.MYC)
    tcp_frame.attach_to(base.scene)
    base.run()