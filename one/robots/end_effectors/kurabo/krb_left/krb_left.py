import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    # 3 links
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "lf_base.stl"),
        scale=0.001,
        loc_rotmat=None,
        loc_pos=None,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER)
    lf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "lf_left.stl"),
        scale=0.001,
        loc_rotmat=None,
        loc_pos=None,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE)
    rf_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "lh_right.stl"),
        scale=0.001,
        loc_rotmat=None,
        loc_pos=None,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SALMON_PINK)
    # 1 joint
    jnt_lf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=lf_lnk,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0, -0.0119, 0], dtype=np.float32),
        lmt_lo=0.0, lmt_up=0.0119)
    jnt_rf = orbms.Joint(
        jnt_type=ouc.JntType.PRISMATIC,
        parent_lnk=base_lnk, child_lnk=rf_lnk,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0, 0.0119, 0], dtype=np.float32),
        mmc=(jnt_lf, 1.0, 0.0),
        lmt_lo=0.0, lmt_up=0.0119)
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


class KRBLeft(oreb.EndEffectorBase, oreb.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        super().__init__(
            loc_tcp_tf=oum.tf_from_rotmat_pos(pos=(0, 0, 0.15)))
        self.jaw_range = np.array([0.0, 0.0238], dtype=np.float32)  # min, max
        self.open_dir = ouc.StandardAxis.Y
        self.set_jaw_width(0.0)

    def set_jaw_width(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f"jaw_width {jaw_width} out of range {self.jaw_range}")
        self.fk(qs=[jaw_width * 0.5, jaw_width * 0.5])

    def clone(self):
        new = super().clone()
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir
        new.set_jaw_width(self.qs[0] * 2)
        return new


if __name__ == '__main__':
    import builtins
    import math

    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=(0.3, -0.3, 0.25), cam_lookat_pos=(0.05, 0.0, 0.05))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    jaw_widths = [0.0, 0.006, 0.012, 0.018, 0.0238]
    template = KRBLeft()
    grippers = []
    phases = []
    for i, jaw_width in enumerate(jaw_widths):
        gripper = template.clone()
        gripper.set_jaw_width(jaw_width)
        gripper.set_rotmat_pos(pos=(0.06 * i, 0.0, 0.0))
        gripper.attach_to(base.scene)
        grippers.append(gripper)
        phases.append(i * 0.5)
        ossop.frame(pos=gripper.gl_tcp_tf[:3, 3],
                    rotmat=gripper.gl_tcp_tf[:3, :3],
                    color_mat=ouc.CoordColor.MYC).attach_to(base.scene)
        print(f'gripper[{i}] jaw_width={jaw_width:.4f} m')

    t = [0.0]
    min_w, max_w = 0.0, 0.0238
    amp = 0.5 * (max_w - min_w)
    mid = 0.5 * (max_w + min_w)

    def update(dt, t, grippers, phases):
        t[0] += dt
        for i, gripper in enumerate(grippers):
            width = mid + amp * math.sin(2.0 * math.pi * 0.5 * t[0] + phases[i])
            gripper.set_jaw_width(width)

    base.schedule_interval(update, interval=1.0 / 60.0, t=t, grippers=grippers, phases=phases)
    base.run()
