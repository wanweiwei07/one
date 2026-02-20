import os

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, 'robotiq_base.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SILVER,
    )
    l_knuckle = orbms.Link.from_file(
        os.path.join(mesh_dir, 'left_knuckle.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE,
    )
    r_knuckle = orbms.Link.from_file(
        os.path.join(mesh_dir, 'right_knuckle.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SALMON_PINK,
    )
    l_finger = orbms.Link.from_file(
        os.path.join(mesh_dir, 'left_finger.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.STEEL_BLUE,
    )
    r_finger = orbms.Link.from_file(
        os.path.join(mesh_dir, 'right_finger.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.SALMON_PINK,
    )
    l_inner_knuckle = orbms.Link.from_file(
        os.path.join(mesh_dir, 'left_inner_knuckle.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.DIM_GRAY,
    )
    r_inner_knuckle = orbms.Link.from_file(
        os.path.join(mesh_dir, 'right_inner_knuckle.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.DIM_GRAY,
    )
    l_tip = orbms.Link.from_file(
        os.path.join(mesh_dir, 'left_finger_tip.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.DIM_GRAY,
    )
    r_tip = orbms.Link.from_file(
        os.path.join(mesh_dir, 'right_finger_tip.stl'),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.DIM_GRAY,
    )

    j_l_knuckle = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=l_knuckle,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0.03060114, 0.0, 0.05490452], dtype=np.float32),
        lmt_lo=0.0,
        lmt_up=0.8,
    )
    j_r_knuckle = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=r_knuckle,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([-0.03060114, 0.0, 0.05490452], dtype=np.float32),
        mmc=(j_l_knuckle, -1.0, 0.0),
        lmt_lo=-0.8,
        lmt_up=0.0,
    )
    j_l_finger = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=l_knuckle,
        child_lnk=l_finger,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.03152616, 0.0, -0.00376347], dtype=np.float32),
        lmt_lo=0.0,
        lmt_up=0.0,
    )
    j_r_finger = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=r_knuckle,
        child_lnk=r_finger,
        axis=ouc.StandardAxis.Z,
        pos=np.array([-0.03152616, 0.0, -0.00376347], dtype=np.float32),
        lmt_lo=0.0,
        lmt_up=0.0,
    )
    j_l_inner_knuckle = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=l_inner_knuckle,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0.0127, 0.0, 0.06142], dtype=np.float32),
        mmc=(j_l_knuckle, 1.0, 0.0),
    )
    j_r_inner_knuckle = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=r_inner_knuckle,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([-0.0127, 0.0, 0.06142], dtype=np.float32),
        mmc=(j_l_knuckle, -1.0, 0.0),
    )
    j_l_tip = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=l_finger,
        child_lnk=l_tip,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0.00563134, 0.0, 0.04718515], dtype=np.float32),
        mmc=(j_l_knuckle, -1.0, 0.0),
    )
    j_r_tip = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=r_finger,
        child_lnk=r_tip,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([-0.00563134, 0.0, 0.04718515], dtype=np.float32),
        mmc=(j_l_knuckle, 1.0, 0.0),
    )

    for lnk in [
        base_lnk,
        l_knuckle,
        r_knuckle,
        l_finger,
        r_finger,
        l_inner_knuckle,
        r_inner_knuckle,
        l_tip,
        r_tip,
    ]:
        structure.add_lnk(lnk)
    for jnt in [
        j_l_knuckle,
        j_r_knuckle,
        j_l_finger,
        j_r_finger,
        j_l_inner_knuckle,
        j_r_inner_knuckle,
        j_l_tip,
        j_r_tip,
    ]:
        structure.add_jnt(jnt)

    structure.ignore_collision(l_tip, r_tip)
    structure.ignore_collision(l_finger, r_finger)
    structure.ignore_collision(l_inner_knuckle, r_inner_knuckle)
    structure.compile()
    return structure


class Rtq2F85(oreb.EndEffectorBase, oreb.GripperMixin):

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self):
        super().__init__(loc_tcp_tf=oum.tf_from_rotmat_pos(pos=(0.0, 0.0, 0.15)))
        self.jaw_range = np.array([0.0, 0.085], dtype=np.float32)
        self.open_dir = ouc.StandardAxis.X  # defined in tcp_rotmat
        self.set_jaw_width(self.jaw_range[1])

    def set_jaw_width(self, jaw_width):
        if jaw_width < self.jaw_range[0] or jaw_width > self.jaw_range[1]:
            raise ValueError(f'jaw_width {jaw_width} out of range {self.jaw_range}')
        close_ratio = 1.0 - jaw_width / self.jaw_range[1]
        q_main = 0.8 * close_ratio
        qs = np.zeros(self.structure.compiled.n_jnts, dtype=np.float32)
        qs[0] = q_main
        self.fk(qs=qs)

    def clone(self):
        new = super().clone()
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir
        jaw_width = self.jaw_range[1] * (1.0 - self.qs[0] / 0.8)
        new.set_jaw_width(jaw_width)
        return new


if __name__ == '__main__':
    import builtins
    import math

    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=(0.45, -0.45, 0.28), cam_lookat_pos=(0.10, 0.0, 0.08))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    jaw_widths = [0.0, 0.02, 0.04, 0.06, 0.085]
    template = Rtq2F85()
    grippers = []
    phases = []
    for i, jaw_width in enumerate(jaw_widths):
        gripper = template.clone()
        gripper.set_jaw_width(jaw_width)
        gripper.set_rotmat_pos(pos=(0.10 * i, 0.0, 0.0))
        gripper.attach_to(base.scene)
        grippers.append(gripper)
        phases.append(i * 0.5)
        ossop.frame(pos=gripper.gl_tcp_tf[:3, 3],
                    rotmat=gripper.gl_tcp_tf[:3, :3],
                    color_mat=ouc.CoordColor.MYC).attach_to(base.scene)
        print(f'gripper[{i}] jaw_width={jaw_width:.3f} m')

    t = [0.0]
    min_w, max_w = 0.0, 0.085
    amp = 0.5 * (max_w - min_w)
    mid = 0.5 * (max_w + min_w)

    def update(dt, t, grippers, phases):
        t[0] += dt
        for i, gripper in enumerate(grippers):
            width = mid + amp * math.sin(2.0 * math.pi * 0.5 * t[0] + phases[i])
            gripper.set_jaw_width(width)

    base.schedule_interval(update, interval=1.0 / 60.0, t=t, grippers=grippers, phases=phases)
    base.run()
