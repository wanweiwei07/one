import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.manipulators.manipulator_base as ormmb


def prepare_mechstruct():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    # 7 links
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "base_link.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    base_lnk.set_inertia(mass=11.0)
    lnk1 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link1.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    lnk1.set_inertia(mass=8.118)
    lnk2 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link2.stl"),
        collision_type=ouc.CollisionType.MESH,
        loc_rotmat=oum.rotmat_from_euler(0, np.pi / 2, 0),
        rgb=ouc.ExtendedColor.BEIGE)
    lnk2.set_inertia(mass=6.826)
    lnk3 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link3.stl"),
        collision_type=ouc.CollisionType.MESH,
        loc_rotmat=oum.rotmat_from_euler(0, np.pi / 2, 0),
        rgb=ouc.ExtendedColor.BEIGE)
    lnk3.set_inertia(mass=5.236)
    lnk4 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link4.stl"),
        collision_type=ouc.CollisionType.MESH,
        loc_pos=np.array([0.0, 0.0, 0.3852], dtype=np.float32),
        rgb=ouc.ExtendedColor.BEIGE)
    lnk4.set_inertia(mass=5.066)
    lnk5 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link5.stl"),
        collision_type=ouc.CollisionType.MESH,
        loc_rotmat=oum.rotmat_from_euler(0, np.pi / 2, 0),
        rgb=ouc.ExtendedColor.BEIGE)
    lnk5.set_inertia(mass=1.625)
    lnk6 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link6.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    lnk6.set_inertia(mass=0.625)
    # 6 joints
    jnt_bl_l1 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk, child_lnk=lnk1,
        axis=-ouc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.36], dtype=np.float32),
        lmt_lo=-np.pi, lmt_up=np.pi)
    jnt_l1_l2 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk1, child_lnk=lnk2,
        axis=ouc.StandardAxis.Z,
        rotmat=oum.rotmat_from_euler(0, -np.pi / 2, 0),
        lmt_lo=-3 / 4 * np.pi, lmt_up=3 / 4 * np.pi)
    jnt_l2_l3 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk2, child_lnk=lnk3,
        axis=-ouc.StandardAxis.Z,
        pos=np.array([0.455, 0.0, 0.0], dtype=np.float32),
        lmt_lo=-7 / 8 * np.pi, lmt_up=7 / 8 * np.pi)
    jnt_l3_l4 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk3, child_lnk=lnk4,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.0925, 0.0, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(0, np.pi / 2, 0),
        lmt_lo=-10 / 9 * np.pi, lmt_up=10 / 9 * np.pi)
    jnt_l4_l5 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk4, child_lnk=lnk5,
        axis=-ouc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.3825], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(0, -np.pi / 2, 0),
        lmt_lo=-25 / 36 * np.pi, lmt_up=25 / 36 * np.pi)
    jnt_l5_l6 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk5, child_lnk=lnk6,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.078, 0.0, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(0, np.pi / 2, 0),
        lmt_lo=-2 * np.pi, lmt_up=2 * np.pi)
    # add links
    structure.add_lnk(base_lnk)
    structure.add_lnk(lnk1)
    structure.add_lnk(lnk2)
    structure.add_lnk(lnk3)
    structure.add_lnk(lnk4)
    structure.add_lnk(lnk5)
    structure.add_lnk(lnk6)
    # add joints
    structure.add_jnt(jnt_bl_l1)
    structure.add_jnt(jnt_l1_l2)
    structure.add_jnt(jnt_l2_l3)
    structure.add_jnt(jnt_l3_l4)
    structure.add_jnt(jnt_l4_l5)
    structure.add_jnt(jnt_l5_l6)
    # ignore collisions pairs
    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    # order joints for quick access
    structure.compile()
    return structure


class RS007L(ormmb.ManipulatorBase):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
