import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.manipulators.manipulator_base as ormmb


def prepare_mechstruct():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    fr_white = ouc.BasicColor.WHITE
    # 8 links: base (link0) + link1..link7
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "link0.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    base_lnk.set_inertia(mass=2.3)
    lnk1 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link1.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk1.set_inertia(mass=2.74)
    lnk2 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link2.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk2.set_inertia(mass=2.74)
    lnk3 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link3.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk3.set_inertia(mass=2.38)
    lnk4 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link4.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk4.set_inertia(mass=2.38)
    lnk5 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link5.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk5.set_inertia(mass=2.74)
    lnk6 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link6.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk6.set_inertia(mass=1.55)
    lnk7 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link7.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=fr_white,
    )
    lnk7.set_inertia(mass=0.54)
    # 7 revolute joints (kinematics from FR3 URDF)
    jnt_bl_l1 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lnk1,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.333], dtype=np.float32),
        lmt_lo=-2.7437,
        lmt_up=2.7437,
    )
    jnt_l1_l2 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk1,
        child_lnk=lnk2,
        axis=ouc.StandardAxis.Z,
        rotmat=oum.rotmat_from_euler(-np.pi / 2, 0, 0),
        lmt_lo=-1.7837,
        lmt_up=1.7837,
    )
    jnt_l2_l3 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk2,
        child_lnk=lnk3,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.0, -0.316, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=-2.9007,
        lmt_up=2.9007,
    )
    jnt_l3_l4 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk3,
        child_lnk=lnk4,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.0825, 0.0, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=-3.0421,
        lmt_up=-0.1518,
    )
    jnt_l4_l5 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk4,
        child_lnk=lnk5,
        axis=ouc.StandardAxis.Z,
        pos=np.array([-0.0825, 0.384, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(-np.pi / 2, 0, 0),
        lmt_lo=-2.8065,
        lmt_up=2.8065,
    )
    jnt_l5_l6 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk5,
        child_lnk=lnk6,
        axis=ouc.StandardAxis.Z,
        rotmat=oum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=0.5445,
        lmt_up=4.5169,
    )
    jnt_l6_l7 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk6,
        child_lnk=lnk7,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.088, 0.0, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=-3.0159,
        lmt_up=3.0159,
    )
    # add links
    structure.add_lnk(base_lnk)
    structure.add_lnk(lnk1)
    structure.add_lnk(lnk2)
    structure.add_lnk(lnk3)
    structure.add_lnk(lnk4)
    structure.add_lnk(lnk5)
    structure.add_lnk(lnk6)
    structure.add_lnk(lnk7)
    # add joints
    structure.add_jnt(jnt_bl_l1)
    structure.add_jnt(jnt_l1_l2)
    structure.add_jnt(jnt_l2_l3)
    structure.add_jnt(jnt_l3_l4)
    structure.add_jnt(jnt_l4_l5)
    structure.add_jnt(jnt_l5_l6)
    structure.add_jnt(jnt_l6_l7)
    # ignore collisions pairs
    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    structure.ignore_collision(lnk5, lnk7)
    # order joints for quick access
    structure.compile()
    return structure


class FR3(ormmb.ManipulatorBase):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(
            rotmat=rotmat,
            pos=pos,
            home_qs=[0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
        )


if __name__ == "__main__":
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=[2.0, 1.0, 1.0], cam_lookat_pos=[0.0, 0.0, 0.5])
    robot = FR3()
    builtins.base = base
    builtins.robot = robot
    robot.attach_to(base.scene)
    ossop.frame().attach_to(base.scene)
    base.run()