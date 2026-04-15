import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.manipulators.manipulator_base as ormmb
import one.robots.base.kine.anaik as orbka


def prepare_mechstruct():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    # 7 links (mesh origins match URDF link frames directly)
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "base_link.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.BasicColor.WHITE,
    )
    base_lnk.set_inertia(mass=1.514)
    lnk1 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link1.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.BasicColor.WHITE,
    )
    lnk1.set_inertia(mass=1.181)
    lnk2 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link2.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.BasicColor.WHITE,
    )
    lnk2.set_inertia(mass=1.246)
    lnk3 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link3.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.BasicColor.WHITE,
    )
    lnk3.set_inertia(mass=1.052)
    lnk4 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link4.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.BasicColor.WHITE,
    )
    lnk4.set_inertia(mass=0.821)
    lnk5 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link5.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.BasicColor.WHITE,
    )
    lnk5.set_inertia(mass=0.406)
    lnk6 = orbms.Link.from_file(
        os.path.join(mesh_dir, "link6.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.BasicColor.WHITE,
    )
    lnk6.set_inertia(mass=0.18)
    # 6 joints — pos/rpy taken verbatim from xarm_description lite6 URDF
    jnt_bl_l1 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lnk1,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.2435], dtype=np.float32),
        lmt_lo=-2 * np.pi,
        lmt_up=2 * np.pi,
    )
    jnt_l1_l2 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk1,
        child_lnk=lnk2,
        axis=ouc.StandardAxis.Z,
        rotmat=oum.rotmat_from_euler(np.pi / 2, -np.pi / 2, np.pi),
        lmt_lo=-2.61799,
        lmt_up=2.61799,
    )
    jnt_l2_l3 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk2,
        child_lnk=lnk3,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.2002, 0.0, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(-np.pi, 0, np.pi / 2),
        lmt_lo=-0.061087,
        lmt_up=5.235988,
    )
    jnt_l3_l4 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk3,
        child_lnk=lnk4,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.087, -0.22761, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=-2 * np.pi,
        lmt_up=2 * np.pi,
    )
    jnt_l4_l5 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk4,
        child_lnk=lnk5,
        axis=ouc.StandardAxis.Z,
        rotmat=oum.rotmat_from_euler(np.pi / 2, 0, 0),
        lmt_lo=-2.1642,
        lmt_up=2.1642,
    )
    jnt_l5_l6 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk5,
        child_lnk=lnk6,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.0, 0.0625, 0.0], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(-np.pi / 2, 0, 0),
        lmt_lo=-2 * np.pi,
        lmt_up=2 * np.pi,
    )
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


class Lite6(ormmb.ManipulatorBase):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)

    def get_solver(self, chain):
        if chain not in self._solvers:
            joint_limits = (chain.lmt_lo, chain.lmt_up)
            self._solvers[chain] = orbka.S456X12(chain, joint_limits)
        return self._solvers[chain]


if __name__ == "__main__":
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=[1.5, 1.0, 1.0], cam_lookat_pos=[0.0, 0.0, 0.3])
    robot = Lite6()
    print("=== Lite6 IK Test ===")
    builtins.base = base
    builtins.robot = robot  # for debug access

    tgt_pos = (0.3, 0.1, 0.3)
    tgt_rotmat = oum.rotmat_from_axangle(
        ouc.StandardAxis.Z, 0
    ) @ oum.rotmat_from_axangle(ouc.StandardAxis.Y, oum.pi)
    ossop.frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base.scene)

    prev_qs = np.zeros(6)
    qs = robot.ik_tcp_nearest(tgt_rotmat=tgt_rotmat,
                              tgt_pos=tgt_pos, ref_qs=prev_qs)
    if qs is not None:
        print("Found IK solution:", qs)
        tmp_robot = robot.clone()
        tmp_robot.fk(qs=qs)
        tmp_robot.attach_to(base.scene)
    base.run()
