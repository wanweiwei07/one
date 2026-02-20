import os
import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.base.kine.numik as orbkn
import one.robots.manipulators.manipulator_base as ormmb


def prepare_mechstruct():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    # Source:
    # https://github.com/DENSORobot/denso_robot_ros2
    # denso_robot_descriptions/robots/cobotta/urdf/denso_robot_kinematics.xacro
    # (DAE meshes converted to STL and stored in this folder.)

    # Link meshes
    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "base_link.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE,
    )
    lnk1 = orbms.Link.from_file(
        os.path.join(mesh_dir, "j1.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE,
    )
    lnk2 = orbms.Link.from_file(
        os.path.join(mesh_dir, "j2.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE,
    )
    lnk3 = orbms.Link.from_file(
        os.path.join(mesh_dir, "j3.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE,
    )
    lnk4 = orbms.Link.from_file(
        os.path.join(mesh_dir, "j4.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE,
    )
    lnk5 = orbms.Link.from_file(
        os.path.join(mesh_dir, "j5.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE,
    )
    lnk6 = orbms.Link.from_file(
        os.path.join(mesh_dir, "j6.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE,
    )

    # Cobotta CVR038A1-NV6-NN kinematic layout from official xacro.
    jnt_b_l1 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk,
        child_lnk=lnk1,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        lmt_lo=-2.61799387799149,
        lmt_up=2.61799387799149,
    )
    jnt_l1_l2 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk1,
        child_lnk=lnk2,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0.0, 0.0, 0.18], dtype=np.float32),
        lmt_lo=-1.0471975511966,
        lmt_up=1.74532925199433,
    )
    jnt_l2_l3 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk2,
        child_lnk=lnk3,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0.0, 0.0, 0.165], dtype=np.float32),
        lmt_lo=0.314159265358979,
        lmt_up=2.44346095279206,
    )
    jnt_l3_l4 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk3,
        child_lnk=lnk4,
        axis=ouc.StandardAxis.Z,
        pos=np.array([-0.012, 0.020, -0.345], dtype=np.float32),
        lmt_lo=-2.96705972839036,
        lmt_up=2.96705972839036,
    )
    jnt_l4_l5 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk4,
        child_lnk=lnk5,
        axis=ouc.StandardAxis.Y,
        pos=np.array([0.0, -0.020, 0.5225], dtype=np.float32),
        lmt_lo=-1.65806278939461,
        lmt_up=2.35619449019234,
    )
    jnt_l5_l6 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk5,
        child_lnk=lnk6,
        axis=ouc.StandardAxis.Z,
        pos=np.array([0.0, -0.0445, 0.042], dtype=np.float32),
        lmt_lo=-2.96705972839036,
        lmt_up=2.96705972839036,
    )

    for lnk in [base_lnk, lnk1, lnk2, lnk3, lnk4, lnk5, lnk6]:
        structure.add_lnk(lnk)
    for jnt in [jnt_b_l1, jnt_l1_l2, jnt_l2_l3, jnt_l3_l4, jnt_l4_l5, jnt_l5_l6]:
        structure.add_jnt(jnt)

    # Typical adjacent-link ignore pairs.
    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    structure.compile()
    return structure


class CVR038(ormmb.ManipulatorBase):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)

    def get_solver(self, chain):
        # Keep CVR038 robust first; can be switched to analytic IK later.
        if chain not in self._solvers:
            self._solvers[chain] = orbkn.NumIKSolver(chain)
        return self._solvers[chain]


if __name__ == "__main__":
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=(1.8, 1.0, 1.2), cam_lookat_pos=(0.0, 0.0, 0.45))
    ossop.frame().attach_to(base.scene)
    robot = CVR038()
    robot.attach_to(base.scene)

    tgt_pos = np.array([0.25, 0.15, 0.25], dtype=np.float32)
    tgt_rotmat = (
        oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 6.0)
        @ oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi)
    )
    ossop.frame(pos=tgt_pos, rotmat=tgt_rotmat, color_mat=ouc.CoordColor.DYO).attach_to(base.scene)

    qs = robot.ik_tcp_nearest(tgt_rotmat=tgt_rotmat, tgt_pos=tgt_pos)
    print("ik:", qs)
    if qs is not None:
        tmp_robot = robot.clone()
        tmp_robot.fk(qs=qs)
        tmp_robot.attach_to(base.scene)
        ossop.frame(
            pos=tmp_robot.gl_tcp_tf[:3, 3],
            rotmat=tmp_robot.gl_tcp_tf[:3, :3],
            color_mat=ouc.CoordColor.MYC,
        ).attach_to(base.scene)
    base.run()
