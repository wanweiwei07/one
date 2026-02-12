import os
import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.kine.numik as orbkn
import one.robots.base.mech_structure as orbms
import one.robots.manipulators.manipulator_base as ormmb

def prepare_mechstruct():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir

    # UR3 kinematics from Universal_Robots_ROS2_Description:
    # config/ur3/default_kinematics.yaml
    shoulder_xyz = np.array([0.0, 0.0, 0.1519], dtype=np.float32)
    shoulder_rpy = (0.0, 0.0, 0.0)
    upper_arm_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper_arm_rpy = (1.570796327, 0.0, 0.0)
    forearm_xyz = np.array([-0.24365, 0.0, 0.0], dtype=np.float32)
    forearm_rpy = (0.0, 0.0, 0.0)
    wrist1_xyz = np.array([-0.21325, 0.0, 0.11235], dtype=np.float32)
    wrist1_rpy = (0.0, 0.0, 0.0)
    wrist2_xyz = np.array([0.0, -0.08535, -1.750557762378351e-11], dtype=np.float32)
    wrist2_rpy = (1.570796327, 0.0, 0.0)
    wrist3_xyz = np.array([0.0, 0.0819, -1.679797079540562e-11], dtype=np.float32)
    wrist3_rpy = (1.570796326589793, np.pi, np.pi)

    base_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "base.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    lnk1 = orbms.Link.from_file(
        os.path.join(mesh_dir, "shoulder.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    lnk2 = orbms.Link.from_file(
        os.path.join(mesh_dir, "upperarm.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    lnk3 = orbms.Link.from_file(
        os.path.join(mesh_dir, "forearm.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    lnk4 = orbms.Link.from_file(
        os.path.join(mesh_dir, "wrist1.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    lnk5 = orbms.Link.from_file(
        os.path.join(mesh_dir, "wrist2.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)
    lnk6 = orbms.Link.from_file(
        os.path.join(mesh_dir, "wrist3.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE)

    # 6 revolute joints, following UR kinematic convention in this framework
    jnt_b_l1 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=base_lnk, child_lnk=lnk1,
        axis=ouc.StandardAxis.Z,
        pos=shoulder_xyz,
        rotmat=oum.rotmat_from_euler(*shoulder_rpy),
        lmt_lo=-2.0 * np.pi, lmt_up=2.0 * np.pi)
    jnt_l1_l2 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk1, child_lnk=lnk2,
        axis=ouc.StandardAxis.Z,
        pos=upper_arm_xyz,
        rotmat=oum.rotmat_from_euler(*upper_arm_rpy),
        lmt_lo=-2.0 * np.pi, lmt_up=2.0 * np.pi)
    jnt_l2_l3 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk2, child_lnk=lnk3,
        axis=ouc.StandardAxis.Z,
        pos=forearm_xyz,
        rotmat=oum.rotmat_from_euler(*forearm_rpy),
        lmt_lo=-2.0 * np.pi, lmt_up=2.0 * np.pi)
    jnt_l3_l4 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk3, child_lnk=lnk4,
        axis=ouc.StandardAxis.Z,
        pos=wrist1_xyz,
        rotmat=oum.rotmat_from_euler(*wrist1_rpy),
        lmt_lo=-2.0 * np.pi, lmt_up=2.0 * np.pi)
    jnt_l4_l5 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk4, child_lnk=lnk5,
        axis=ouc.StandardAxis.Z,
        pos=wrist2_xyz,
        rotmat=oum.rotmat_from_euler(*wrist2_rpy),
        lmt_lo=-2.0 * np.pi, lmt_up=2.0 * np.pi)
    jnt_l5_l6 = orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=lnk5, child_lnk=lnk6,
        axis=ouc.StandardAxis.Z,
        pos=wrist3_xyz,
        rotmat=oum.rotmat_from_euler(*wrist3_rpy),
        lmt_lo=-2.0 * np.pi, lmt_up=2.0 * np.pi)

    for lnk in [base_lnk, lnk1, lnk2, lnk3, lnk4, lnk5, lnk6]:
        structure.add_lnk(lnk)
    for jnt in [jnt_b_l1, jnt_l1_l2, jnt_l2_l3, jnt_l3_l4, jnt_l4_l5, jnt_l5_l6]:
        structure.add_jnt(jnt)

    structure.ignore_collision(base_lnk, lnk2)
    structure.ignore_collision(lnk1, lnk3)
    structure.ignore_collision(lnk2, lnk4)
    structure.ignore_collision(lnk3, lnk5)
    structure.ignore_collision(lnk4, lnk6)
    structure.compile()
    return structure


class UR3(ormmb.ManipulatorBase):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)

    def get_solver(self, chain):
        if chain not in self._solvers:
            self._solvers[chain] = orbkn.NumIKSolver(chain)
        return self._solvers[chain]

if __name__ == '__main__':
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    import builtins
    
    base = ovw.World(
        cam_pos=(2,1,1), cam_lookat_pos=(0, 0, 0.5)
    )
    builtins.base = base
    scene = base.scene
    oframe = ossop.frame()
    oframe.attach_to(scene)
    robot = UR3()
    robot.attach_to(scene)
    builtins.robot = robot
    base.run()
