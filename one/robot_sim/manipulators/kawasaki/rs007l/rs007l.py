import os
import numpy as np
import one.robot_sim.base.mech_structure as rstruct
import one.robot_sim.manipulators.manipulator_base as mbase
import one.utils.constant as const
import one.utils.math as rm


def prepare_mechstruct():
    structure = rstruct.MechStruct()
    mesh_dir = os.path.join(os.path.dirname(__file__), "meshes")
    # 7 links
    base_link = rstruct.Link.from_file(os.path.join(mesh_dir, "base_link.stl"),
                                       collision_type=const.CollisionType.MESH,
                                       is_fixed=True, name="base_link",
                                       rgb=const.ExtendedColor.BEIGE)
    base_link.set_inertia(mass=11.0)
    link1 = rstruct.Link.from_file(os.path.join(mesh_dir, "link1.stl"),
                                   collision_type=const.CollisionType.MESH,
                                   name="link1", rgb=const.ExtendedColor.BEIGE)
    link1.set_inertia(mass=8.118)
    link2 = rstruct.Link.from_file(os.path.join(mesh_dir, "link2.stl"),
                                   collision_type=const.CollisionType.MESH,
                                   local_rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                                   name="link2", rgb=const.ExtendedColor.BEIGE)
    link2.set_inertia(mass=6.826)
    link3 = rstruct.Link.from_file(os.path.join(mesh_dir, "link3.stl"),
                                   collision_type=const.CollisionType.MESH,
                                   local_rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                                   name="link3", rgb=const.ExtendedColor.BEIGE)
    link3.set_inertia(mass=5.236)
    link4 = rstruct.Link.from_file(os.path.join(mesh_dir, "link4.stl"),
                                   collision_type=const.CollisionType.MESH,
                                   local_pos=np.array([0.0, 0.0, 0.3852], dtype=np.float32),
                                   name="link4", rgb=const.ExtendedColor.BEIGE)
    link4.set_inertia(mass=5.066)
    link5 = rstruct.Link.from_file(os.path.join(mesh_dir, "link5.stl"),
                                   collision_type=const.CollisionType.MESH,
                                   local_rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                                   name="link5", rgb=const.ExtendedColor.BEIGE)
    link5.set_inertia(mass=1.625)
    link6 = rstruct.Link.from_file(os.path.join(mesh_dir, "link6.stl"),
                                   collision_type=const.CollisionType.MESH,
                                   name="link6", rgb=const.ExtendedColor.BEIGE)
    link6.set_inertia(mass=0.625)
    # 6 joints
    joint_bl_l1 = rstruct.Joint("joint_bl_l1", jnt_type=const.JntType.REVOLUTE,
                                parent_lnk=base_link, child_lnk=link1,
                                axis=-const.StandardAxis.Z,
                                pos=np.array([0.0, 0.0, 0.36], dtype=np.float32),
                                lmt_low=-np.pi, lmt_up=np.pi)
    joint_l1_l2 = rstruct.Joint("joint_l1_12", jnt_type=const.JntType.REVOLUTE,
                                parent_lnk=link1, child_lnk=link2,
                                axis=const.StandardAxis.Z,
                                rotmat=rm.rotmat_from_euler(0, -np.pi / 2, 0),
                                lmt_low=-3 / 4 * np.pi, lmt_up=3 / 4 * np.pi)
    joint_l2_l3 = rstruct.Joint("joint_l2_l3", jnt_type=const.JntType.REVOLUTE,
                                parent_lnk=link2, child_lnk=link3,
                                axis=-const.StandardAxis.Z,
                                pos=np.array([0.455, 0.0, 0.0], dtype=np.float32),
                                lmt_low=-7 / 8 * np.pi, lmt_up=7 / 8 * np.pi)
    joint_l3_l4 = rstruct.Joint("joint_l3_l4", jnt_type=const.JntType.REVOLUTE,
                                parent_lnk=link3, child_lnk=link4,
                                axis=const.StandardAxis.Z,
                                pos=np.array([0.0925, 0.0, 0.0], dtype=np.float32),
                                rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                                lmt_low=-10 / 9 * np.pi, lmt_up=10 / 9 * np.pi)
    joint_l4_l5 = rstruct.Joint("joint_l4_l5", jnt_type=const.JntType.REVOLUTE,
                                parent_lnk=link4, child_lnk=link5,
                                axis=-const.StandardAxis.Z,
                                pos=np.array([0.0, 0.0, 0.3825], dtype=np.float32),
                                rotmat=rm.rotmat_from_euler(0, -np.pi / 2, 0),
                                lmt_low=-25 / 36 * np.pi, lmt_up=25 / 36 * np.pi)
    joint_l5_l6 = rstruct.Joint("joint_l5_l6", jnt_type=const.JntType.REVOLUTE,
                                parent_lnk=link5, child_lnk=link6,
                                axis=const.StandardAxis.Z,
                                pos=np.array([0.078, 0.0, 0.0], dtype=np.float32),
                                rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                                lmt_low=-2 * np.pi, lmt_up=2 * np.pi)
    # add links
    structure.add_lnk(base_link)
    structure.add_lnk(link1)
    structure.add_lnk(link2)
    structure.add_lnk(link3)
    structure.add_lnk(link4)
    structure.add_lnk(link5)
    structure.add_lnk(link6)
    # add joints
    structure.add_jnt(joint_bl_l1)
    structure.add_jnt(joint_l1_l2)
    structure.add_jnt(joint_l2_l3)
    structure.add_jnt(joint_l3_l4)
    structure.add_jnt(joint_l4_l5)
    structure.add_jnt(joint_l5_l6)
    # order joints for quick access
    structure.compile()
    return structure


class RS007L(mbase.ManipulatorBase):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, base_rotmat=None, base_pos=None):
        super().__init__(base_rotmat=base_rotmat, base_pos=base_pos)

    def engage(self, child, engage_tfmat=None, update=True):
        super().mount(child=child,
                      parent_link=self.structure.lnks[-1],
                      engage_tfmat=engage_tfmat)
        if update:
            self._update_mounting(self._mountings[child])

    def mount(self, *args, **kwargs):
        """turn off mount() to avoid confusion"""
        raise RuntimeError("RS007L.mount() is disabled. "
                           "Use engage(child, engage_tfmat) instead.")
