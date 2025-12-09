import os

import numpy as np

import one.utils.constant as const
import one.utils.math as rm
import one.robot_sim.base.robot_state as rstate
import one.robot_sim.base.robot_structure as rstruct


class RS007L:

    def __init__(self):
        mesh_dir = os.path.join(os.path.dirname(__file__), "meshes")
        self.structure = rstruct.RobotStructure()
        # 7 links
        base_link = rstruct.Link.from_file(os.path.joint(mesh_dir, "base_link.stl"),
                                           rgb=const.ExtendedColor.BEIGE)
        self.structure.add_link("base_link", base_link)
        link1 = rstruct.Link.from_file(os.path.join(mesh_dir, "link1.stl"),
                                       rgb=const.ExtendedColor.BEIGE)
        self.structure.add_link("link1", link1)
        link2 = rstruct.Link.from_file(os.path.join(mesh_dir, "link2.stl"),
                                       rgb=const.ExtendedColor.BEIGE)
        self.structure.add_link("link2", link2)
        link3 = rstruct.Link.from_file(os.path.join(mesh_dir, "link3.stl"),
                                       rgb=const.ExtendedColor.BEIGE)
        self.structure.add_link("link3", link3)
        link4 = rstruct.Link.from_file(os.path.join(mesh_dir, "link4.stl"),
                                       rgb=const.ExtendedColor.BEIGE)
        self.structure.add_link("link4", link4)
        link5 = rstruct.Link.from_file(os.path.join(mesh_dir, "link5.stl"),
                                       rgb=const.ExtendedColor.BEIGE)
        self.structure.add_link("link5", link5)
        link6 = rstruct.Link.from_file(os.path.join(mesh_dir, "link6.stl"),
                                       rgb=const.ExtendedColor.BEIGE)
        self.structure.add_link("link6", link6)
        # 6 joints
        joint_bl_l1 = rstruct.Joint("joint_bl_l1", joint_type=const.JointType.REVOLUTE,
                                    parent_link=base_link, child_link=link1,
                                    axis=-const.StandardAxis.Z,
                                    origin_pos=np.array([0.0, 0.0, 0.36], dtype=np.float32),
                                    limit_lower=-np.pi, limit_upper=np.pi)
        self.structure.add_joint(joint_bl_l1)
        joint_l1_l2 = rstruct.Joint("joint_l1_12", joint_type=const.JointType.REVOLUTE,
                                    parent_link=link1, child_link=link2,
                                    axis=const.StandardAxis.Z,
                                    origin_rotmat=rm.rotmat_from_euler(0, -np.pi / 2, 0),
                                    limit_lower=-3 / 4 * np.pi, limit_upper=3 / 4 * np.pi)
        self.structure.add_joint(joint_l1_l2)
        joint_l2_l3 = rstruct.Joint("joint_l2_l3", joint_type=const.JointType.REVOLUTE,
                                    parent_link=link2, child_link=link3,
                                    axis=-const.StandardAxis.Z,
                                    origin_pos=np.array([0.455, 0.0, 0.0], dtype=np.float32),
                                    limit_lower=-7 / 8 * np.pi, limit_upper=7 / 8 * np.pi)
        self.structure.add_joint(joint_l2_l3)
        joint_l3_l4 = rstruct.Joint("joint_l3_l4", joint_type=const.JointType.REVOLUTE,
                                    parent_link=link3, child_link=link4,
                                    axis=const.StandardAxis.Z,
                                    origin_pos=np.array([0.0925, 0.0, 0.0], dtype=np.float32),
                                    origin_rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                                    limit_lower=-10/9*np.pi, limit_upper=10/9*np.pi)
        self.structure.add_joint(joint_l3_l4)
        joint_l4_l5 = rstruct.Joint("joint_l4_l5", joint_type=const.JointType.REVOLUTE,
                                    parent_link=link4, child_link=link5,
                                    axis=-const.StandardAxis.Z,
                                    origin_pos=np.array([0.0, 0.0, 0.3825], dtype=np.float32),
                                    origin_rotmat=rm.rotmat_from_euler(0, -np.pi / 2, 0),
                                    limit_lower=-25/36*np.pi, limit_upper=25/36*np.pi)
        self.structure.add_joint(joint_l4_l5)
        joint_l5_l6 = rstruct.Joint("joint_l5_l6", joint_type=const.JointType.REVOLUTE,
                                    parent_link=link5, child_link=link6,
                                    axis=const.StandardAxis.Z,
                                    origin_pos=np.array([0.078, 0.0, 0.094], dtype=np.float32),
                                    origin_rotmat=rm.rotmat_from_euler(0, np.pi / 2, 0),
                                    limit_lower=-2 * np.pi, limit_upper=2 * np.pi)
        self.structure.add_joint(joint_l5_l6)
        self.structure.finalize()
        # default robot pose
        self.default_qs = np.zeros(6, dtype=np.float32)
        # robot state
        self.state = rstate.RobotState(self.structure)
        self.state.set_qs(values = self.default_qs)

    def fk(self, qs=None, root_tfmat=None):
        if qs is not None:
            self.state.set_qs(qs)
        self.state.fk(root_tfmat)
        return self.state.link_wd_tfmats

    def attach_to(self, scene):
        for link in self.visual_links:
            scene.add(link)

    def update(self):
        tfmat = self.state.link_wd_tfmats
        for i, link in enumerate(self.visual_links):
            link.set_tfmat(T[i])

    def snapshot(self):
        """Return a snapshot object that can be drawn independently."""
        snap_links = []
        T = self.state.link_wd_tfmats
        for i, link in enumerate(self.structure.link_order):
            clone = link.clone()
            clone.set_tfmat(T[i])
            snap_links.append(clone)
        return snap_links  # or wrap in a class like Snapshot