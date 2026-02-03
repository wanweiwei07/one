import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.base.mech_base as orbmb
import one.robots.manipulators.manipulator_base as ormmb


def _inertia_3x3(ixx, ixy, ixz, iyy, iyz, izz):
    """(ixx, ixy, ixz, iyy, iyz, izz) -> 3x3 symmetric matrix."""
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float32)


lnk_pos = (
    np.array([0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, -0.0625], dtype=np.float32),
    np.array([0.0301, 0.0, -0.1225], dtype=np.float32),
    np.array([0.0, 0.0, -0.18875], dtype=np.float32),
    np.array([0.0, -0.0315, -0.3425], dtype=np.float32),
    np.array([0.0, 0.0, -0.438], dtype=np.float32),
    np.array([-0.0375, 0.0, -0.5585], dtype=np.float32),
    np.array([0.0, 0.0, -0.5585], dtype=np.float32),)
lnk_mass = (1.143, 1.142, 0.278, 1.074, 0.635, 0.616, 0.475, 0.466)
lnk_com = (
    np.array([-0.000948, -0.000158, 0.03077], dtype=np.float32),
    np.array([0.001147, -3.32e-5, 0.05395], dtype=np.float32),
    np.array([0.008396, 0.0, 0.03257], dtype=np.float32),
    np.array([-0.002105, -0.000555, 0.09047], dtype=np.float32),
    np.array([-0.002901, -0.03031, 0.06340], dtype=np.float32),
    np.array([-0.003050, -0.000887, 0.04308], dtype=np.float32),
    np.array([-0.03714, -0.000332, -9.5e-5], dtype=np.float32),
    np.array([6.88e-5, -0.01266, 0.06952], dtype=np.float32),)
lnk_intertia = (
    _inertia_3x3(0.001128, -4e-6, -3.3e-5, 0.000962, -7e-6, 0.00147),
    _inertia_3x3(0.001567, -1e-6, -2.9e-5, 0.001273, 1e-6, 0.001016),
    _inertia_3x3(0.000359, 1e-6, -0.000109, 0.000376, 1e-6, 0.000232),
    _inertia_3x3(0.004372, 1e-6, 1.1e-5, 0.004319, -3.6e-5, 0.000661),
    _inertia_3x3(0.000623, -1e-6, -1.9e-5, 0.000511, 3.8e-5, 0.000334),
    _inertia_3x3(0.000423, -8e-6, 6e-6, 0.000445, -6e-6, 0.000324),
    _inertia_3x3(0.000143, 1e-6, 1e-6, 0.000157, 1e-6, 0.000159),
    _inertia_3x3(0.000639, 1e-6, 1e-6, 0.000497, 8.9e-5, 0.000342),)

jnt_lmts = {
    # left arm joints
    'lft_jnt1': {'lo': -1.396263, 'up': 3.490659},
    'lft_jnt2': {'lo': -3.316125, 'up': 0.174533},
    'lft_jnt3': {'lo': -1.570796, 'up': 1.570796},
    'lft_jnt4': {'lo': 0.0, 'up': 2.443461},
    'lft_jnt5': {'lo': -1.570796, 'up': 1.570796},
    'lft_jnt6': {'lo': -0.785398, 'up': 0.785398},
    'lft_jnt7': {'lo': -1.570796, 'up': 1.570796},
    # right arm joints
    'rgt_jnt1': {'lo': -1.396263, 'up': 3.490659},
    'rgt_jnt2': {'lo': -0.174533, 'up': 3.316125},
    'rgt_jnt3': {'lo': -1.570796, 'up': 1.570796},
    'rgt_jnt4': {'lo': 0.0, 'up': 2.443461},
    'rgt_jnt5': {'lo': -1.570796, 'up': 1.570796},
    'rgt_jnt6': {'lo': -0.785398, 'up': 0.785398},
    'rgt_jnt7': {'lo': -1.570796, 'up': 1.570796}}
jnt_kin = {
    # left arm joints
    'lft_jnt1': {'xyz': [0.0, 0.0, 0.0625], 'rpy': [0, 0, 0]},
    'lft_jnt2': {'xyz': [-0.0301, 0.0, 0.06], 'rpy': [-np.pi / 2, 0, 0]},
    'lft_jnt3': {'xyz': [0.0301, 0.0, 0.06625], 'rpy': [0, 0, 0]},
    'lft_jnt4': {'xyz': [0.0, 0.0315, 0.15375], 'rpy': [0, 0, 0]},
    'lft_jnt5': {'xyz': [0.0, -0.0315, 0.0955], 'rpy': [0, 0, 0]},
    'lft_jnt6': {'xyz': [0.0375, 0.0, 0.1205], 'rpy': [0, 0, 0]},
    'lft_jnt7': {'xyz': [-0.0375, 0.0, 0.0], 'rpy': [0, 0, 0]},
    # right arm joints
    'rgt_jnt1': {'xyz': [0.0, 0.0, 0.0625], 'rpy': [0, 0, 0]},
    'rgt_jnt2': {'xyz': [-0.0301, 0.0, 0.06], 'rpy': [np.pi / 2, 0, 0]},
    'rgt_jnt3': {'xyz': [0.0301, 0.0, 0.06625], 'rpy': [0, 0, 0]},
    'rgt_jnt4': {'xyz': [0.0, 0.0315, 0.15375], 'rpy': [0, 0, 0]},
    'rgt_jnt5': {'xyz': [0.0, -0.0315, 0.0955], 'rpy': [0, 0, 0]},
    'rgt_jnt6': {'xyz': [0.0375, 0.0, 0.1205], 'rpy': [0, 0, 0]},
    'rgt_jnt7': {'xyz': [-0.0375, 0.0, 0.0], 'rpy': [0, 0, 0]}}
jnt_ax = {
    # left arm joints
    'lft_jnt1': ouc.StandardAxis.Z,  # z
    'lft_jnt2': -ouc.StandardAxis.X,  # -x
    'lft_jnt3': ouc.StandardAxis.Z,  # z
    'lft_jnt4': ouc.StandardAxis.Y,  # y
    'lft_jnt5': ouc.StandardAxis.Z,  # z
    'lft_jnt6': ouc.StandardAxis.X,  # x
    'lft_jnt7': -ouc.StandardAxis.Y,
    # right arm joints
    'rgt_jnt1': ouc.StandardAxis.Z,  # z
    'rgt_jnt2': -ouc.StandardAxis.X,  # -x
    'rgt_jnt3': ouc.StandardAxis.Z,  # z
    'rgt_jnt4': ouc.StandardAxis.Y,  # y
    'rgt_jnt5': ouc.StandardAxis.Z,  # z
    'rgt_jnt6': ouc.StandardAxis.X,  # x
    'rgt_jnt7': ouc.StandardAxis.Y}  # -y


def link(prefix, name, idx, mesh_dir,
         rgb=ouc.ExtendedColor.DEEP_GRAY):
    loc_pos = lnk_pos[idx]
    mass = lnk_mass[idx]
    com = lnk_com[idx]
    inrtmat = lnk_intertia[idx]
    path = os.path.join(mesh_dir, f"{name}_symp.stl")
    if os.path.isfile(path):
        scale = (0.001, 0.001, 0.001)
        if prefix == 'lft' and idx == 7:
            scale = (0.001, -0.001, 0.001)
        lnk = orbms.Link.from_file(
            path, loc_pos=loc_pos,
            collision_type=ouc.CollisionType.MESH,
            rgb=rgb, scale=scale)
        lnk.set_inertia(inrtmat=inrtmat, com=com, mass=mass)
        return lnk
    else:
        raise FileNotFoundError(f"Mesh file {path} not found")


def joint(prefix, name, parent_lnk, child_lnk):
    name = prefix + "_" + name
    axis = jnt_ax[name]
    pos = jnt_kin[name]["xyz"]
    rotmat = oum.rotmat_from_euler(*jnt_kin[name]["rpy"])
    lmt_lo = jnt_lmts[name]["lo"]
    lmt_up = jnt_lmts[name]["up"]
    return orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE,
        parent_lnk=parent_lnk, child_lnk=child_lnk,
        axis=axis,
        pos=pos, rotmat=rotmat,
        lmt_lo=lmt_lo, lmt_up=lmt_up)


def prepare_arm_ms(prefix="lft"):
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    # 7 links
    base_lnk = link(prefix, "link0", 0, mesh_dir)
    lnk1 = link(prefix, "link1", 1, mesh_dir)
    lnk2 = link(prefix, "link2", 2, mesh_dir)
    lnk3 = link(prefix, "link3", 3, mesh_dir)
    lnk4 = link(prefix, "link4", 4, mesh_dir)
    lnk5 = link(prefix, "link5", 5, mesh_dir)
    lnk6 = link(prefix, "link6", 6, mesh_dir,
                rgb=ouc.ExtendedColor.STEEL_GRAY)
    lnk7 = link(prefix, "link7", 7, mesh_dir)
    # 7 joints
    jnt_bl_l1 = joint(prefix, "jnt1", base_lnk, lnk1)
    jnt_l1_l2 = joint(prefix, "jnt2", lnk1, lnk2)
    jnt_l2_l3 = joint(prefix, "jnt3", lnk2, lnk3)
    jnt_l3_l4 = joint(prefix, "jnt4", lnk3, lnk4)
    jnt_l4_l5 = joint(prefix, "jnt5", lnk4, lnk5)
    jnt_l5_l6 = joint(prefix, "jnt6", lnk5, lnk6)
    jnt_l6_l7 = joint(prefix, "jnt7", lnk6, lnk7)
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


def prepare_bdy_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    body_lnk = orbms.Link.from_file(
        os.path.join(mesh_dir, "body_link0_symp.stl"),
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.DEEP_GRAY,
        scale=(0.001, 0.001, 0.001))
    structure.add_lnk(body_lnk)
    structure.compile()
    return structure


class OALft(ormmb.ManipulatorBase):

    @classmethod
    def _build_structure(cls):
        return prepare_arm_ms("lft")

    def __init__(self, rotmat=None,
                 pos=None, is_free=True):
        super().__init__(
            rotmat=rotmat, pos=pos,
            is_free=is_free, data_dir="data_la")


class OARgt(ormmb.ManipulatorBase):

    @classmethod
    def _build_structure(cls):
        return prepare_arm_ms("rgt")

    def __init__(self, rotmat=None,
                 pos=None, is_free=True):
        super().__init__(
            rotmat=rotmat, pos=pos,
            is_free=is_free, data_dir="data_ra")


class OABody(orbmb.MechBase):
    """Single body link (openarm_body_link0)."""

    @classmethod
    def _build_structure(cls):
        return prepare_bdy_ms()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos, is_free=True)


class OpenArm:

    def __init__(self, rotmat=None, pos=None):
        self._rotmat = oum.ensure_rotmat(rotmat)
        self._pos = oum.ensure_pos(pos)
        self.body = OABody(rotmat=rotmat, pos=pos)
        self.lft_arm = OALft()
        self.rgt_arm = OARgt()
        self._left_tf = oum.tf_from_rotmat_pos(
            rotmat=oum.rotmat_from_euler(-1.5708, 0, 0), pos=(0.0, 0.031, 0.698))
        self._right_tf = oum.tf_from_rotmat_pos(
            rotmat=oum.rotmat_from_euler(1.5708, 0, 0), pos=(0.0, -0.031, 0.698))
        self.body.mount(self.lft_arm, self.body.runtime_lnks[-1], self._left_tf)
        self.body.mount(self.rgt_arm, self.body.runtime_lnks[-1], self._right_tf)
        self.body._update_mounting(self.body._mountings[self.lft_arm])
        self.body._update_mounting(self.body._mountings[self.rgt_arm])

    def attach_to(self, scene):
        self.body.attach_to(scene)
        # self.lft_arm.attach_to(scene)
        # self.rgt_arm.attach_to(scene)

    def detach_from(self, scene):
        self.body.detach_from(scene)
        # self.lft_arm.detach_from(scene)
        # self.rgt_arm.detach_from(scene)

    @property
    def rotmat(self):
        return self.body.rotmat

    @property
    def pos(self):
        return self.body.pos

    def set_rotmat_pos(self, rotmat=None, pos=None):
        self.body.set_rotmat_pos(rotmat=rotmat, pos=pos)
        self.body._update_mounting(self.body._mountings[self.lft_arm])
        self.body._update_mounting(self.body._mountings[self.rgt_arm])
