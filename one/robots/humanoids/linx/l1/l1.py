import os
import xml.etree.ElementTree as et

import numpy as np

import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_base as orbmb
import one.robots.base.mech_structure as orbms


_H0602_URDF = os.path.join(
    os.path.dirname(__file__),
    'urdf',
    'h0602.urdf',
)


def _parse_vec(text, default=None):
    if text is None:
        if default is None:
            default = (0.0, 0.0, 0.0)
        return np.asarray(default, dtype=np.float32)
    return np.asarray([float(v) for v in text.split()], dtype=np.float32)


def _parse_origin(node):
    if node is None:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    pos = _parse_vec(node.attrib.get('xyz'))
    rpy = _parse_vec(node.attrib.get('rpy'))
    rotmat = oum.rotmat_from_euler(rpy[0], rpy[1], rpy[2], order='sxyz')
    return rotmat, pos


def _parse_rgb(link_node):
    color_node = link_node.find('./visual/material/color')
    if color_node is None:
        return None, 1.0
    rgba = _parse_vec(color_node.attrib.get('rgba'), default=(1.0, 1.0, 1.0, 1.0))
    return rgba[:3], float(rgba[3])


def _resolve_mesh_path(urdf_dir, filename):
    if filename.startswith('package://'):
        raise ValueError(f'package:// meshes are not supported yet: {filename}')
    return os.path.abspath(os.path.join(urdf_dir, filename))


def _make_link(link_node, urdf_dir, collision_type):
    name = link_node.attrib['name']
    visual_node = link_node.find('./visual')
    mesh_node = link_node.find('./visual/geometry/mesh')
    rgb, alpha = _parse_rgb(link_node)

    if visual_node is not None and mesh_node is not None:
        loc_rotmat, loc_pos = _parse_origin(visual_node.find('origin'))
        mesh_path = _resolve_mesh_path(urdf_dir, mesh_node.attrib['filename'])
        lnk = orbms.Link.from_file(
            mesh_path,
            loc_rotmat=loc_rotmat,
            loc_pos=loc_pos,
            collision_type=collision_type,
            rgb=rgb,
            alpha=alpha,
        )
    else:
        lnk = orbms.Link(collision_type=None)

    lnk.name = name
    inertial_node = link_node.find('./inertial')
    if inertial_node is not None:
        mass_node = inertial_node.find('mass')
        inertia_node = inertial_node.find('inertia')
        _, com = _parse_origin(inertial_node.find('origin'))
        mass = None if mass_node is None else float(mass_node.attrib['value'])
        inrtmat = None
        if inertia_node is not None:
            i = inertia_node.attrib
            inrtmat = np.array([
                [float(i['ixx']), float(i['ixy']), float(i['ixz'])],
                [float(i['ixy']), float(i['iyy']), float(i['iyz'])],
                [float(i['ixz']), float(i['iyz']), float(i['izz'])],
            ], dtype=np.float32)
        lnk.set_inertia(inrtmat=inrtmat, com=com, mass=mass)
    return lnk


def _make_joint(joint_node, lnk_map):
    jnt_type_text = joint_node.attrib['type']
    if jnt_type_text == 'fixed':
        jnt_type = ouc.JntType.FIXED
    elif jnt_type_text in ('revolute', 'continuous'):
        jnt_type = ouc.JntType.REVOLUTE
    elif jnt_type_text == 'prismatic':
        jnt_type = ouc.JntType.PRISMATIC
    else:
        raise ValueError(f'Unsupported joint type: {jnt_type_text}')

    parent_name = joint_node.find('parent').attrib['link']
    child_name = joint_node.find('child').attrib['link']
    rotmat, pos = _parse_origin(joint_node.find('origin'))
    axis_node = joint_node.find('axis')
    axis = _parse_vec(None if axis_node is None else axis_node.attrib.get('xyz'),
                      default=(0.0, 0.0, 1.0))

    limit_node = joint_node.find('limit')
    lmt_lo = None
    lmt_up = None
    if limit_node is not None:
        if 'lower' in limit_node.attrib:
            lmt_lo = float(limit_node.attrib['lower'])
        if 'upper' in limit_node.attrib:
            lmt_up = float(limit_node.attrib['upper'])

    jnt = orbms.Joint(
        jnt_type=jnt_type,
        parent_lnk=lnk_map[parent_name],
        child_lnk=lnk_map[child_name],
        axis=axis,
        rotmat=rotmat,
        pos=pos,
        lmt_lo=lmt_lo,
        lmt_up=lmt_up,
    )
    jnt.name = joint_node.attrib['name']
    return jnt


def prepare_mechstruct(collision_type=ouc.CollisionType.MESH):
    """Load the Linx L1/H0602 upper-body URDF into a MechStruct."""
    urdf_path = os.path.abspath(_H0602_URDF)
    urdf_dir = os.path.dirname(urdf_path)
    root = et.parse(urdf_path).getroot()

    structure = orbms.MechStruct()
    structure.res_dir = os.path.dirname(__file__)
    structure.default_mesh_dir = os.path.abspath(
        os.path.join(urdf_dir, '..', 'meshes'))

    lnk_map = {}
    for link_node in root.findall('link'):
        lnk = _make_link(link_node, urdf_dir, collision_type)
        lnk_map[lnk.name] = lnk
        structure.add_lnk(lnk)

    jnt_map = {}
    for joint_node in root.findall('joint'):
        jnt = _make_joint(joint_node, lnk_map)
        jnt_map[jnt.name] = jnt
        structure.add_jnt(jnt)

    structure.lnk_map = lnk_map
    structure.jnt_map = jnt_map
    structure.compile()
    return structure


class L1(orbmb.MechBase):
    """Linx L1 upper-body humanoid model based on the H0602 URDF."""

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None, home_qs=None, is_free=True):
        super().__init__(rotmat=rotmat, pos=pos,
                         home_qs=home_qs, is_free=is_free)

    def lnk(self, name):
        lidx = self.structure.compiled.lidx_map[self.structure.lnk_map[name]]
        return self.runtime_lnks[lidx]

    def jnt(self, name):
        return self.structure.jnt_map[name]

    def chain(self, base_lnk_name, tip_lnk_name):
        return self.structure.get_chain(
            self.structure.lnk_map[base_lnk_name],
            self.structure.lnk_map[tip_lnk_name],
        )

    @property
    def left_arm_chain(self):
        return self.chain('waist_link2', 'left_arm_link_6')

    @property
    def right_arm_chain(self):
        return self.chain('waist_link2', 'right_arm_link_6')

    @property
    def left_arm_waist_chain(self):
        return self.chain('waist_link1', 'left_arm_link_6')

    @property
    def right_arm_waist_chain(self):
        return self.chain('waist_link1', 'right_arm_link_6')

    @property
    def neck_chain(self):
        return self.chain('waist_link2', 'neck_link2')


if __name__ == '__main__':
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(2.2, 1.4, 1.6),
                     cam_lookat_pos=(0.0, 0.0, 0.9))
    robot = L1()
    robot.attach_to(base.scene)
    loc_tcp_tf = np.eye(4, dtype=np.float32)
    loc_tcp_tf[:3, 3] = np.array([0.04, 0.0, 0.12], dtype=np.float32)
    left_tcp = ossop.frame_from_tf(
        loc_tcp_tf,
        length_scale=0.18,
        radius_scale=0.6,
    )
    right_tcp = ossop.frame_from_tf(
        loc_tcp_tf,
        length_scale=0.18,
        radius_scale=0.6,
    )
    left_tcp.attach_to(robot.lnk('left_arm_link_6'))
    right_tcp.attach_to(robot.lnk('right_arm_link_6'))
    base.run()
