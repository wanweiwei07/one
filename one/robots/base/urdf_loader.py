import os
import re
import xacro
from urdf_parser_py.urdf import URDF
from xacro import substitution_args

def auto_scan_packages(main_file, search_root):
    """
    Scan main_file and auto-detect "$(find pkg)" patterns,
    then search under search_root for matching folders.
    Returns: dict pkg_name → path
    """
    # Read xacro text
    with open(main_file, "r") as f:
        text = f.read()
    # Regex: match $(find xxx)
    pkg_names = set(re.findall(r"\$\(find\s+([^\)]+)\)", text))
    mapping = {}
    # Search directory recursively
    for root, dirs, files in os.walk(search_root):
        for pkg in pkg_names:
            if pkg in root:
                mapping[pkg] = root.replace("\\", "/")
                return mapping

def load_robot_from_xacro(main_file, base_dir=None, mappings=None):
    """
    Load a robot URDF from a xacro or urdf file.
    :param main_file: path to .xacro or .urdf
    :param paramers: paramers mappings, e.g. {"model": "cobotta"}
    """

    # absolute path
    main_file = os.path.abspath(main_file)
    if base_dir is None:
        raise RuntimeError("Package base_dir must be specified for loading xacro files.")
    known_packages = auto_scan_packages(main_file, base_dir)
    # local resolver for $(find ...)
    def eval_find(pkg_name):
        if pkg_name in known_packages:
            return known_packages[pkg_name]
        if ("control" in pkg_name
                or "gazebo" in pkg_name
                or pkg_name.endswith("_moveit_config")):
            return ""
        raise RuntimeError(f"Unknown ROS package in $(find): {pkg_name}")

    # override xacro's find
    substitution_args._eval_find = eval_find
    if main_file.endswith(".xacro"):
        doc = xacro.process_file(main_file, mappings=mappings or {})
        xml = doc.toxml()
    else:
        xml = open(main_file).read()
    return URDF.from_xml_string(xml)


def urdf_to_mechstruct(urdf, urdf_dir, collision_type=None,
                       res_dir=None, mesh_resolver=None):
    """Convert a urdf_parser_py URDF (Robot) object into a one MechStruct.

    :param urdf: urdf_parser_py.urdf.URDF object (from load_robot_from_xacro).
    :param urdf_dir: directory of the urdf file, used by the default mesh
        resolver to resolve relative mesh paths.
    :param collision_type: ouc.CollisionType for link meshes.
    :param res_dir: overrides the structure's resource dir (where solver data
        is cached); defaults to the caller-inferred dir.
    :param mesh_resolver: optional callable(filename) -> abs path; the default
        resolves relative to urdf_dir and rejects package:// urls.
    """
    import numpy as np
    import one.utils.constant as ouc
    import one.utils.math as oum
    import one.robots.base.mech_structure as orbms

    if mesh_resolver is None:
        def mesh_resolver(filename):
            if filename.startswith("package://"):
                raise ValueError(
                    f"package:// meshes are not supported by the default "
                    f"resolver: {filename}")
            return os.path.abspath(os.path.join(urdf_dir, filename))

    jtype_map = {
        "fixed": ouc.JntType.FIXED,
        "revolute": ouc.JntType.REVOLUTE,
        "continuous": ouc.JntType.REVOLUTE,
        "prismatic": ouc.JntType.PRISMATIC,
    }

    def _pose(origin):
        if origin is None:
            return None, None
        xyz = np.asarray(origin.xyz if origin.xyz is not None else (0.0, 0.0, 0.0),
                         dtype=np.float32)
        rpy = origin.rpy if origin.rpy is not None else (0.0, 0.0, 0.0)
        rotmat = oum.rotmat_from_euler(rpy[0], rpy[1], rpy[2], order="sxyz")
        return rotmat, xyz

    structure = orbms.MechStruct()
    lnk_map = {}
    for ul in urdf.links:
        visuals = (ul.visuals if getattr(ul, "visuals", None)
                   else ([ul.visual] if getattr(ul, "visual", None) else []))
        mesh_vis = next(
            (v for v in visuals
             if getattr(getattr(v, "geometry", None), "filename", None)), None)
        if mesh_vis is not None:
            loc_rotmat, loc_pos = _pose(mesh_vis.origin)
            rgb, alpha = None, 1.0
            if mesh_vis.material is not None and mesh_vis.material.color is not None:
                rgba = mesh_vis.material.color.rgba
                rgb, alpha = rgba[:3], float(rgba[3])
            lnk = orbms.Link.from_file(
                mesh_resolver(mesh_vis.geometry.filename),
                loc_rotmat=loc_rotmat, loc_pos=loc_pos,
                collision_type=collision_type, rgb=rgb, alpha=alpha)
        else:
            lnk = orbms.Link(collision_type=None)
        lnk.name = ul.name
        if ul.inertial is not None:
            _, com = _pose(ul.inertial.origin)
            mass = None if ul.inertial.mass is None else float(ul.inertial.mass)
            inrtmat = None
            ip = ul.inertial.inertia
            if ip is not None:
                inrtmat = np.array(
                    [[ip.ixx, ip.ixy, ip.ixz],
                     [ip.ixy, ip.iyy, ip.iyz],
                     [ip.ixz, ip.iyz, ip.izz]], dtype=np.float32)
            lnk.set_inertia(inrtmat=inrtmat, com=com, mass=mass)
        lnk_map[ul.name] = lnk
        structure.add_lnk(lnk)

    jnt_map = {}
    for uj in urdf.joints:
        jtype = jtype_map.get(uj.type)
        if jtype is None:
            raise ValueError(f"Unsupported joint type: {uj.type}")
        rotmat, pos = _pose(uj.origin)
        axis = np.asarray(uj.axis if uj.axis is not None else (0.0, 0.0, 1.0),
                          dtype=np.float32)
        lmt_lo = lmt_up = None
        if uj.limit is not None:
            lmt_lo = None if uj.limit.lower is None else float(uj.limit.lower)
            lmt_up = None if uj.limit.upper is None else float(uj.limit.upper)
        jnt = orbms.Joint(
            jnt_type=jtype,
            parent_lnk=lnk_map[uj.parent],
            child_lnk=lnk_map[uj.child],
            axis=axis, rotmat=rotmat, pos=pos,
            lmt_lo=lmt_lo, lmt_up=lmt_up)
        jnt.name = uj.name
        jnt_map[uj.name] = jnt
        structure.add_jnt(jnt)

    # second pass: wire up <mimic> couplings (source joints now all exist) so
    # mimic joints follow their source (q = mult*q_src + offset) and drop out of
    # the active dof -- e.g. the O6 hand's dip/ip joints coupled to mcp/pitch.
    for uj in urdf.joints:
        mimic = getattr(uj, "mimic", None)
        if mimic is not None and getattr(mimic, "joint", None):
            mult = 1.0 if mimic.multiplier is None else float(mimic.multiplier)
            offset = 0.0 if mimic.offset is None else float(mimic.offset)
            jnt_map[uj.name].mmc = (jnt_map[mimic.joint], mult, offset)

    structure.lnk_map = lnk_map
    structure.jnt_map = jnt_map
    if res_dir is not None:
        structure.res_dir = res_dir
    structure.compile()
    return structure
