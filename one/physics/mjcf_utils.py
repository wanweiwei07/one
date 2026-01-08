import numpy as np
import one.utils.math as rm
import one.utils.constant as const
import one.scene.collision as sco

model_template = """<mujoco>
  <option gravity="{gx} {gy} {gz}"/>
  <option timestep="{timestep}"/>
  <default>
    <joint damping="5" armature="0.05" frictionloss="0.7"/>
    <geom friction="1 0.1 0.1"/>
  </default>
  <asset>
{assets}
  </asset>
  <worldbody>
{bodies}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
  <compiler angle="radian" inertiafromgeom="true"/>
</mujoco>
"""

body_template = """<body name="{name}" pos="{px} {py} {pz}" quat="{qw} {qx} {qy} {qz}">
{body_inner}
</body>"""


def indent(text, n=0):
    pad = " " * n
    return "\n".join(pad + line if line else "" for line in text.splitlines())


def join_nonempty(lines, n_indent=0):
    return "\n".join(" " * n_indent + line
                     for line in lines
                     if line and line.strip())


def inertial_to_xml(mass, com, inrtmat):
    """all arguments required"""
    if (mass is None) or (com is None) or (inrtmat is None):
        return ""
    return (f'<inertial mass="{mass}" '
            f'pos="{com[0]} {com[1]} {com[2]}" '
            f'fullinertia="{inrtmat[0][0]} {inrtmat[1][1]} {inrtmat[2][2]} '
            f'{inrtmat[0][1]} {inrtmat[0][2]} {inrtmat[1][2]}"/>')


def collision_to_xml(c, mesh_assets, namer, mass=None):
    mass_attr = f'mass="{mass}" ' if mass is not None else ""
    pos = c.pos
    qx, qy, qz, qw = c.quat
    common = f'{mass_attr}pos="{pos[0]} {pos[1]} {pos[2]}" quat="{qw} {qx} {qy} {qz}"'
    asset_xml = None
    if isinstance(c, sco.SphereCollisionShape):
        return f'<geom type="sphere" size="{c.radius}" {common}/>', None
    elif isinstance(c, sco.CapsuleCollisionShape):
        r, l = c.radius, c.half_length
        return f'<geom type="capsule" size="{r} {l}" {common}/>', None
    elif isinstance(c, (sco.AABBCollisionShape, sco.OBBCollisionShape)):
        sx, sy, sz = c.half_extents
        return f'<geom type="box" size="{sx} {sy} {sz}" {common}/>', None
    elif isinstance(c, sco.PlaneCollisionShape):
        return f'<geom type="plane" size="1 1 0.1" {common}/>', None
    elif isinstance(c, sco.MeshCollisionShape):
        if c.file_path not in mesh_assets:
            mesh_name = namer.unique_name("mesh", "mesh")
            mesh_assets[c.file_path] = mesh_name
            asset_xml = f'<mesh name="{mesh_name}" file="{c.file_path}"/>'
        else:
            mesh_name = mesh_assets[c.file_path]
        return f'<geom type="mesh" mesh="{mesh_name}" {common}/>', asset_xml
    else:
        raise NotImplementedError


def sobj_to_mjcf(sobj, mesh_assets, namer):
    inertial_xml = inertial_to_xml(sobj.mass, sobj.com, sobj.inrtmat)
    geoms = []
    assets = []
    first_geom_mass = None if inertial_xml else sobj.mass
    for idx, c in enumerate(getattr(sobj, "collisions", [])):
        mass = first_geom_mass if idx == 0 else None
        geom_xml, asset_xml = collision_to_xml(c, mesh_assets,
                                               namer, mass=mass)
        geoms.append(geom_xml)
        if asset_xml:
            assets.append(asset_xml)
    return inertial_xml, geoms, assets


def sobj_to_mjcf_body(sobj, mesh_assets, namer):
    px, py, pz = sobj.pos
    qx, qy, qz, qw = sobj.quat
    inertial_xml, geoms, assets = sobj_to_mjcf(sobj, mesh_assets, namer)
    joint_xml = "" if not sobj.is_free else "<freejoint/>"
    body_inner = join_nonempty([joint_xml, inertial_xml, "\n".join(geoms)])
    body_name = namer.unique_name("body", sobj.name)
    namer.reg_bdy(sobj, body_name)
    body_xml = body_template.format(name=body_name, px=px, py=py, pz=pz,
                                    qw=qw, qx=qx, qy=qy, qz=qz,
                                    body_inner=indent(body_inner, n=2))
    return assets, body_xml


#
def state_to_mjcf_body(state, mesh_assets, namer):
    compiled = state._compiled
    if compiled is None:
        raise RuntimeError("structure must be compiled before exporting to MJCF")
    assets = []
    actuators = []

    def _build_child_body_with_joint(jidx, lidx, level, attach_to_parent_body=False):
        lnk = state.runtime_lnks[lidx]
        # joint origin in parent link frame
        tfmat = compiled.jotfmat_by_idx[jidx]
        rotmat = tfmat[:3, :3]
        px, py, pz = tfmat[:3, 3]
        qx, qy, qz, qw = rm.quat_from_rotmat(rotmat)
        # joint xml
        jnt_type = compiled.jtypes_by_idx[jidx]
        if jnt_type == const.JntType.REVOLUTE:
            jtype_str = "hinge"
        elif jnt_type == const.JntType.PRISMATIC:
            jtype_str = "slide"
        else:
            jtype_str = "fixed"
        axis = compiled.jax_by_idx[jidx]
        range_low = compiled.jlmt_low_by_idx[jidx]
        range_high = compiled.jlmt_high_by_idx[jidx]
        jnt_name = namer.unique_name("joint", f"jnt{jidx}_lnk{lidx}")
        namer.reg_jnt((state, jidx), jnt_name)  # jnt shared, so (state, jidx) for key
        joint_xml = (f'<joint name="{jnt_name}" type="{jtype_str}" '
                     f'axis="{axis[0]} {axis[1]} {axis[2]}" '
                     f'range="{range_low} {range_high}"/>')
        if jtype_str == "hinge" or jtype_str == "slide":
            act_name = namer.unique_name("act", f"act{jidx}")
            actuator_xml = f'<position name="{act_name}" joint="{jnt_name}" kp="500"/>'
            actuators.append(actuator_xml)
        inertial_xml, geoms, new_assets = sobj_to_mjcf(lnk,
                                                       mesh_assets,
                                                       namer)
        assets.extend(new_assets)
        is_childbody = (len(geoms) == 0)
        # build children (grand-children links)
        child_bodies = []
        for child_lidx in compiled.clnk_ids_of_lidx[lidx]:
            pjidx_of_clidx = compiled.pjidx_of_lidx[child_lidx]
            if pjidx_of_clidx < 0:
                continue
            child_bodies.append(
                _build_child_body_with_joint(pjidx_of_clidx,
                                             child_lidx,
                                             level + 1,
                                             is_childbody))
        if geoms and not attach_to_parent_body:
            body_inner = join_nonempty([joint_xml, inertial_xml,
                                        "\n".join(geoms),
                                        "\n".join(child_bodies)])
            body_name = namer.unique_name("body", lnk.name)
            namer.reg_bdy(lnk, body_name)
            body_xml = body_template.format(name=body_name,
                                            px=px, py=py, pz=pz,
                                            qw=qw, qx=qx, qy=qy, qz=qz,
                                            body_inner=indent(body_inner, n=2))
            return body_xml
        elif geoms and attach_to_parent_body:
            return join_nonempty([joint_xml, inertial_xml,
                                  "\n".join(geoms),
                                  "\n".join(child_bodies)])
        else:
            return join_nonempty([joint_xml, "\n".join(child_bodies)])

    def _build_root_body():
        root_lnk_idx = compiled.root_lnk_idx
        root_lnk = state.runtime_lnks[root_lnk_idx]
        px, py, pz = root_lnk.pos
        qx, qy, qz, qw = rm.quat_from_rotmat( root_lnk.rotmat)
        inertial_xml, geoms, root_assets = sobj_to_mjcf(root_lnk,
                                                        mesh_assets,
                                                        namer)
        assets.extend(root_assets)
        root_joint_xml = "" if not root_lnk.is_free else "<freejoint/>"
        child_bodies = []
        for clnk_idx in compiled.clnk_ids_of_lidx[root_lnk_idx]:
            pjidx_of_clidx = compiled.pjidx_of_lidx[clnk_idx]
            if pjidx_of_clidx < 0:
                continue
            child_bodies.append(
                _build_child_body_with_joint(pjidx_of_clidx,
                                             clnk_idx, level=1))
        body_inner = join_nonempty(
            [root_joint_xml, inertial_xml,
             "\n".join(geoms), "\n".join(child_bodies)])
        body_name = namer.unique_name("body", root_lnk.name)
        namer.reg_bdy(root_lnk, body_name)
        body_xml = body_template.format(name=body_name,
                                        px=px, py=py, pz=pz,
                                        qw=qw, qx=qx, qy=qy, qz=qz,
                                        body_inner=indent(body_inner, n=2))
        return body_xml

    root_body = _build_root_body()
    return assets, actuators, root_body


# def state_to_mjcf_body(state, mesh_assets, namer):
#     compiled = state._compiled
#     if compiled is None:
#         raise RuntimeError("structure must be compiled before exporting to MJCF")
#     assets = []
#     actuators = []
#
#     def build_subtree(lidx, tfmat_acc, parent_is_entity):
#         lnk = state.runtime_lnks[lidx]
#         pjidx = compiled.pjidx_of_lidx[lidx]
#         # parent joint transform（joint->parent_link）
#         if pjidx >= 0:
#             tfmat_jnt = compiled.jotfmat_by_idx[pjidx]
#             tfmat_here = tfmat_acc @ tfmat_jnt
#         else:
#             tfmat_here = tfmat_acc
#         is_entity = (lnk.collision_type is not None)
#         joint_xmls = []
#         if pjidx >= 0:  # TODO is it necessary?
#             jtype = compiled.jtypes_by_idx[pjidx]
#             if jtype == const.JntType.REVOLUTE:
#                 jtype_str = "hinge"
#             elif jtype == const.JntType.PRISMATIC:
#                 jtype_str = "slide"
#             else:
#                 jtype_str = "fixed"
#             axis = compiled.jax_by_idx[pjidx]
#             jname = namer.unique_name("joint", f"j{pjidx}_l{lidx}")
#             namer.reg_jnt((state, pjidx), jname)
#             joint_xmls.append(f'<joint name="{jname}" type="{jtype_str}" '
#                               f'axis="{axis[0]} {axis[1]} {axis[2]}"/>')
#             if jtype_str in ("hinge", "slide"):
#                 an = namer.unique_name("act", f"act{pjidx}")
#                 actuators.append(f'<position name="{an}" joint="{jname}" kp="500"/>')
#         inertial_xml, geoms, new_assets = sobj_to_mjcf(lnk,
#                                                        mesh_assets,
#                                                        namer)
#         assets.extend(new_assets)
#         self_inline_xml = join_nonempty(["\n".join(joint_xmls),
#                                          inertial_xml,
#                                          "\n".join(geoms)])
#         inline_xmls = [self_inline_xml]
#         if not parent_is_entity:
#             for clidx in compiled.clnk_ids_of_lidx[lidx]:
#                 inline_xmls.append(build_subtree(clidx, tfmat_here,
#                                                  parent_is_entity))
#             return "\n".join(inline_xmls)
#         if parent_is_entity and not is_entity:
#             for clidx in compiled.clnk_ids_of_lidx[lidx]:
#                 inline_xmls.append(build_subtree(clidx, tfmat_here,
#                                                  parent_is_entity))
#             return "\n".join(inline_xmls)
#         # children
#         for clidx in compiled.clnk_ids_of_lidx[lidx]:
#             inline_xmls.append(build_subtree(clidx, tfmat_here, parent_is_entity))
#         pos, quat = rm.pos_quat_from_tfmat(tfmat_here)
#         qx, qy, qz, qw = quat
#         body_inner = "\n".join(inline_xmls)
#         bname = namer.unique_name("body", lnk.name)
#         namer.reg_bdy(lnk, bname)
#         return body_template.format(name=bname,
#                                     px=pos[0], py=pos[1], pz=pos[2],
#                                     qw=qw, qx=qx, qy=qy, qz=qz,
#                                     body_inner=indent(body_inner, 2))
#
#     root_idx = compiled.root_lnk_idx
#     root_lnk = state.runtime_lnks[root_idx]
#     root_tfmat = np.eye(4)
#     root_tfmat[:3, :3] = root_lnk.rotmat
#     root_tfmat[:3, 3] = root_lnk.pos
#     child_xmls = []
#     parent_is_entity = (root_lnk.collision_type is not None)
#     for cl in compiled.clnk_ids_of_lidx[root_idx]:
#         child_xmls.append(build_subtree(cl, root_tfmat, parent_is_entity))
#     inertial_xml, geoms, root_assets = sobj_to_mjcf(root_lnk, mesh_assets, namer)
#     assets.extend(root_assets)
#     root_joint_xml = "" if root_lnk.is_fixed else "<freejoint/>"
#     px, py, pz = root_lnk.pos
#     qx, qy, qz, qw = rm.quat_from_rotmat(root_lnk.rotmat)
#     body_inner = join_nonempty([root_joint_xml,
#                                 inertial_xml,
#                                 "\n".join(geoms),
#                                 "\n".join(child_xmls)])
#     bname = namer.unique_name("body", root_lnk.name)
#     namer.reg_bdy(root_lnk, bname)
#     root_body = body_template.format(name=bname,
#                                      px=px, py=py, pz=pz,
#                                      qw=qw, qx=qx, qy=qy, qz=qz,
#                                      body_inner=indent(body_inner, 2))
#     return assets, actuators, root_body