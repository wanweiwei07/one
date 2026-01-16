import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.render_model as osrm
import one.scene.geometry_primitive as osgp


# kwargs in the functions are defined as in _parse_phys
def _parse_phys(kwargs):
    return (kwargs.get("inertia", None),
            kwargs.get("com", None),
            kwargs.get("mass", None),
            kwargs.get("collision_type", None),
            kwargs.get("is_free", False))


def gen_cylinder(spos=(0, 0, 0),
                 epos=(0.01, 0.01, 0.01),
                 radius=0.05, segments=8,
                 rgb=ouc.BasicColor.DEFAULT,
                 alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = oum.unit_vec(epos - spos,
                                   return_length=True)
    geometry = osgp.gen_cylinder_geom(length, radius, segments)
    rotmat = oum.rotmat_between_vecs(ouc.StandardAxis.Z, dir_vec)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    o.set_rotmat_pos(rotmat=rotmat, pos=spos)
    amc = False if collision_type is None else True
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rgb=rgb, alpha=alpha),
        auto_make_collision=amc)
    o.set_inertia(inertia, com, mass)
    return o


def gen_cone(spos=(0, 0, 0),
             epos=(0.01, 0.01, 0.01),
             radius=0.05, segments=8,
             rgb=ouc.BasicColor.DEFAULT,
             alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = oum.unit_vec(epos - spos,
                                   return_length=True)
    geometry = osgp.gen_cone_geom(
        length, radius, segments)
    rotmat = oum.rotmat_between_vecs(
        ouc.StandardAxis.Z, dir_vec)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    o.set_rotmat_pos(rotmat=rotmat, pos=spos)
    amc = False if collision_type is None else True
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rgb=rgb, alpha=alpha),
        auto_make_collision=amc)
    o.set_inertia(inertia, com, mass)
    return o


def gen_sphere(pos=(0, 0, 0),
               radius=0.05, segments=8,
               rgb=ouc.BasicColor.DEFAULT,
               alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    geometry = osgp.gen_sphere_geom(radius, segments)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    o.pos = pos
    amc = False if collision_type is None else True
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rgb=rgb, alpha=alpha),
        auto_make_collision=amc)
    o.set_inertia(inertia, com, mass)
    return o


def gen_icosphere(pos=(0, 0, 0),
                  radius=0.05, subdivisions=2,
                  rgb=ouc.BasicColor.DEFAULT,
                  alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    geometry = osgp.gen_icosphere_geom(radius, subdivisions)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    o.pos = pos
    amc = False if collision_type is None else True
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rgb=rgb, alpha=alpha),
        auto_make_collision=amc)
    o.set_inertia(inertia, com, mass)
    return o


def gen_box(pos=(0, 0, 0),
            half_extents=(0.05, 0.05, 0.05),
            rotmat=None, rgb=ouc.BasicColor.DEFAULT,
            alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    half_extents = np.asarray(half_extents, np.float32)
    geometry = osgp.gen_box_geom(half_extents)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    o.set_rotmat_pos(rotmat=rotmat, pos=pos)
    amc = False if collision_type is None else True
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rgb=rgb, alpha=alpha),
        auto_make_collision=amc)
    o.set_inertia(inertia, com, mass)
    return o


def gen_arrow(spos=np.zeros(3), epos=np.ones(3) * 0.01,
              shaft_radius=ouc.ArrowSize.SHAFT_RADIUS,
              head_radius=ouc.ArrowSize.HEAD_RADIUS,
              head_length=ouc.ArrowSize.HEAD_LENGTH,
              segments=8, rgb=ouc.BasicColor.DEFAULT,
              alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    # if is_free:
    #     print("Warning: frame is usually not free. Setting to False.")
    #     is_free = False
    is_free = False
    # collider must be ignored for arrow
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = oum.unit_vec(epos - spos,
                                   return_length=True)
    geometry = osgp.gen_arrow_geom(length,
                                   shaft_radius,
                                   head_radius,
                                   head_length,
                                   segments)
    rotmat = oum.rotmat_between_vecs(ouc.StandardAxis.Z, dir_vec)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    o.set_rotmat_pos(rotmat=rotmat, pos=spos)
    amc = False if collision_type is None else True
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rgb=rgb, alpha=alpha),
        auto_make_collision=amc)
    o.set_inertia(inertia, com, mass)
    return o


def gen_frame(pos=np.zeros(3), rotmat=np.eye(3),
              length_scale=1.0, radius_scale=1.0,
              segments=8, color_mat=ouc.CoordColor.RGB,
              alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    # if is_free:
    #     print("Warning: frame is usually not free. Setting to False.")
    #     is_free = False
    is_free = False
    # collider must be ignored for frame
    arrow_length = ouc.StandardAxis.ARROW_LENGTH * length_scale
    shaft_radius = ouc.StandardAxis.ARROW_SHAFT_RADIUS * radius_scale
    head_length = ouc.StandardAxis.ARROW_HEAD_LENGTH * radius_scale
    head_radius = ouc.StandardAxis.ARROW_HEAD_RADIUS * radius_scale
    geometry = osgp.gen_arrow_geom(arrow_length,
                                   shaft_radius,
                                   head_radius,
                                   head_length,
                                   segments)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    o.set_rotmat_pos(rotmat=rotmat, pos=pos)
    amc = False if collision_type is None else True
    # x-axis
    loc_rotmat = oum.rotmat_between_vecs(ouc.StandardAxis.Z, ouc.StandardAxis.X)
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rotmat=loc_rotmat,
        rgb=color_mat[:, 0], alpha=alpha),
        auto_make_collision=amc)
    # y-axis
    loc_rotmat = oum.rotmat_between_vecs(ouc.StandardAxis.Z, ouc.StandardAxis.Y)
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rotmat=loc_rotmat,
        rgb=color_mat[:, 1], alpha=alpha),
        auto_make_collision=amc)
    # z-axis
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rotmat=np.eye(3, dtype=np.float32),
        rgb=color_mat[:, 2], alpha=alpha),
        auto_make_collision=amc)
    o.set_inertia(inertia, com, mass)
    return o


def gen_plane(pos=(0, 0, 0),
              normal=ouc.StandardAxis.Z,
              size=(100.0, 100.0),
              thickness=1e-3,
              rgb=ouc.BasicColor.GRAY, alpha=1.0):
    pos = np.asarray(pos, np.float32)
    size = np.asarray(size, np.float32)
    half_extents = np.array([size[0] / 2,
                             size[1] / 2,
                             thickness],
                            np.float32)
    geometry = osgp.gen_box_geom(half_extents)
    rotmat = oum.rotmat_between_vecs(
        ouc.StandardAxis.Z, normal)
    o = osso.SceneObject(
        collision_type=ouc.CollisionType.PLANE,
        is_free=False)
    o.set_rotmat_pos(rotmat=rotmat, pos=pos)
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rgb=rgb, alpha=alpha))
    return o


def gen_custom(verts, faces, rgb=ouc.BasicColor.RED, alpha=1.0):
    """
    Build a SceneObject from user-specified vertices/faces.
    verts: (N,3)
    faces: (M,3)
    """
    verts = np.asarray(verts, np.float32)
    faces = np.asarray(faces, np.uint32)
    geometry = (verts, faces)
    o = osso.SceneObject(collision_type=None, is_free=False)
    o.add_visual(osrm.RenderModel(
        geometry=geometry, rgb=rgb, alpha=alpha),
        auto_make_collision=False)
    return o
