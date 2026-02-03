import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.render_model_primitive as osrmp


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
    length, dir_vec = oum.unit_vec(
        epos - spos, return_length=True)
    rmodel = osrmp.gen_cylinder_rmodel(length=length, radius=radius,
                                       n_segs=segments,
                                       rgb=rgb, alpha=alpha)
    rotmat = oum.rotmat_between_vecs(ouc.StandardAxis.Z, dir_vec)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    amc = False if collision_type is None else True
    o.add_visual(rmodel, auto_make_collision=amc)
    o.set_rotmat_pos(rotmat=rotmat, pos=spos)
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
    length, dir_vec = oum.unit_vec(
        epos - spos, return_length=True)
    rmodel = osrmp.gen_cone_rmodel(length=length, radius=radius,
                                   n_segs=segments,
                                   rgb=rgb, alpha=alpha)
    rotmat = oum.rotmat_between_vecs(
        ouc.StandardAxis.Z, dir_vec)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    amc = False if collision_type is None else True
    o.add_visual(rmodel, auto_make_collision=amc)
    o.set_rotmat_pos(rotmat=rotmat, pos=spos)
    o.set_inertia(inertia, com, mass)
    return o


def gen_sphere(pos=(0, 0, 0),
               radius=0.05, segments=8,
               rgb=ouc.BasicColor.DEFAULT,
               alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    rmodel = osrmp.gen_sphere_rmodel(
        radius=radius, n_segs=segments,
        rgb=rgb, alpha=alpha)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    amc = False if collision_type is None else True
    o.add_visual(rmodel, auto_make_collision=amc)
    o.pos = pos
    o.set_inertia(inertia, com, mass)
    return o


def gen_icosphere(pos=(0, 0, 0),
                  radius=0.05, subdivisions=2,
                  rgb=ouc.BasicColor.DEFAULT,
                  alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    rmodel = osrmp.gen_icosphere_rmodel(
        radius=radius, n_subs=subdivisions,
        rgb=rgb, alpha=alpha)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    amc = False if collision_type is None else True
    o.add_visual(rmodel, auto_make_collision=amc)
    o.pos = pos
    o.set_inertia(inertia, com, mass)
    return o


def gen_box(pos=(0, 0, 0),
            half_extents=(0.05, 0.05, 0.05),
            rotmat=None, rgb=ouc.BasicColor.DEFAULT,
            alpha=1.0, **kwargs):
    _psd = _parse_phys(kwargs)
    inertia, com, mass, collision_type, is_free = _psd
    half_extents = np.asarray(half_extents, np.float32)
    rmodel = osrmp.gen_box_rmodel(half_extents=half_extents,
                                  rgb=rgb, alpha=alpha)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    amc = False if collision_type is None else True
    o.add_visual(rmodel, auto_make_collision=amc)
    o.set_rotmat_pos(rotmat=rotmat, pos=pos)
    o.set_inertia(inertia, com, mass)
    return o


def gen_linsegs(segs, radius=0.001, srgbs=None, alpha=1.0):
    """ segs: (N,2,3), srgb: None | scalar | (3,) | (N,3)
    returns: SceneObject with N cylinder segments  """
    segs = np.asarray(segs, dtype=np.float32)
    if segs.ndim != 3 or segs.shape[1:] != (2, 3):
        raise ValueError("segs must be (N,2,3)")
    n = segs.shape[0]
    if srgbs is None:
        srgbs = np.tile(np.array(
            ouc.BasicColor.BLACK,
            dtype=np.float32), (n, 1))
    elif srgbs.shape == (3,):
        srgbs = np.tile(srgbs, (n, 1))
    elif srgbs.shape == (n, 3):
        srgbs = np.asarray(srgbs, dtype=np.float32)
    else:
        raise ValueError("srgb must be scalar, (3,), or (N,3)")
    # build single SceneObject
    o = osso.SceneObject(
        collision_type=None, is_free=False)
    for i in range(n):
        a = segs[i, 0]
        b = segs[i, 1]
        if np.linalg.norm(b - a) < 1e-12:
            continue
        length, dir_vec = oum.unit_vec(
            b - a, return_length=True)
        rotmat = oum.rotmat_between_vecs(
            ouc.StandardAxis.Z, dir_vec)
        rmodel = osrmp.gen_cylinder_rmodel(
            length=length, radius=radius, rotmat=rotmat,
            pos=a, rgb=srgbs[i], alpha=alpha)
        o.add_visual(rmodel, auto_make_collision=False)
    return o


def gen_arrow(spos=np.zeros(3), epos=np.ones(3) * 0.01,
              shaft_radius=ouc.ArrowSize.SHAFT_RADIUS,
              head_radius=ouc.ArrowSize.HEAD_RADIUS,
              head_length=ouc.ArrowSize.HEAD_LENGTH,
              n_segs=8, rgb=ouc.BasicColor.DEFAULT,
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
    rmodel = osrmp.gen_arrow_rmodel(
        length, shaft_radius, head_length, head_radius,
        n_segs, rgb=rgb, alpha=alpha)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    amc = False if collision_type is None else True
    o.add_visual(rmodel, auto_make_collision=amc)
    rotmat = oum.rotmat_between_vecs(ouc.StandardAxis.Z, dir_vec)
    o.set_rotmat_pos(rotmat=rotmat, pos=spos)
    o.set_inertia(inertia, com, mass)
    return o


def gen_frame(pos=np.zeros(3), rotmat=np.eye(3),
              length_scale=1.0, radius_scale=1.0,
              n_segs=8, color_mat=ouc.CoordColor.RGB,
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
    rmodel_x = osrmp.gen_arrow_rmodel(
        arrow_length, shaft_radius, head_length,
        head_radius, n_segs,
        oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi / 2),
        rgb=color_mat[:, 0], alpha=alpha)
    rmodel_y = osrmp.gen_arrow_rmodel(
        arrow_length, shaft_radius, head_length,
        head_radius, n_segs,
        oum.rotmat_from_axangle(ouc.StandardAxis.X, -np.pi / 2),
        rgb=color_mat[:, 1], alpha=alpha)
    rmodel_z = osrmp.gen_arrow_rmodel(
        arrow_length, shaft_radius, head_length,
        head_radius, n_segs,
        rgb=color_mat[:, 2], alpha=alpha)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    amc = False if collision_type is None else True
    o.add_visual(rmodel_x, auto_make_collision=amc)
    o.add_visual(rmodel_y, auto_make_collision=amc)
    o.add_visual(rmodel_z, auto_make_collision=amc)
    o.set_rotmat_pos(rotmat=rotmat, pos=pos)
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
                             thickness / 2],
                            np.float32)
    rmodel = osrmp.gen_box_rmodel(half_extents=half_extents,
                                  rgb=rgb, alpha=alpha)
    o = osso.SceneObject(
        collision_type=ouc.CollisionType.PLANE,
        is_free=False)
    o.add_visual(rmodel)
    rotmat = oum.rotmat_between_vecs(
        ouc.StandardAxis.Z, normal)
    o.set_rotmat_pos(rotmat=rotmat, pos=pos)
    return o


def gen_pcd(vs, vrgbs, alpha=1.0):
    """
    Build a SceneObject from user-specified vertices/faces.
    verts: (N,3)
    faces: (M,3)
    """
    vs = np.asarray(vs, np.float32)
    rmodel = osrmp.gen_pcd_rmodel(vs, vrgbs, alpha)
    o = osso.SceneObject(collision_type=None, is_free=False)
    o.add_visual(rmodel, auto_make_collision=False)
    return o


def gen_mesh(vs, fs, collision_type=None,
             is_free=False, rgb=ouc.BasicColor.DEFAULT,
             alpha=1.0, **kwargs):
    """
    Build a SceneObject from user-specified vertices/faces.
    vs: (N,3), fs: (M,3)
    """
    _psd = _parse_phys(kwargs)
    inertia, com, mass, _, _ = _psd
    vs = np.asarray(vs, np.float32)
    fs = np.asarray(fs, np.uint32)
    rmodel = osrmp.gen_mesh_rmodel(
        vs=vs, fs=fs, rgb=rgb, alpha=alpha)
    o = osso.SceneObject(collision_type=collision_type,
                         is_free=is_free)
    amc = False if collision_type is None else True
    o.add_visual(rmodel, auto_make_collision=amc)
    o.set_inertia(inertia, com, mass)
    return o
