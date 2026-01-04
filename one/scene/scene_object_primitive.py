import numpy as np
import one.scene.scene_object as sob
import one.scene.render_model as mdl
import one.scene.geometry_primitive as gprim
import one.utils.constant as const
import one.utils.math as rm

# kwargs in the functions are defined as in _parse_phys
def _parse_phys(kwargs):
    return (kwargs.get("inertia", None),
            kwargs.get("com", None),
            kwargs.get("mass", None),
            kwargs.get("collision_type", None),
            kwargs.get("is_fixed", True))


def gen_cylinder(spos=(0, 0, 0), epos=(0.01, 0.01, 0.01),
                 radius=0.05, segments=8,
                 rgb=const.BasicColor.DEFAULT, alpha=1.0, **kwargs):
    inertia, com, mass, collision_type, is_fixed = _parse_phys(kwargs)
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.unit_vec(epos - spos, return_length=True)
    geometry = gprim.gen_cylinder_geom(length, radius, segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    o = sob.SceneObject(rotmat=rotmat, pos=spos,
                        collision_type=collision_type,
                        is_fixed=is_fixed)
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    o.set_inertia(inertia, com, mass)
    return o


def gen_cone(spos=(0, 0, 0), epos=(0.01, 0.01, 0.01),
             radius=0.05, segments=8,
             rgb=const.BasicColor.DEFAULT, alpha=1.0, **kwargs):
    inertia, com, mass, collision_type, is_fixed = _parse_phys(kwargs)
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.unit_vec(epos - spos, return_length=True)
    geometry = gprim.gen_cone_geom(length, radius, segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    o = sob.SceneObject(rotmat=rotmat, pos=spos,
                        collision_type=collision_type,
                        is_fixed=is_fixed)
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    o.set_inertia(inertia, com, mass)
    return o


def gen_sphere(pos=(0, 0, 0), radius=0.05, segments=8,
               rgb=const.BasicColor.DEFAULT, alpha=1.0, **kwargs):
    inertia, com, mass, collision_type, is_fixed = _parse_phys(kwargs)
    geometry = gprim.gen_sphere_geom(radius, segments)
    o = sob.SceneObject(pos=pos, collision_type=collision_type,
                        is_fixed=is_fixed)
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    o.set_inertia(inertia, com, mass)
    return o


def gen_icosphere(pos=(0, 0, 0), radius=0.05, subdivisions=2,
                  rgb=const.BasicColor.DEFAULT, alpha=1.0, **kwargs):
    inertia, com, mass, collision_type, is_fixed = _parse_phys(kwargs)
    geometry = gprim.gen_icosphere_geom(radius, subdivisions)
    o = sob.SceneObject(pos=pos, collision_type=collision_type,
                        is_fixed=is_fixed)
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    o.set_inertia(inertia, com, mass)
    return o


def gen_box(pos=(0, 0, 0), half_extents=(0.05, 0.05, 0.05),
            rotmat=None, rgb=const.BasicColor.DEFAULT, alpha=1.0, **kwargs):
    inertia, com, mass, collision_type, is_fixed = _parse_phys(kwargs)
    half_extents = np.asarray(half_extents, np.float32)
    geometry = gprim.gen_box_geom(half_extents)
    o = sob.SceneObject(rotmat=rm.ensure_rotmat(rotmat),
                        pos=rm.ensure_pos(pos),
                        collision_type=collision_type,
                        is_fixed=is_fixed)
    o.add_visual(mdl.RenderModel(geometry=geometry,
                                 rgb=rgb, alpha=alpha))
    o.set_inertia(inertia, com, mass)
    return o


def gen_arrow(spos=np.zeros(3), epos=np.ones(3) * 0.01,
              shaft_radius=const.ArrowSize.SHAFT_RADIUS,
              head_radius=const.ArrowSize.HEAD_RADIUS,
              head_length=const.ArrowSize.HEAD_LENGTH,
              segments=8, rgb=const.BasicColor.DEFAULT,
              alpha=1.0, **kwargs):
    inertia, com, mass, collision_type, is_fixed = _parse_phys(kwargs)
    # collision must be ignored for arrow
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.unit_vec(epos - spos, return_length=True)
    geometry = gprim.gen_arrow_geom(length,
                                    shaft_radius,
                                    head_radius,
                                    head_length,
                                    segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    o = sob.SceneObject(rotmat=rotmat, pos=spos,
                        collision_type=collision_type,
                        is_fixed=is_fixed)
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    o.set_inertia(inertia, com, mass)
    return o


def gen_frame(pos=np.zeros(3), rotmat=np.eye(3),
              length_scale=1.0, radius_scale=1.0,
              segments=8, color_mat=const.CoordColor.RGB,
              alpha=1.0, **kwargs):
    inertia, com, mass, collision_type, is_fixed = _parse_phys(kwargs)
    # collision must be ignored for frame
    arrow_length = const.StandardAxis.ARROW_LENGTH * length_scale
    shaft_radius = const.StandardAxis.ARROW_SHAFT_RADIUS * radius_scale
    head_length = const.StandardAxis.ARROW_HEAD_LENGTH * radius_scale
    head_radius = const.StandardAxis.ARROW_HEAD_RADIUS * radius_scale
    geometry = gprim.gen_arrow_geom(arrow_length,
                                    shaft_radius,
                                    head_radius,
                                    head_length,
                                    segments)
    o = sob.SceneObject(rotmat=rotmat, pos=pos,
                        collision_type=collision_type,
                        is_fixed=is_fixed)
    # x-axis
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, const.StandardAxis.X)
    o.add_visual(mdl.RenderModel(geometry=geometry, rotmat=rotmat,
                                 rgb=color_mat[:, 0], alpha=alpha))
    # y-axis
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, const.StandardAxis.Y)
    o.add_visual(mdl.RenderModel(geometry=geometry, rotmat=rotmat,
                                 rgb=color_mat[:, 1], alpha=alpha))
    # z-axis
    o.add_visual(mdl.RenderModel(geometry=geometry, rotmat=np.eye(3, dtype=np.float32),
                                 rgb=color_mat[:, 2], alpha=alpha))
    o.set_inertia(inertia, com, mass)
    return o


def gen_plane(pos=(0, 0, 0),
              normal=const.StandardAxis.Z,
              size=(100.0, 100.0),
              thickness=1e-3,
              collision_type=const.CollisionType.PLANE,
              rgb=const.BasicColor.GRAY, alpha=1.0):
    pos = np.asarray(pos, np.float32)
    size = np.asarray(size, np.float32)
    half_extents = np.array([size[0] / 2, size[1] / 2, thickness], np.float32)
    geometry = gprim.gen_box_geom(half_extents)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, normal)
    o = sob.SceneObject(rotmat=rotmat, pos=pos,
                        collision_type=collision_type, is_fixed=True)
    o.add_visual(mdl.RenderModel(geometry=geometry,
                                 rgb=rgb, alpha=alpha))
    return o
