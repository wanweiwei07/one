import numpy as np
import one.scene.scene_object as sob
import one.scene.render_model as mdl
import one.scene.geometry_primitive as gprim
import one.utils.constant as const
import one.utils.math as rm


def gen_cylinder(spos=np.zeros(3), epos=np.ones(3) * 0.01,
                 radius=0.05, segments=8, collision_type=None,
                 rgb=const.BasicColor.DEFAULT, alpha=1.0):
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.unit_vec(epos - spos, return_length=True)
    geometry = gprim.gen_cylinder_geom(length, radius, segments)
    o = sob.SceneObject(collision_type=collision_type)
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    o.set_rotmat_pos(rotmat, spos)
    return o


def gen_cone(spos=np.zeros(3), epos=np.ones(3) * 0.01,
             radius=0.05, segments=8,
             rgb=const.BasicColor.DEFAULT, alpha=1.0):
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.unit_vec(epos - spos, return_length=True)
    geometry = gprim.gen_cone_geom(length, radius, segments)
    o = sob.SceneObject()
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    o.set_rotmat_pos(rotmat, spos)
    return o


def gen_sphere(pos=np.zeros(3), radius=0.05, segments=8,
               rgb=const.BasicColor.DEFAULT, alpha=1.0):
    geometry = gprim.gen_sphere_geom(radius, segments)
    o = sob.SceneObject()
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    o.pos = pos
    return o


def gen_icosphere(pos=np.zeros(3), radius=0.05, subdivisions=2,
                  rgb=const.BasicColor.DEFAULT, alpha=1.0):
    geometry = gprim.gen_icosphere_geom(radius, subdivisions)
    o = sob.SceneObject()
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    o.pos = pos
    return o


def gen_arrow(spos=np.zeros(3), epos=np.ones(3) * 0.01,
              shaft_radius=const.ArrowSize.SHAFT_RADIUS,
              head_radius=const.ArrowSize.HEAD_RADIUS,
              head_length=const.ArrowSize.HEAD_LENGTH,
              segments=8,
              rgb=const.BasicColor.DEFAULT, alpha=1.0):
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.unit_vec(epos - spos, return_length=True)
    geometry = gprim.gen_arrow_geom(length,
                                    shaft_radius,
                                    head_radius,
                                    head_length,
                                    segments)
    o = sob.SceneObject()
    o.add_visual(mdl.RenderModel(geometry=geometry, rgb=rgb, alpha=alpha))
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    o.set_rotmat_pos(rotmat, spos)
    return o


def gen_frame(pos=np.zeros(3), rotmat=np.eye(3),
              length_scale=1.0, radius_scale=1.0,
              segments=8, color_mat=const.CoordColor.RGB, alpha=1.0):
    arrow_length = const.StandardAxis.ARROW_LENGTH * length_scale
    shaft_radius = const.StandardAxis.ARROW_SHAFT_RADIUS * radius_scale
    head_length = const.StandardAxis.ARROW_HEAD_LENGTH * radius_scale
    head_radius = const.StandardAxis.ARROW_HEAD_RADIUS * radius_scale
    geometry = gprim.gen_arrow_geom(arrow_length,
                                    shaft_radius,
                                    head_radius,
                                    head_length,
                                    segments)
    o = sob.SceneObject(rotmat=rotmat, pos=pos)
    # x-axis
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, const.StandardAxis.X)
    o.add_visual(mdl.RenderModel(geometry=geometry, rotmat=rotmat,
                                 rgb=color_mat[:, 0], alpha=alpha))
    # y-axis
    o.add_visual(mdl.RenderModel(geometry=geometry, rotmat=rotmat,
                                 rgb=color_mat[:, 1], alpha=alpha))
    # z-axis
    o.add_visual(mdl.RenderModel(geometry=geometry, rotmat=np.eye(3,dtype=np.float32),
                                 rgb=color_mat[:, 2], alpha=alpha))
    return o
