import numpy as np
import one.scene.geometry_operations as gops
import one.scene.scene_object as sob
import one.utils.constant as const
import one.utils.math as rm


def gen_cylinder(spos=np.zeros(3), epos=np.ones(3) * 0.01,
                 radius=0.05, segments=8, rgb=const.BasicColor.DEFAULT, alpha=1.0):
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.normalize(epos - spos, return_length=True)
    profile = [(radius, 0.0), (radius, length)]
    verts, faces = gops.revolve(profile, segments=segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    verts = verts @ rotmat.T + spos
    return_obj = sob.SceneObject()
    return_obj.add_visual(mdl.Model(geometry=(verts, faces), rgb=rgb, alpha=alpha))
    return return_obj


def gen_cone(spos=np.zeros(3), epos=np.ones(3) * 0.01,
             radius=0.05, segments=8, rgb=const.BasicColor.DEFAULT, alpha=1.0):
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.normalize(epos - spos, return_length=True)
    profile = [(radius, 0), (0, length)]
    verts, faces = gops.revolve(profile, segments=segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    verts = verts @ rotmat.T + spos
    return_obj = sob.SceneObject()
    return_obj.add_visual(mdl.Model(geometry=(verts, faces), rgb=rgb, alpha=alpha))
    return return_obj


def gen_sphere(pos=np.zeros(3), radius=0.05, segments=8,
               rgb=const.BasicColor.DEFAULT, alpha=1.0):
    theta = np.linspace(0, np.pi, segments // 2 + 2)
    r = radius * np.sin(theta)
    z = -radius * np.cos(theta)
    profile = np.stack([r, z], axis=1)
    verts, faces = gops.revolve(profile, segments=segments)
    return_obj = sob.SceneObject()
    return_obj.set_pos(pos=pos)
    return_obj.add_visual(mdl.Model(geometry=(verts, faces), rgb=rgb, alpha=alpha))
    return return_obj


def gen_icosphere(pos=np.zeros(3), radius=0.05, subdivisions=2,
                  rgb=const.BasicColor.DEFAULT, alpha=1.0):
    verts, faces = gops.icosahedron()
    for _ in range(subdivisions):
        verts, faces = gops.subdivide_once(verts, faces)
    verts = verts * radius
    return_obj = sob.SceneObject()
    return_obj.set_pos(pos=pos)
    return_obj.add_visual(mdl.Model(geometry=(verts, faces), rgb=rgb, alpha=alpha))
    return return_obj


def gen_arrow(spos=np.zeros(3), epos=np.ones(3) * 0.01,
              shaft_radius=const.ArrowSize.SHAFT_RADIUS,
              head_radius=const.ArrowSize.HEAD_RADIUS,
              head_length=const.ArrowSize.HEAD_LENGTH,
              segments=8, rgb=const.BasicColor.DEFAULT, alpha=1.0):
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.normalize(epos - spos, return_length=True)
    shaft_profile = [(shaft_radius, 0.0), (shaft_radius, length - head_length)]
    head_profile = [(head_radius, length - head_length), (0.0, length)]
    profile = shaft_profile + head_profile
    verts, faces = gops.revolve(profile, segments=segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    verts = verts @ rotmat.T + spos
    return_obj = sob.SceneObject()
    return_obj.add_visual(mdl.Model(geometry=(verts, faces), rgb=rgb, alpha=alpha))
    return return_obj


def gen_frame(pos=np.zeros(3), rotmat=np.eye(3),
              arrow_length=const.StandardAxis.ARROW_LENGTH,
              arrow_shaft_radius=const.StandardAxis.ARROW_SHAFT_RADIUS,
              arrow_head_length=const.StandardAxis.ARROW_HEAD_LENGTH,
              arrow_head_radius=const.StandardAxis.ARROW_HEAD_RADIUS,
              segments=8, color_mat=const.CoordColor.RGB, alpha=1.0):
    shaft_profile = [(arrow_shaft_radius, 0.0), (arrow_shaft_radius, arrow_length - arrow_head_length)]
    head_profile = [(arrow_head_radius, arrow_length - arrow_head_length), (0.0, arrow_length)]
    profile = shaft_profile + head_profile
    verts, faces = gops.revolve(profile, segments=segments)
    return_obj = sob.SceneObject(rotmat=rotmat, pos=pos)
    rotmat_x = rm.rotmat_between_vecs(const.StandardAxis.Z, const.StandardAxis.X)
    rotmat_y = rm.rotmat_between_vecs(const.StandardAxis.Z, const.StandardAxis.Y)
    rotmat_z = np.eye(3, dtype=np.float32)
    return_obj.add_visual(mdl.Model(geometry=(verts, faces), rotmat=rotmat_x, rgb=color_mat[:, 0], alpha=alpha))
    return_obj.add_visual(mdl.Model(geometry=(verts, faces), rotmat=rotmat_y, rgb=color_mat[:, 1], alpha=alpha))
    return_obj.add_visual(mdl.Model(geometry=(verts, faces), rotmat=rotmat_z, rgb=color_mat[:, 2], alpha=alpha))
    return return_obj


if __name__ == '__main__':
    from one import wd, scn, mdl

    # cyl = mdl.Model(gen_cone(radius=0.01, height=0.02, segments=36))
    cyl = mdl.Model(gen_frame())
    scene = scn.Scene()
    scene.add(cyl)
    base = wd.World(cam_pos=np.array([.5, 0, 1]), toggle_auto_cam_orbit=True)
    base.set_scene(scene)
    base.run()
