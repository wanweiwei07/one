import numpy as np
from torch.nn.init import dirac_

import one.scene.geom as geom
import one.scene.geomops as geomops
import one.utils.constant as const
import one.utils.math as rm


def combine_mesh(*meshes):
    verts_list = []
    faces_list = []
    face_normals_list = []
    vert_offset = 0
    for mesh in meshes:
        verts_list.append(mesh.verts)
        faces_list.append(mesh.faces + vert_offset)
        face_normals_list.append(mesh.face_normals)
        vert_offset += len(mesh.verts)
    verts = np.vstack(verts_list)
    faces = np.vstack(faces_list)
    face_normals = np.vstack(face_normals_list)
    return geom.Mesh(verts=verts, faces=faces, face_normals=face_normals)


def gen_cylinder(spos=np.zeros(3), epos=np.ones(3) * 0.01,
                 radius=0.05, segments=8, rgb=const.BasicColor.DEFAULT, alpha=1.0):
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.normalize(epos - spos, return_length=True)
    profile = [(radius, 0.0), (radius, length)]
    verts, faces = geomops.revolve(profile, segments=segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    verts = verts @ rotmat.T + spos
    return geom.Mesh(verts=verts, faces=faces, rgb=rgb, alpha=alpha)


def gen_cone(spos=np.zeros(3), epos=np.ones(3) * 0.01,
             radius=0.05, segments=8, rgb=const.BasicColor.DEFAULT, alpha=1.0):
    spos = np.asarray(spos, np.float32)
    epos = np.asarray(epos, np.float32)
    length, dir_vec = rm.normalize(epos - spos, return_length=True)
    profile = [(radius, 0), (0, length)]
    verts, faces = geomops.revolve(profile, segments=segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    verts = verts @ rotmat.T + spos
    return geom.Mesh(verts=verts, faces=faces, rgb=rgb, alpha=alpha)


def gen_sphere(pos=np.zeros(3), radius=0.05, segments=8,
               rgb=const.BasicColor.DEFAULT, alpha=1.0):
    theta = np.linspace(0, np.pi, segments // 2 + 2)
    r = radius * np.sin(theta)
    z = -radius * np.cos(theta)
    profile = np.stack([r, z], axis=1)
    verts, faces = geomops.revolve(profile, segments=segments)
    return geom.Mesh(verts=verts, faces=faces, rgb=rgb, alpha=alpha)


def gen_icosphere(pos=np.zeros(3), radius=0.05, subdivisions=2,
                  rgb=const.BasicColor.DEFAULT, alpha=1.0):
    verts, faces = geomops.icosahedron()
    for _ in range(subdivisions):
        verts, faces = geomops.subdivide_once(verts, faces)
    verts = verts * radius + np.asarray(pos, dtype=np.float32)
    return geom.Mesh(verts=verts, faces=faces, rgb=rgb, alpha=alpha)


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
    verts, faces = geomops.revolve(profile, segments=segments)
    rotmat = rm.rotmat_between_vecs(const.StandardAxis.Z, dir_vec)
    verts = verts @ rotmat.T + spos
    return geom.Mesh(verts=verts, faces=faces, rgb=rgb, alpha=alpha)


def gen_frame(pos=np.zeros(3), rotmat=np.eye(3),
              arrow_length=const.StandardAxis.ARROW_LENGTH,
              arrow_shaft_radius=const.StandardAxis.ARROW_SHAFT_RADIUS,
              arrow_head_length=const.StandardAxis.ARROW_HEAD_LENGTH,
              arrow_head_radius=const.StandardAxis.ARROW_HEAD_RADIUS,
              segments=8, color_mat=const.CoordColor.RGB, alpha=1.0):
    shaft_profile = [(arrow_shaft_radius, 0.0), (arrow_shaft_radius, arrow_length - arrow_head_length)]
    head_profile = [(arrow_head_radius, arrow_length - arrow_head_length), (0.0, arrow_length)]
    profile = shaft_profile + head_profile
    verts, faces = geomops.revolve(profile, segments=segments)
    rgb = np.vstack([np.tile(color_mat[:, 0], (verts.shape[0], 1)),  # X - red
                     np.tile(color_mat[:, 1], (verts.shape[0], 1)),  # Y - green
                     np.tile(color_mat[:, 2], (verts.shape[0], 1))])
    n_verts = verts.shape[0]
    rot_x = rm.rotmat_between_vecs(const.StandardAxis.Z, rotmat[:, 0])
    rot_y = rm.rotmat_between_vecs(const.StandardAxis.Z, rotmat[:, 1])
    rot_z = rm.rotmat_between_vecs(const.StandardAxis.Z, rotmat[:, 2])
    verts_x = verts @ rot_x.T + pos
    verts_y = verts @ rot_y.T + pos
    verts_z = verts @ rot_z.T + pos
    verts = np.vstack([verts_x, verts_y, verts_z])
    offsets = np.array([0, n_verts, 2 * n_verts], dtype=np.uint32)
    faces = np.vstack([faces + offsets[i] for i in range(3)])
    return geom.Mesh(verts=verts, faces=faces, rgb=rgb, alpha=alpha)


if __name__ == '__main__':
    from one import wd, scn, mdl

    # cyl = mdl.Model(gen_cone(radius=0.01, height=0.02, segments=36))
    cyl = mdl.Model(gen_frame())
    scene = scn.Scene()
    scene.add(cyl)
    base = wd.World(cam_pos=np.array([.5, 0, 1]), toggle_auto_cam_orbit=True)
    base.set_scene(scene)
    base.run()
