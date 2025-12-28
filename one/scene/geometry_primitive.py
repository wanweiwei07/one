import numpy as np
import one.scene.geometry as geom
import one.scene.geometry_operation as gops
import one.utils.constant as const

_primitive_cache = {}


def gen_cylinder_geom(length,
                      radius=0.05,
                      segments=8):
    key = ("cylinder", radius, length, segments)
    if key in _primitive_cache:
        return _primitive_cache[key]
    profile = [(radius, 0.0), (radius, length)]
    verts, faces = gops.revolve(profile, segments=segments)
    g = geom.Geometry(verts=verts, faces=faces)
    _primitive_cache[key] = g
    return g


def gen_cone_geom(length,
                  radius=0.05,
                  segments=8):
    key = ("cone", radius, length, segments)
    if key in _primitive_cache:
        return _primitive_cache[key]
    profile = [(radius, 0), (0, length)]
    verts, faces = gops.revolve(profile, segments=segments)
    g = geom.Geometry(verts=verts, faces=faces)
    _primitive_cache[key] = g
    return g


def gen_sphere_geom(radius=0.05, segments=8):
    key = ("sphere", radius, segments)
    if key in _primitive_cache:
        return _primitive_cache[key]
    theta = np.linspace(0, np.pi, segments // 2 + 2)
    r = radius * np.sin(theta)
    z = -radius * np.cos(theta)
    profile = np.stack([r, z], axis=1)
    verts, faces = gops.revolve(profile, segments=segments)
    g = geom.Geometry(verts=verts, faces=faces)
    _primitive_cache[key] = g
    return g


def gen_icosphere_geom(radius=0.05, subdivisions=2):
    key = ("icosphere", radius, subdivisions)
    if key in _primitive_cache:
        return _primitive_cache[key]
    verts, faces = gops.icosahedron()
    for _ in range(subdivisions):
        verts, faces = gops.subdivide_once(verts, faces)
    verts = verts * radius
    g = geom.Geometry(verts=verts, faces=faces)
    _primitive_cache[key] = g
    return g


def gen_arrow_geom(length,
                   shaft_radius=const.ArrowSize.SHAFT_RADIUS,
                   head_radius=const.ArrowSize.HEAD_RADIUS,
                   head_length=const.ArrowSize.HEAD_LENGTH,
                   segments=8):
    key = ("arrow", shaft_radius, length, head_radius, head_length, segments)
    if key in _primitive_cache:
        return _primitive_cache[key]
    shaft_profile = [(shaft_radius, 0.0),
                     (shaft_radius, length - head_length)]
    head_profile = [(head_radius, length - head_length),
                    (0.0, length)]
    profile = shaft_profile + head_profile
    verts, faces = gops.revolve(profile, segments=segments)
    g = geom.Geometry(verts=verts, faces=faces)
    _primitive_cache[key] = g
    return g


def gen_box_geom(half_extents=(0.05, 0.05, 0.05)):
    hx, hy, hz = half_extents
    key = ("box", hx, hy, hz)
    if key in _primitive_cache:
        return _primitive_cache[key]
    verts = np.array([[-hx, -hy, -hz],
                      [hx, -hy, -hz],
                      [hx, hy, -hz],
                      [-hx, hy, -hz],
                      [-hx, -hy, hz],
                      [hx, -hy, hz],
                      [hx, hy, hz],
                      [-hx, hy, hz]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 3, 2],  # bottom
                      [4, 5, 6], [4, 6, 7],  # top
                      [0, 5, 4], [0, 1, 5],  # -y
                      [1, 6, 5], [1, 2, 6],  # +x
                      [2, 7, 6], [2, 3, 7],  # +y
                      [3, 4, 7], [3, 0, 4]], dtype=np.uint32)  # -x
    g = geom.Geometry(verts=verts, faces=faces)
    _primitive_cache[key] = g
    return g


def gen_capsule_geom(radius=0.05, half_length=0.1, segments=32):
    key = ("capsule", radius, half_length, segments)
    if key in _primitive_cache:
        return _primitive_cache[key]
    # z goes from -half_length-radius  ->  +half_length+radius
    # center cylinder spans [-half_length,+half_length]
    theta = np.linspace(0, np.pi / 2, segments // 2 + 1)
    r_hemi = radius * np.sin(theta)
    z_hemi = radius * np.cos(theta)
    # lower hemisphere (shift down)
    lower = np.stack([r_hemi, -half_length - z_hemi], axis=1)
    # upper hemisphere (shift up)
    upper = np.stack([r_hemi[::-1], half_length + z_hemi[::-1]], axis=1)
    # middle
    mid = np.array([[radius, -half_length], [radius, +half_length]])
    # remove duplicate radius=0 middle point once
    profile = np.concatenate([lower, mid, upper], axis=0)
    verts, faces = gops.revolve(profile, segments=segments)
    g = geom.Geometry(verts=verts, faces=faces)
    _primitive_cache[key] = g
    return g
