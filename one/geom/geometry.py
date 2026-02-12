import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.scene.geometry_ops as osgo

_geom_cache = {}


def gen_geom_from_raw(vs, fs=None):
    if fs is None:
        key = hash(vs.tobytes())
    else:
        vs, fs = _merge_vs_and_fs(vs, fs)
        key = hash(vs.tobytes() + fs.tobytes())
    if key in _geom_cache:
        return _geom_cache[key]
    g = _Geom(vs=vs, fs=fs)
    _geom_cache[key] = g
    return g


def gen_cylinder_geom(length,
                      radius=0.05,
                      n_segs=8):
    key = ("cylinder", radius, length, n_segs)
    if key in _geom_cache:
        return _geom_cache[key]
    profile = [(radius, 0.0), (radius, length)]
    verts, faces = osgo.revolve(profile, n_segs=n_segs)
    g = _Geom(vs=verts, fs=faces)
    _geom_cache[key] = g
    return g


def gen_cone_geom(length,
                  radius=0.05,
                  n_segs=8):
    key = ("cone", radius, length, n_segs)
    if key in _geom_cache:
        return _geom_cache[key]
    profile = [(radius, 0), (0, length)]
    verts, faces = osgo.revolve(profile, n_segs=n_segs)
    g = _Geom(vs=verts, fs=faces)
    _geom_cache[key] = g
    return g


def gen_sphere_geom(radius=0.05, n_segs=8):
    key = ("sphere", radius, n_segs)
    if key in _geom_cache:
        return _geom_cache[key]
    theta = np.linspace(0, np.pi, n_segs // 2 + 2)
    r = radius * np.sin(theta)
    z = -radius * np.cos(theta)
    profile = np.stack([r, z], axis=1)
    verts, faces = osgo.revolve(profile, n_segs=n_segs)
    g = _Geom(vs=verts, fs=faces)
    _geom_cache[key] = g
    return g


def gen_icosphere_geom(radius=0.05, n_subs=2):
    key = ("icosphere", radius, n_subs)
    if key in _geom_cache:
        return _geom_cache[key]
    verts, faces = osgo.icosahedron()
    for _ in range(n_subs):
        verts, faces = osgo.subdivide_once(verts, faces)
    verts = verts * radius
    g = _Geom(vs=verts, fs=faces)
    _geom_cache[key] = g
    return g


def gen_arrow_geom(length,
                   shaft_radius=ouc.ArrowSize.SHAFT_RADIUS,
                   head_length=ouc.ArrowSize.HEAD_LENGTH,
                   head_radius=ouc.ArrowSize.HEAD_RADIUS,
                   n_segs=8):
    key = ("arrow", shaft_radius, length, head_radius, head_length, n_segs)
    if key in _geom_cache:
        return _geom_cache[key]
    shaft_profile = [(shaft_radius, 0.0),
                     (shaft_radius, length - head_length)]
    head_profile = [(head_radius, length - head_length),
                    (0.0, length)]
    profile = shaft_profile + head_profile
    verts, faces = osgo.revolve(profile, n_segs=n_segs)
    g = _Geom(vs=verts, fs=faces)
    _geom_cache[key] = g
    return g


def gen_box_geom(half_extents=(0.05, 0.05, 0.05)):
    hx, hy, hz = half_extents
    key = ("box", hx, hy, hz)
    if key in _geom_cache:
        return _geom_cache[key]
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
    g = _Geom(vs=verts, fs=faces)
    _geom_cache[key] = g
    return g


def gen_trapezoid_geom(height=0.05,
                       bottom_length=0.05,
                       top_length=0.03):
    height = float(height)
    bottom_half = float(bottom_length) * 0.5
    top_half = float(top_length) * 0.5
    key = ("trapezoid", height, bottom_half, top_half)
    if key in _geom_cache:
        return _geom_cache[key]

    b0 = np.array([-bottom_half, -bottom_half, 0.0], dtype=np.float32)
    b1 = np.array([bottom_half, -bottom_half, 0.0], dtype=np.float32)
    b2 = np.array([bottom_half, bottom_half, 0.0], dtype=np.float32)
    b3 = np.array([-bottom_half, bottom_half, 0.0], dtype=np.float32)
    t0 = np.array([-top_half, -top_half, height], dtype=np.float32)
    t1 = np.array([top_half, -top_half, height], dtype=np.float32)
    t2 = np.array([top_half, top_half, height], dtype=np.float32)
    t3 = np.array([-top_half, top_half, height], dtype=np.float32)

    verts = np.asarray([b0, b1, b2, b3, t0, t1, t2, t3], dtype=np.float32)
    faces = np.asarray([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 5, 1], [0, 4, 5],  # side 0-1
        [1, 6, 2], [1, 5, 6],  # side 1-2
        [2, 7, 3], [2, 6, 7],  # side 2-3
        [3, 4, 0], [3, 7, 4],  # side 3-0
    ], dtype=np.uint32)
    # Ensure outward triangle winding (important for consistent normals/shading).
    center = np.mean(verts, axis=0)
    tri_vs = verts[faces]
    tri_normals = np.cross(tri_vs[:, 1] - tri_vs[:, 0], tri_vs[:, 2] - tri_vs[:, 0])
    tri_centers = np.mean(tri_vs, axis=1)
    inward_mask = np.einsum('ij,ij->i', tri_normals, tri_centers - center) < 0.0
    if np.any(inward_mask):
        faces[inward_mask] = faces[inward_mask][:, [0, 2, 1]]
    g = _Geom(vs=verts, fs=faces)
    _geom_cache[key] = g
    return g


def gen_capsule_geom(radius=0.05, half_length=0.1, n_segs=32):
    key = ("capsule", radius, half_length, n_segs)
    if key in _geom_cache:
        return _geom_cache[key]
    # z goes from -half_length-radius  ->  +half_length+radius
    # center cylinder spans [-half_length,+half_length]
    theta = np.linspace(0, np.pi / 2, n_segs // 2 + 1)
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
    verts, faces = osgo.revolve(profile, n_segs=n_segs)
    g = _Geom(vs=verts, fs=faces)
    _geom_cache[key] = g
    return g


def _merge_vs_and_fs(vs, fs, tol=1e-6):
    q = np.round(vs / tol).astype(np.int64)
    unique_q, inv = np.unique(q, axis=0, return_inverse=True)
    new_vs = np.zeros((len(unique_q), 3), dtype=np.float32)
    np.add.at(new_vs, inv, vs)
    counts = np.bincount(inv)
    new_vs /= counts[:, None]
    new_fs = inv[fs].astype(np.uint32).copy()  # ensure contiguous
    return new_vs, new_fs


class _Geom:
    """DO NOT USE DIRECTLY. Use geometry_primitive instead."""

    def __init__(self, vs, fs=None, vrgbs=None):
        if fs is not None:
            self._vs, self._fs = vs, fs
            self._fns, self._vns, self._fareas = self._compute_vns()
        else:
            self._vs = vs
            self._fs = None
            self._fns = None
            self._vns = None
            self._vrgbs = vrgbs

    @property
    def vs(self):  # verts
        return self._vs

    @property
    def fs(self):  # faces
        return self._fs

    @property
    def vns(self):  # vertex normals
        return self._vns

    @property
    def fns(self):  # face normals
        return self._fns

    def _compute_vns(self):
        v1 = self._vs[self._fs[:, 1]] - self._vs[self._fs[:, 0]]
        v2 = self._vs[self._fs[:, 2]] - self._vs[self._fs[:, 0]]
        raw_fns = np.cross(v1, v2).astype(np.float32)
        fn_lens, unit_fns = oum.unit_vec(raw_fns)
        fareas = 0.5 * fn_lens  # face areas
        # vert normals
        raw_vns = np.zeros_like(self._vs)
        np.add.at(raw_vns, self._fs[:, 0], unit_fns)
        np.add.at(raw_vns, self._fs[:, 1], unit_fns)
        np.add.at(raw_vns, self._fs[:, 2], unit_fns)
        _, unit_vns = oum.unit_vec(raw_vns)
        return unit_fns, unit_vns, fareas
