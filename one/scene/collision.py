import numpy as np
import one.utils.math as rm
import one.utils.constant as const
import one.scene.geometry_primitive as gprim
import one.scene.render_model as mdl
from enum import Enum


class CollisionShape:
    def __init__(self, rotmat=None, pos=None):
        self._rotmat = np.eye(3, dtype=np.float32) if rotmat is None else rotmat
        self._pos = np.zeros(3, dtype=np.float32) if pos is None else pos

    def to_render_model(self):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError


class SphereCollisionShape(CollisionShape):

    @classmethod
    def fit_from_model(cls, m):
        verts = (m.rotmat @ m.geometry.verts.T).T + m.pos
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = np.linalg.norm(verts - center, axis=1).max()
        shape = cls(radius=radius, pos=center)
        return shape

    def __init__(self, radius, pos=None):
        super().__init__(rotmat=np.eye(3, dtype=np.float32), pos=pos)
        self._radius = radius

    def clone(self):
        return self.__class__(radius=self._radius,
                              pos=self._pos.copy())

    def to_render_model(self):
        g = gprim.gen_icosphere_geom(radius=self._radius)
        return mdl.RenderModel(geometry=g, rotmat=self._rotmat, pos=self._pos,
                               rgb=const.BasicColor.GRAY, alpha=0.3)

    @property
    def radius(self):
        return self._radius


class CapsuleCollisionShape(CollisionShape):

    @classmethod
    def fit_from_model(cls, m):
        verts = (m.rotmat @ m.geometry.verts.T).T + m.pos
        faces = m.geometry.faces
        mean, pcmat = rm.area_weighted_pca(verts, faces)
        pc_ax = pcmat[:, -1]
        proj = (verts - mean) @ pc_ax
        mn = proj.min()
        mx = proj.max()
        center = mean + pc_ax * (mn + mx) * 0.5
        d = verts - center
        axial = d @ pc_ax
        radial_sq = np.sum(d * d, axis=1) - axial * axial
        radial_sq = np.maximum(radial_sq, 0.0)
        radius = np.sqrt(radial_sq).max()
        half_length = (mx - mn) * 0.5 - radius / 1.2
        half_length = max(half_length, 0.0)
        shape = cls(radius=radius,
                    half_length=half_length,
                    rotmat=pcmat,
                    pos=center)
        return shape

    def __init__(self, radius, half_length, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        self._radius = radius
        self._half_length = half_length

    def clone(self):
        return self.__class__(radius=self._radius,
                              half_length=self._half_length,
                              rotmat=self._rotmat.copy(),
                              pos=self._pos.copy())

    def to_render_model(self):
        g = gprim.gen_capsule_geom(radius=self._radius, half_length=self._half_length)
        return mdl.RenderModel(geometry=g, rotmat=self._rotmat, pos=self._pos,
                               rgb=const.BasicColor.GRAY, alpha=0.3)

    @property
    def radius(self):
        return self._radius

    @property
    def half_length(self):
        return self._half_length


class AABBCollisionShape(CollisionShape):

    @classmethod
    def fit_from_model(cls, m):
        verts = (m.rotmat @ m.geometry.verts.T).T + m.pos
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        half_extents = (vmax - vmin) * 0.5
        center = (vmin + vmax) * 0.5
        shape = cls(half_extents=half_extents, pos=center)
        return shape

    def __init__(self, half_extents, pos=None):
        super().__init__(rotmat=np.eye(3, dtype=np.float32), pos=pos)
        # half_extents: [length/2, width/2, height/2]
        self._half_extents = half_extents

    def clone(self):
        return self.__class__(half_extents=self._half_extents,
                              pos=self._pos.copy())

    def to_render_model(self):
        g = gprim.gen_box_geom(half_extents=self._half_extents)
        return mdl.RenderModel(geometry=g, rotmat=self._rotmat, pos=self._pos,
                               rgb=const.BasicColor.GRAY, alpha=0.3)

    @property
    def half_extents(self):
        return self._half_extents


class OBBCollisionShape(CollisionShape):

    @classmethod
    def fit_from_model(cls, m):
        verts = (m.rotmat @ m.geometry.verts.T).T + m.pos
        faces = m.geometry.faces
        mean, pcmat = rm.area_weighted_pca(verts, faces)
        local = (verts - mean) @ pcmat
        loc_vmin = local.min(axis=0)
        loc_vmax = local.max(axis=0)
        half_extents = (loc_vmax - loc_vmin) * 0.5
        loc_center = (loc_vmin + loc_vmax) * 0.5
        center = mean + pcmat @ loc_center
        shape = cls(half_extents=half_extents,
                    rotmat=pcmat,
                    pos=center)
        return shape

    def __init__(self, half_extents, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        # half_extents: [length/2, width/2, height/2]
        self._half_extents = half_extents

    def clone(self):
        return self.__class__(half_extents=self._half_extents,
                              rotmat=self._rotmat.copy(),
                              pos=self._pos.copy())

    def to_render_model(self):
        g = gprim.gen_box_geom(half_extents=self._half_extents)
        return mdl.RenderModel(geometry=g, rotmat=self._rotmat, pos=self._pos,
                               rgb=const.BasicColor.GRAY, alpha=0.3)

    @property
    def half_extents(self):
        return self._half_extents


class MeshCollisionShape(CollisionShape):

    def __init__(self, file_path=None, geometry=None, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        self._file_path = file_path
        self._geometry = geometry

    def clone(self):
        return self.__class__(file_path=self._file_path,
                              geometry=self._geometry,
                              rotmat=self._rotmat.copy(),
                              pos=self._pos.copy())

    def to_render_model(self):
        return mdl.RenderModel(geometry=self._geometry, rotmat=self._rotmat, pos=self._pos,
                               rgb=const.BasicColor.GRAY, alpha=0.3)

    @property
    def file_path(self):
        return self._file_path

    @property
    def geometry(self):
        return self._geometry
