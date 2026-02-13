import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.geom.geometry as ogg
import one.scene.render_model as osrm


class CollisionShape:
    def __init__(self, rotmat=None, pos=None):
        self._tf = oum.tf_from_rotmat_pos(
            rotmat, pos)
        self._geom = None  # lazy geom cache
        self._aabb = None

    def to_render_model(self):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    @property
    def geom(self):
        if self._geom is None:
            self._geom = self._build_geom()
        return self._geom

    @property
    def tf(self):
        return self._tf.copy()

    @property
    def quat(self):
        return oum.quat_from_rotmat(
            self._tf[:3, :3])

    @property
    def rotmat(self):
        return self._tf[:3, :3].copy()

    @property
    def pos(self):
        return self._tf[:3, 3].copy()

    @property
    def aabb(self):
        raise NotImplementedError

    def _build_geom(self):
        raise NotImplementedError


class SphereCollisionShape(CollisionShape):

    @classmethod
    def fit_from_geom(
            cls, geom, rotmat, pos):
        vs = (rotmat @
              geom.vs.T).T + pos
        mins = vs.min(axis=0)
        maxs = vs.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = np.linalg.norm(
            vs - center, axis=1).max()
        shape = cls(radius=radius, pos=center)
        return shape

    def __init__(self, radius, pos=None):
        super().__init__(pos=pos)
        self._radius = radius

    def clone(self):
        return self.__class__(
            radius=self._radius,
            pos=self.pos)

    def to_render_model(self):
        return osrm.RenderModel(
            geom=self.geom,
            pos=self.pos,
            rgb=ouc.BasicColor.ORANGE,
            alpha=ouc.ALPHA.TRANSPARENT)

    @property
    def radius(self):
        return self._radius

    @property
    def aabb(self):
        if self._aabb is not None:
            return self._aabb
        min_corner = self.pos - self._radius
        max_corner = self.pos + self._radius
        return min_corner, max_corner

    def _build_geom(self):
        return ogg.gen_icosphere_geom(
            radius=self._radius)


class CapsuleCollisionShape(CollisionShape):

    @classmethod
    def fit_from_geom(cls, geom, rotmat, pos):
        vs = (rotmat @ geom.vs.T).T + pos
        fs = geom.fs
        mean, pcmat = oum.area_weighted_pca(vs, fs)
        pc_ax = pcmat[:, -1]
        proj = (vs - mean) @ pc_ax
        mn = proj.min()
        mx = proj.max()
        center = mean + pc_ax * (mn + mx) * 0.5
        d = vs - center
        axial = d @ pc_ax
        radial_sq = np.sum(d * d, axis=1) - axial * axial
        radial_sq = np.maximum(radial_sq, 0.0)
        radius = np.sqrt(radial_sq).max()
        half_length = (mx - mn) * 0.5 - radius / 1.2
        half_length = max(half_length, 0.001)
        shape = cls(radius=radius,
                    half_length=half_length,
                    rotmat=pcmat,
                    pos=center)
        return shape

    def __init__(
            self, radius, half_length,
            rotmat=None, pos=None):
        super().__init__(
            rotmat=rotmat, pos=pos)
        self._radius = radius
        self._half_length = half_length

    def clone(self):
        return self.__class__(
            radius=self._radius,
            half_length=self._half_length,
            rotmat=self.rotmat,
            pos=self.pos)

    def to_render_model(self):
        return osrm.RenderModel(
            geom=self.geom,
            rotmat=self.rotmat, pos=self.pos,
            rgb=ouc.BasicColor.ORANGE,
            alpha=ouc.ALPHA.TRANSPARENT)

    def _build_geom(self):
        return ogg.gen_capsule_geom(
            radius=self._radius,
            half_length=self._half_length)

    @property
    def radius(self):
        return self._radius

    @property
    def half_length(self):
        return self._half_length

    @property
    def aabb(self):
        if self._aabb is not None:
            return self._aabb
        half = np.array([self._radius,
                         self._radius,
                         self._half_length + self._radius],
                        dtype=np.float32)
        ext = np.abs(self.rotmat) @ half
        min_corner = self.pos - ext
        max_corner = self.pos + ext
        return min_corner, max_corner


class AABBCollisionShape(CollisionShape):

    @classmethod
    def fit_from_geom(
            cls, geom, rotmat, pos):
        vs = (rotmat @ geom.vs.T).T + pos
        vmin = vs.min(axis=0)
        vmax = vs.max(axis=0)
        half_extents = (vmax - vmin) * 0.5
        center = (vmin + vmax) * 0.5
        shape = cls(
            half_extents=half_extents,
            pos=center)
        return shape

    def __init__(self, half_extents,
                 pos=None):
        super().__init__(pos=pos)
        # half_extents: [length/2, width/2, height/2]
        self._half_extents = half_extents

    def clone(self):
        return self.__class__(
            half_extents=self._half_extents,
            pos=self.pos)

    def to_render_model(self):
        return osrm.RenderModel(
            geom=self.geom, pos=self.pos,
            rgb=ouc.BasicColor.ORANGE,
            alpha=ouc.ALPHA.TRANSPARENT)

    @property
    def half_extents(self):
        return self._half_extents

    @property
    def aabb(self):
        if self._aabb is not None:
            return self._aabb
        half = np.asarray(self._half_extents,
                          dtype=np.float32)
        ext = np.abs(self.rotmat) @ half
        min_corner = self.pos - ext
        max_corner = self.pos + ext
        return min_corner, max_corner

    def _build_geom(self):
        return ogg.gen_box_geom(
            half_extents=self._half_extents)


class OBBCollisionShape(CollisionShape):

    @classmethod
    def fit_from_geom(
            cls, geom, rotmat, pos):
        vs = (rotmat @ geom.vs.T).T + pos
        fs = geom.fs
        mean, pcmat = oum.area_weighted_pca(
            vs, fs)
        local = (vs - mean) @ pcmat
        loc_vmin = local.min(axis=0)
        loc_vmax = local.max(axis=0)
        half_extents = (loc_vmax - loc_vmin) * 0.5
        loc_center = (loc_vmin + loc_vmax) * 0.5
        center = mean + pcmat @ loc_center
        shape = cls(half_extents=half_extents,
                    rotmat=pcmat, pos=center)
        return shape

    def __init__(self, half_extents,
                 rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        # half_extents: [length/2, width/2, height/2]
        self._half_extents = half_extents

    def clone(self):
        return self.__class__(
            half_extents=self._half_extents,
            rotmat=self.rotmat, pos=self.pos)

    def to_render_model(self):
        return osrm.RenderModel(
            geom=self.geom,
            rotmat=self.rotmat, pos=self.pos,
            rgb=ouc.BasicColor.ORANGE,
            alpha=ouc.ALPHA.TRANSPARENT)

    @property
    def half_extents(self):
        return self._half_extents

    @property
    def aabb(self):
        if self._aabb is not None:
            return self._aabb
        half = np.asarray(self._half_extents,
                          dtype=np.float32)
        ext = np.abs(self.rotmat) @ half
        min_corner = self.pos - ext
        max_corner = self.pos + ext
        return min_corner, max_corner

    def _build_geom(self):
        return ogg.gen_box_geom(
            half_extents=self._half_extents)


class PlaneCollisionShape(CollisionShape):
    @classmethod
    def fit_from_geom(
            cls, geom, rotmat, pos):
        vs = (rotmat @ geom.vs.T).T + pos
        fs = geom.fs
        mean, pcmat = oum.area_weighted_pca(
            vs, fs)
        center = mean
        normal = pcmat[:, 0]
        shape = cls(normal=normal, pos=center)
        return shape

    def __init__(
            self, normal=ouc.StandardAxis.Z,
            pos=None):
        rotmat = oum.rotmat_between_vecs(
            ouc.StandardAxis.Z, normal)
        super().__init__(rotmat=rotmat, pos=pos)

    def clone(self):
        return self.__class__(
            normal=self.rotmat[:, 2], pos=self.pos)

    def to_render_model(self):
        return osrm.RenderModel(
            geom=self.geom,
            rotmat=self.rotmat,
            pos=self.pos,
            rgb=ouc.BasicColor.ORANGE,
            alpha=ouc.ALPHA.TRANSPARENT)

    @property
    def normal(self):
        return self.rotmat[:, 2].copy()

    @property
    def aabb(self):
        if self._aabb is not None:
            return self._aabb
        half = np.array([100.0, 100.0, 1e-3], dtype=np.float32)
        ext = np.abs(self.rotmat) @ half
        min_corner = self.pos - ext
        max_corner = self.pos + ext
        return min_corner, max_corner

    def _build_geom(self):
        half_extents = np.array(
            [100.0, 100.0, 1e-3],
            dtype=np.float32)
        return ogg.gen_box_geom(
            half_extents=half_extents)


class MeshCollisionShape(CollisionShape):

    def __init__(self, file_path=None,
                 geom=None, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        self._file_path = file_path
        self._geom = geom

    def clone(self):
        return self.__class__(
            file_path=self._file_path, geom=self._geom,
            rotmat=self.rotmat, pos=self.pos)

    def to_render_model(self):
        vs = self._geom.vs
        ext = min(vs.max(axis=0) - vs.min(axis=0)) * .01
        vs = vs + self._geom.vns * ext
        return osrm.RenderModel(
            geom=(vs, self._geom.fs),
            rotmat=self.rotmat, pos=self.pos,
            rgb=ouc.BasicColor.ORANGE,
            alpha=ouc.ALPHA.TRANSPARENT)

    @property
    def file_path(self):
        return self._file_path

    @property
    def geom(self):
        return self._geom

    @property
    def aabb(self):
        if self._aabb is not None:
            return self._aabb
        geom = self.geom
        vs = geom.vs
        transformed = vs @ self.rotmat.T + self.pos
        min_corner = transformed.min(axis=0)
        max_corner = transformed.max(axis=0)
        return min_corner, max_corner
