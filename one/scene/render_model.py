import numpy as np
import one.utils.math as oum
import one.utils.decorator as oud
import one.geom.geometry as ogg
import one.viewer.device_buffer as ovdb

_device_buffer_cache = {}


class RenderModel:
    """
    rotmat and pos of model is for transforming local geometries
    it is intended to be immutable after creation.
    runtime pose updates must go through SceneObject.node.
    """

    def __init__(self,
                 geom=None,
                 rotmat=None,
                 pos=None,
                 rgb=None,
                 alpha=1.0,
                 shader=None,
                 **kwargs):
        if isinstance(geom, tuple):
            self.geom = ogg.gen_geom_from_raw(*geom)
        else:
            self.geom = geom
        self.shader = shader
        self._rgb = oum.ensure_rgb(rgb)
        self._vrgbs = kwargs.get("vrgbs", None)
        self._alpha = alpha
        self._rotmat = oum.ensure_rotmat(rotmat)
        self._pos = oum.ensure_pos(pos)
        # cached
        self._tf = oum.tf_from_rotmat_pos(self._rotmat, self._pos)
        self._dirty = True

    def clone(self):
        new = self.__class__(geom=self.geom,
                             rotmat=self._rotmat.copy(),
                             pos=self._pos.copy(),
                             rgb=self.rgb.copy(),
                             alpha=self.alpha,
                             shader=self.shader)
        return new

    def get_device_buffer(self):
        if self.geom.fs is None:
            # do not cache point cloud buffers
            if self._vrgbs is None:
                raise ValueError("PointCloudBuffer requires per-vertex rgb colors")
            return ovdb.PointCloudBuffer(
                self.geom.vs, self._vrgbs)
        gid = id(self.geom)
        if gid in _device_buffer_cache:
            return _device_buffer_cache[gid]
        else:
            buf = ovdb.MeshBuffer(
                self.geom.vs, self.geom.fs, self.geom.vns)
            _device_buffer_cache[gid] = buf
            return buf

    @property
    def rgb(self):
        return self._rgb.copy()

    @rgb.setter
    def rgb(self, rgb):
        self._rgb = oum.ensure_rgb(rgb)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def quat(self):
        # TODO cache?
        return oum.quat_from_rotmat(self._rotmat)

    @property
    def pos(self):
        return self._pos.copy()

    @pos.setter
    @oud.mark_dirty('_mark_dirty')
    def pos(self, pos):
        self._pos[:] = oum.ensure_pos(pos)

    @property
    def rotmat(self):
        return self._rotmat.copy()

    @rotmat.setter
    @oud.mark_dirty('_mark_dirty')
    def rotmat(self, rotmat):
        self._rotmat[:] = oum.ensure_rotmat(rotmat)

    @property
    @oud.lazy_update("_dirty", "_rebuild_tf")
    def tf(self):
        return self._tf.copy()

    @oud.mark_dirty('_mark_dirty')
    def set_rotmat_pos(self, rotmat, pos):
        self._rotmat[:] = oum.ensure_rotmat(rotmat)
        self._pos[:] = oum.ensure_pos(pos)

    def _rebuild_tf(self):
        if not self._dirty:
            return
        self._tf[:] = np.eye(4, dtype=np.float32)
        self._tf[:3, :3] = self._rotmat
        self._tf[:3, 3] = self._pos
        self._dirty = False

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
