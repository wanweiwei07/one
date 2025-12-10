import numpy as np
import one.utils.decorator as deco
import one.utils.constant as const
import one.scene.geometry as geom
import one.viewer.device_buffer as dvb


class Model:

    def __init__(
            self, geometry=None, rotmat=None, pos=None, rgb=None, alpha=1.0, shader=None
    ):
        if isinstance(geometry, tuple):
            verts = geometry[0]
            faces = geometry[1] if len(geometry) > 1 else None
            per_vert_rgbs = geometry[2] if len(geometry) > 2 else None
            self.geometry = geom.Geometry(
                verts=verts, faces=faces, per_vert_rgbs=per_vert_rgbs
            )
        else:
            self.geometry = geometry
        self.rgb = const.BasicColor.DEFAULT if rgb is None else rgb
        self.alpha = alpha
        self.shader = shader
        self._rotmat = np.eye(3) if rotmat is None else rotmat
        self._pos = np.zeros(3) if pos is None else pos
        self._tfmat = np.eye(4, dtype=np.float32)
        self._dirty = True

    @deco.mark_dirty('_mark_dirty')
    def set_rotmat_pos(self, rotmat, pos):
        self._rotmat[:] = rotmat
        self._pos[:] = pos
        self._dirty = True

    def get_device_buffer(self):
        if self.geometry.device_buffer is None:
            if self.geometry.faces is None:
                self.geometry.device_buffer = dvb.PointCloudBuffer(
                    self.geometry.verts, self.geometry.per_vert_rgbs
                )
            else:
                self.geometry.device_buffer = dvb.MeshBuffer(
                    self.geometry.verts, self.geometry.faces, self.geometry.face_normals
                )
        return self.geometry.device_buffer

    def clone(self, keep_transform=True):
        new = Model(
            geometry=self.geometry,
            rotmat=(
                self._rotmat.copy() if keep_transform else np.eye(3, dtype=np.float32)
            ),
            pos=self._pos.copy() if keep_transform else np.zeros(3, dtype=np.float32),
            rgb=self.rgb,
            alpha=self.alpha,
            shader=self.shader,
        )
        return new

    @property
    def pos(self):
        return self._pos

    @pos.setter
    @deco.mark_dirty('_mark_dirty')
    def pos(self, pos):
        self._pos = pos

    @property
    def rotmat(self):
        return self._rotmat

    @rotmat.setter
    @deco.mark_dirty('_mark_dirty')
    def rotmat(self, rotmat):
        self._rotmat = rotmat

    @property
    @deco.readonly_view
    @deco.lazy_update('_dirty', '_rebuild_tfmat')
    def tfmat(self):
        return self._tfmat

    def _rebuild_tfmat(self):
        if not self._dirty:
            return
        self._tfmat[:] = np.eye(4, dtype=np.float32)
        self._tfmat[:3, :3] = self._rotmat
        self._tfmat[:3, 3] = self._pos
        self._dirty = False

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
