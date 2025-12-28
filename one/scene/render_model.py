import numpy as np
import one.utils.math as rm
import one.utils.decorator as deco
import one.utils.constant as const
import one.scene.geometry as geom


class RenderModel:
    """
    rotmat and pos of model is for transforming local geometries
    it is intended to be immutable after creation.
    runtime pose updates must go through SceneObject.node.
    """

    def __init__(self,
                 geometry=None,
                 rotmat=None,
                 pos=None,
                 rgb=None,
                 alpha=1.0,
                 shader=None):
        if isinstance(geometry, tuple):
            verts = geometry[0]
            faces = geometry[1] if len(geometry) > 1 else None
            per_vert_rgbs = geometry[2] if len(geometry) > 2 else None
            self.geometry = geom.Geometry(verts=verts,
                                          faces=faces,
                                          per_vert_rgbs=per_vert_rgbs)
        else:
            self.geometry = geometry
        self.rgb = const.BasicColor.DEFAULT if rgb is None else rgb
        self.alpha = alpha
        self.shader = shader
        self._rotmat = np.eye(3) if rotmat is None else rotmat
        self._pos = np.zeros(3) if pos is None else pos
        self._tfmat = rm.tfmat_from_rotmat_pos(self._rotmat, self._pos)

    def clone(self):
        new = RenderModel(
            geometry=self.geometry,
            rotmat=self._rotmat.copy(),
            pos=self._pos.copy(),
            rgb=self.rgb,
            alpha=self.alpha,
            shader=self.shader,
        )
        return new

    @property
    @deco.readonly_view
    def quat(self):
        return rm.quat_from_rotmat(self._rotmat)

    @property
    @deco.readonly_view
    def pos(self):
        return self._pos

    @property
    @deco.readonly_view
    def rotmat(self):
        return self._rotmat

    @property
    @deco.readonly_view
    def tfmat(self):
        return self._tfmat

    # @deco.mark_dirty('_mark_dirty')
    # def set_rotmat_pos(self, rotmat, pos):
    #     self._rotmat[:] = rotmat
    #     self._pos[:] = pos
    #     self._dirty = True

    # @pos.setter
    # @deco.mark_dirty('_mark_dirty')
    # def pos(self, pos):
    #     self._pos = pos

    # @rotmat.setter
    # @deco.mark_dirty('_mark_dirty')
    # def rotmat(self, rotmat):
    #     self._rotmat = rotmat

    # def _rebuild_tfmat(self):
    #     if not self._dirty:
    #         return
    #     self._tfmat[:] = np.eye(4, dtype=np.float32)
    #     self._tfmat[:3, :3] = self._rotmat
    #     self._tfmat[:3, 3] = self._pos
    #     self._dirty = False
    #
    # def _mark_dirty(self):
    #     if not self._dirty:
    #         self._dirty = True
