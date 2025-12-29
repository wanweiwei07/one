import numpy as np
import one.utils.math as rm
import one.utils.decorator as deco
import one.utils.constant as const
import one.scene.geometry as geom


class RenderModel:
    """
    rotmat and pos of rendermodel is for transforming local geometries
    it is intended to be immutable after creation.
    runtime pose updates must go through SceneObject.
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
        self._tfmat = rm.tfmat_from_rotmat_pos(rotmat, pos)

    def clone(self):
        new = RenderModel(geometry=self.geometry,
                          rotmat=self.rotmat,
                          pos=self.pos,
                          rgb=self.rgb,
                          alpha=self.alpha,
                          shader=self.shader)
        return new

    @property
    @deco.readonly_view
    def quat(self):
        return rm.quat_from_rotmat(self.rotmat)

    @property
    @deco.readonly_view
    def pos(self):
        return self._tfmat[:3, 3]

    @property
    @deco.readonly_view
    def rotmat(self):
        return self._tfmat[:3, :3]

    @property
    @deco.readonly_view
    def tfmat(self):
        return self._tfmat
