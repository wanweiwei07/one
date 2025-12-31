import numpy as np
import one.utils.math as rm
import one.utils.decorator as deco


# TODO SceneNode for Scene Graph
# To be deleted since the system does not rely on a scene graph

class SceneNode:

    def __init__(self, rotmat=None, pos=None, parent=None):
        # local transform
        self._rotmat = rm.ensure_rotmat(rotmat)
        self._pos = rm.ensure_pos(pos)
        # cached
        self._tfmat = rm.tfmat_from_rotmat_pos(self._rotmat, self._pos)
        self._wd_tfmat = self._tfmat.copy()
        # dirty flag
        self._dirty = True
        # tree structure
        self.children = []
        self.parent = parent
        if parent is not None:
            parent.children.append(self)

    @deco.mark_dirty('_mark_dirty')
    def set_parent(self, new_parent):
        if self.parent is not None:
            try:
                self.parent.children.remove(self)
            except ValueError:
                raise Exception("Parent model does not have this model as a child.")
        self.parent = new_parent
        if new_parent is not None and self not in new_parent.children:
            new_parent.children.append(self)

    def _rebuild_tfmat(self):
        """Override the base class method to propagate to children."""
        if not self._dirty:
            return
        self._tfmat[:] = rm.tfmat_from_rotmat_pos(self._rotmat, self._pos)
        if self.parent is None:
            self._wd_tfmat[:3,:3] = self._rotmat
            self._wd_tfmat[:3,3] = self._pos
        else:
            self.parent._rebuild_tfmat()
            p_wd_tf = self.parent._wd_tfmat
            self._wd_tfmat[:3,:3] = p_wd_tf[:3, :3] @ self._rotmat
            self._wd_tfmat[:3,3] = p_wd_tf[:3, :3] @ self._pos + p_wd_tf[:3, 3]
        self._dirty = False

    @deco.mark_dirty('_mark_dirty')
    def set_rotmat_pos(self, rotmat, pos):
        self._rotmat[:] = rm.ensure_rotmat(rotmat)
        self._pos[:] = rm.ensure_pos(pos)

    @property
    def quat(self):
        # TODO cache?
        return rm.quat_from_rotmat(self._rotmat)

    @property
    def pos(self):
        return self._pos.copy()

    @pos.setter
    @deco.mark_dirty('_mark_dirty')
    def pos(self, pos):
        self._pos[:] = rm.ensure_pos(pos)

    @property
    def rotmat(self):
        return self._rotmat.copy()

    @rotmat.setter
    @deco.mark_dirty('_mark_dirty')
    def rotmat(self, rotmat):
        self._rotmat[:] = rm.ensure_rotmat(rotmat)

    @property
    @deco.lazy_update('_dirty', '_rebuild_tfmat')
    def tfmat(self):
        return self._tfmat.copy()

    @tfmat.setter
    @deco.mark_dirty('_mark_dirty')
    def tfmat(self, tfmat):
        tfmat = rm.ensure_tfmat(tfmat)
        self._rotmat[:] = tfmat[:3, :3]
        self._pos[:] = tfmat[:3, 3]

    @property
    @deco.lazy_update('_dirty', '_rebuild_tfmat')
    def wd_tfmat(self):
        return self._wd_tfmat.copy()

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            for c in self.children:
                c._mark_dirty()
