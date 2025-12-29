import numpy as np
import one.utils.math as rm
import one.utils.decorator as deco


# TODO SceneNode for Scene Graph
# To be deleted since the system does not rely on a scene graph

class SceneNode:

    def __init__(self, rotmat=None, pos=None, parent=None):
        # local transform
        self._tfmat = rm.tfmat_from_rotmat_pos(rotmat, pos)
        # world transform cache
        self._wd_tfmat = self._tfmat.copy()
        # dirty flag
        self._dirty = True
        # tree structure
        self.children = []
        self.parent = parent
        if parent is not None:
            parent.children.append(self)

    def set_parent(self, new_parent):
        if self.parent is not None:
            try:
                self.parent.children.remove(self)
            except ValueError:
                raise Exception("Parent model does not have this model as a child.")
        self.parent = new_parent
        if new_parent is not None:
            new_parent.children.append(self)
        self._mark_dirty()

    def update(self):
        """Override the base class method to propagate to children."""
        if not self._dirty:
            return
        if self.parent is None:
            self._wd_tfmat = self._tfmat.copy()
        else:
            self.parent.update()
            self._wd_tfmat = self.parent._wd_tfmat @ self._tfmat
        self._dirty = False

    def set_rotmat_pos(self, rotmat, pos):
        self._tfmat = rm.tfmat_from_rotmat_pos(rotmat, pos)
        self._mark_dirty()

    @property
    @deco.readonly_view
    def quat(self):
        return rm.quat_from_rotmat(self._tfmat[:3, :3])

    @property
    @deco.readonly_view
    def pos(self):
        return self._tfmat[:3, 3]

    @pos.setter
    @deco.mark_dirty('_mark_dirty')
    def pos(self, value):
        self._tfmat[:3, 3] = rm.ensure_pos(value)

    @property
    @deco.readonly_view
    def rotmat(self):
        return self._tfmat[:3, :3]

    @rotmat.setter
    @deco.mark_dirty('_mark_dirty')
    def rotmat(self, value):
        self._tfmat[:3, :3] = rm.ensure_rotmat(value)

    @property
    @deco.readonly_view
    def tfmat(self):
        return self._tfmat

    @tfmat.setter
    @deco.mark_dirty('_mark_dirty')
    def tfmat(self, value):
        self._tfmat = rm.ensure_tfmat(value)

    @property
    @deco.lazy_update('_dirty', 'update')
    @deco.readonly_view
    def wd_pos(self):
        return self._wd_tfmat[:3, 3]

    @property
    @deco.lazy_update('_dirty', 'update')
    @deco.readonly_view
    def wd_rotmat(self):
        return self._wd_tfmat[:3, :3]

    @property
    @deco.lazy_update('_dirty', 'update')
    @deco.readonly_view
    def wd_tfmat(self):
        return self._wd_tfmat

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            for c in self.children:
                c._mark_dirty()