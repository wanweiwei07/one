import one.utils.math as rm
import one.utils.decorator as deco


class SceneNode:

    def __init__(self, rotmat=None, pos=None, parent=None):
        # local transform
        self._rotmat = rm.np.eye(3, dtype=rm.np.float32) if rotmat is None else rotmat.astype(rm.np.float32)
        self._pos = rm.np.zeros(3, dtype=rm.np.float32) if pos is None else pos.astype(rm.np.float32)
        # world transform cache
        self._wd_rotmat = rm.np.eye(3, dtype=rm.np.float32)
        self._wd_pos = rm.np.zeros(3, dtype=rm.np.float32)
        self._wd_tfmat = rm.np.eye(4, dtype=rm.np.float32)
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
            self._wd_rotmat = self._rotmat.copy()
            self._wd_pos = self._pos.copy()
        else:
            self.parent.update()
            self._wd_rotmat = self.parent._wd_rotmat @ self._rotmat
            self._wd_pos = self.parent._wd_rotmat @ self._pos + self.parent._wd_pos
        # update world mat4
        self._wd_tfmat = rm.tfmat_from_rotmat_pos(self._wd_rotmat, self._wd_pos)
        self._dirty = False

    def set_rotmat_pos(self, rotmat, pos):
        self._rotmat = rotmat.astype(rm.np.float32)
        self._pos = pos.astype(rm.np.float32)
        self._mark_dirty()

    def set_tfmat(self, tfmat):
        tfmat = tfmat.astype(rm.np.float32)
        self._rotmat = tfmat[:3, :3]
        self._pos = tfmat[:3, 3]
        self._mark_dirty()

    @property
    @deco.readonly_view
    def pos(self):
        return self._pos

    @pos.setter
    @deco.mark_dirty('_mark_dirty')
    def pos(self, value):
        self._pos = rm.np.asarray(value, dtype=rm.np.float32)

    @property
    @deco.readonly_view
    def rotmat(self):
        return self._rotmat

    @rotmat.setter
    @deco.mark_dirty('_mark_dirty')
    def rotmat(self, value):
        self._rotmat = rm.np.asarray(value, dtype=rm.np.float32)

    @property
    @deco.lazy_update('_dirty', 'update')
    @deco.readonly_view
    def wd_pos(self):
        return self._wd_pos

    @property
    @deco.lazy_update('_dirty', 'update')
    @deco.readonly_view
    def wd_rotmat(self):
        return self._wd_rotmat

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
