import numpy as np
import wrs.utils.math as rm


class Model:

    def __init__(self, geometry=None, rotmat=None, pos=None, parent=None):
        self.geometry = geometry  # geometry.Geometry
        self.geometry_buffer = None
        self.parent = parent
        self.children = []
        if parent is not None:
            self.parent.children.append(self)
        # --- local transform ---
        self._rotmat = np.eye(3, dtype=np.float32) if rotmat is None else rotmat.astype(np.float32)
        self._pos = np.zeros(3, dtype=np.float32) if pos is None else pos.astype(np.float32)
        # --- world transform cache ---
        self._wd_rotmat = np.eye(3, dtype=np.float32)
        self._wd_pos = np.zeros(3, dtype=np.float32)
        self._wd_tfmat = np.eye(4, dtype=np.float32)
        # dirty flag
        self._dirty = True

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

    def set_pose(self, rotmat, pos):
        self._rotmat = rotmat.astype(np.float32)
        self._pos = pos.astype(np.float32)
        self._mark_dirty()

    def update(self):
        """
        Compute world_rotmat, world_p, world_mat4 recursively.
        """
        if not self._dirty:
            return
        if self.parent is None:
            self._wd_rotmat = self._rotmat.copy()
            self._wd_pos = self._pos.copy()
        else:
            self.parent.update()
            self._wd_rotmat = self.parent._wd_rotmat @ self._rotmat
            self._wd_pos = self.parent._wd_rotmat @ self._pos + self.parent._wd_pos
        # update world 4x4
        self._wd_tfmat = rm.tfmat_from_rotmat_pos(self._wd_rotmat, self._wd_pos)
        self._dirty = False

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            for child in self.children:
                child._mark_dirty()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos):
        self._pos = np.asarray(pos, dtype=np.float32)
        self._mark_dirty()

    @property
    def rotmat(self):
        return self._rotmat

    @rotmat.setter
    def rotmat(self, rotmat):
        self._rotmat = np.asarray(rotmat, dtype=np.float32)
        self._mark_dirty()

    @property
    def wd_pos(self):
        if self._dirty:
            self.update()
        return self._wd_pos

    @property
    def wd_rotmat(self):
        if self._dirty:
            self.update()
        return self._wd_rotmat

    @property
    def wd_tfmat(self):
        if self._dirty:
            self.update()
        return self._wd_tfmat