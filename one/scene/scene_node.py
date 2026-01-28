import one.utils.math as oum
import one.utils.decorator as oud


# TODO SceneNode for Scene Graph
# To be deleted since the system does not rely on a scene graph

class SceneNode:

    def __init__(self, rotmat=None, pos=None, parent=None):
        # local transform
        self._rotmat = oum.ensure_rotmat(rotmat)
        self._pos = oum.ensure_pos(pos)
        # cached
        self._tf = oum.tf_from_rotmat_pos(self._rotmat, self._pos)
        self._wd_tf = self._tf.copy()
        # dirty flag
        self._dirty = True
        # tree structure
        self.children = []
        self.parent = parent
        if parent is not None:
            parent.children.append(self)

    @oud.mark_dirty('_mark_dirty')
    def set_parent(self, new_parent):
        if self.parent is not None:
            try:
                self.parent.children.remove(self)
            except ValueError:
                raise Exception("Parent model does not have this model as a child.")
        self.parent = new_parent
        if new_parent is not None and self not in new_parent.children:
            new_parent.children.append(self)

    def _rebuild_tf(self):
        """Override the base class method to propagate to children."""
        if not self._dirty:
            return
        self._tf[:] = oum.tf_from_rotmat_pos(self._rotmat, self._pos)
        if self.parent is None:
            self._wd_tf[:3, :3] = self._rotmat
            self._wd_tf[:3, 3] = self._pos
        else:
            self.parent._rebuild_tf()
            p_wd_tf = self.parent._wd_tf
            self._wd_tf[:3, :3] = p_wd_tf[:3, :3] @ self._rotmat
            self._wd_tf[:3, 3] = p_wd_tf[:3, :3] @ self._pos + p_wd_tf[:3, 3]
        self._dirty = False

    @oud.mark_dirty('_mark_dirty')
    def set_rotmat_pos(self, rotmat=None, pos=None):
        self._rotmat[:] = oum.ensure_rotmat(rotmat)
        self._pos[:] = oum.ensure_pos(pos)

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
    @oud.lazy_update('_dirty', '_rebuild_tf')
    def tf(self):
        return self._tf.copy()

    @tf.setter
    @oud.mark_dirty('_mark_dirty')
    def tf(self, tf):
        tf = oum.ensure_tf(tf)
        self._rotmat[:] = tf[:3, :3]
        self._pos[:] = tf[:3, 3]

    @property
    @oud.lazy_update('_dirty', '_rebuild_tf')
    def wd_tf(self):
        return self._wd_tf.copy()

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            for c in self.children:
                c._mark_dirty()
