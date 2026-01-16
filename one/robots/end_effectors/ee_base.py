import numpy as np
import one.utils.math as oum
import one.robots.base.mech_base as orbmb


class EndEffectorBase(orbmb.MechBase):

    def __init__(self, tcp_tf=None, is_free=True):
        super().__init__(is_free=is_free)
        self._tcp_tf = oum.ensure_tf(tcp_tf)

    def clone(self):
        new = super().clone()
        new._tcp_tf = self._tcp_tf.copy()
        return new

    def set_tcp_rotmat_pos(self, rotmat=None, pos=None):
        self._tcp_tf[:3, :3] = oum.ensure_tf(rotmat)
        self._tcp_tf[:3, 3] = oum.ensure_pos(pos)

    @property
    def tcp_tf(self):
        return self._tcp_tf

    @tcp_tf.setter
    def tcp_tf(self, value):
        self._tcp_tf[:] = oum.ensure_tf(value)


class GripperMixin:

    def open(self):
        self.set_jaw_width(self.jaw_range[1])

    def close(self):
        self.set_jaw_width(self.jaw_range[0])

    def grasp(self, child, jaw_width=None):
        """
        :param jaw_width:
        :param child: scene.SceneObject
        :return:
        """
        jaw_width = self.jaw_range[0] if jaw_width is None else jaw_width
        self.set_jaw_width(jaw_width)
        parent_tf = self.runtime_root_lnk.tf
        engage_tf = np.linalg.inv(parent_tf).dot(child.tf)
        self.mount(child, self.runtime_root_lnk, engage_tf)

    def release(self, child, jaw_width=None):
        jaw_width = self.jaw_range[1] if jaw_width is None else jaw_width
        self.set_jaw_width(jaw_width)  # TODO jaw_width should increase
        self.unmount(child)

    def set_jaw_width(self, width):
        raise NotImplementedError

    def grip_at(self, tgt_pos, tgt_rotmat, tgt_jaw_width):
        """
        Move TCP to target pose, set jaw width, return base tf.
        :param tgt_pos: (3,)
        :param tgt_rotmat: (3,3)
        :param tgt_jaw_width: float
        :return: base_tf (4,4)
        """
        if (tgt_jaw_width < self.jaw_range[0] or
                tgt_jaw_width > self.jaw_range[1]):
            raise ValueError(f"jaw_width {tgt_jaw_width}"
                             f" out of range {self.jaw_range}")
        tgt_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        base_tf = tgt_tf @ np.linalg.inv(self.tcp_tf)
        self.set_rotmat_pos(base_tf[:3, :3], base_tf[:3, 3])
        self.set_jaw_width(tgt_jaw_width)
        return base_tf

    def _require_attr(self, name):
        if not hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} must define {name}")

