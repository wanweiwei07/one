import numpy as np
import one.utils.math as oum
import one.robots.base.mech_base as orbmb


class EndEffectorBase(orbmb.MechBase):

    def __init__(self, tcp_tf=None, is_free=True):
        super().__init__(is_free=is_free)
        self._tcp_tf = oum.ensure_tf(tcp_tf)

    def clone(self):
        new = super().clone()
        return new

    def set_tcp_rotmat_pos(self, rotmat=None, pos=None):
        self._tcp_tf[:3, :3] = oum.ensure_tf(rotmat)
        self._tcp_tf[:3, 3] = oum.ensure_pos(pos)

    @property
    def tcp_tfmat(self):
        return self._tcp_tf

    @tcp_tfmat.setter
    def tcp_tfmat(self, tfmat):
        self._tcp_tf[:] = oum.ensure_tf(tfmat)


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

    def _require_attr(self, name):
        if not hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} must define {name}")
