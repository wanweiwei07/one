import numpy as np
import one.robots.base.mech_base as orb


class EndEffectorBase(orb.MechBase):

    def __init__(self):
        super().__init__()
        self._is_engaged = False

    def clone(self):
        new = super().clone()
        new._is_engaged = self._is_engaged
        return new

    @property
    def is_engaged(self):
        return self._is_engaged


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
        parent_tfmat = self.get_lnk_ref_tfmat(self.structure.lnks[0])
        engage_tfmat = np.linalg.inv(parent_tfmat).dot(child.tfmat)
        self.mount(child, self.structure.lnks[0], engage_tfmat=engage_tfmat)

    def release(self, child, jaw_width=None):
        jaw_width = self.jaw_range[1] if jaw_width is None else jaw_width
        self.set_jaw_width(jaw_width)  # TODO jaw_width should increase
        self.unmount(child)

    def set_jaw_width(self, width):
        raise NotImplementedError

    def _require_attr(self, name):
        if not hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} must define {name}")
