import numpy as np
import one.robot_sim.base.robot_base as rbase


class EndEffectorBase(rbase.RobotBase):

    def __init__(self):
        super().__init__()
        self._is_engaged = False

    @property
    def is_engaged(self):
        return self._is_engaged

    @property
    def toggle_render_collision(self):
        return self.kin_state.runtime_links[0].toggle_render_collision

    @toggle_render_collision.setter
    def toggle_render_collision(self, flag=True):
        for link in self.kin_state.runtime_links:
            link.toggle_render_collision = flag


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
        parent_tfmat = self.get_link_wd_tfmat(self.structure.links[0])
        engage_tfmat = np.linalg.inv(parent_tfmat).dot(child.tfmat)
        self.mount(child, self.structure.links[0], engage_tfmat=engage_tfmat)

    def release(self, child, jaw_width=None):
        jaw_width = self.jaw_range[1] if jaw_width is None else jaw_width
        self.set_jaw_width(jaw_width)  # TODO jaw_width should increase
        self.unmount(child)

    def set_jaw_width(self, width):
        raise NotImplementedError

    def _require_attr(self, name):
        if not hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} must define {name}")
