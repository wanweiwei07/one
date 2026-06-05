import numpy as np
import one.utils.math as oum


# End effectors are plain MechBase subclasses + a behavior mixin (GripperMixin /
# PointMixin). The working point is a registered tcp ('grasp_center' for
# grippers, 'tcp' for point tools), so there is no EndEffectorBase: positioning
# an EE goes through cross-object ik, e.g.
#   arm.ik(pos, rotmat, tcp=gripper.tcp('grasp_center'))


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
        loc_tf = np.linalg.inv(parent_tf).dot(child.tf)
        self.mount(child, self.runtime_root_lnk, loc_tf)

    def release(self, child, jaw_width=None):
        jaw_width = self.jaw_range[1] if jaw_width is None else jaw_width
        self.set_jaw_width(jaw_width)  # TODO jaw_width should increase
        self.unmount(child)

    def set_jaw_width(self, width):
        raise NotImplementedError

    def _grasp_loc_tf(self):
        """Grasp-center offset relative to the gripper root link (the tcp named
        'grasp_center' that every gripper registers)."""
        return self.tcp('grasp_center').loc_tf

    def grip_at(self, tgt_pos, tgt_rotmat, tgt_jaw_width):
        """
        Move grasp center to target pose, set jaw width, return base tf.
        :param tgt_pos: (3,)
        :param tgt_rotmat: (3,3)
        :param tgt_jaw_width: float
        :return: base_tf (4,4)
        """
        if (tgt_jaw_width < self.jaw_range[0] or
                tgt_jaw_width > self.jaw_range[1]):
            raise ValueError(f"jaw_width {tgt_jaw_width}"
                             f" out of range {self.jaw_range}")
        tgt_tf = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat)
        base_tf = tgt_tf @ np.linalg.inv(self._grasp_loc_tf())
        self.set_pos_rotmat(base_tf[:3, 3], base_tf[:3, :3])
        self.set_jaw_width(tgt_jaw_width)
        return base_tf

    def _require_attr(self, name):
        if not hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} must define {name}")


class PointMixin:

    def _tip_loc_tf(self):
        """Tool-point offset relative to the root link (the tcp named 'tip'
        that every point tool registers)."""
        return self.tcp('tip').loc_tf

    def activate(self):
        self._set_activation_state(True)

    def deactivate(self):
        self._set_activation_state(False)

    def touch_at(self, tgt_pos, tgt_rotmat, activate=False):
        """Move tool tip to target pose, return base tf"""
        tgt_tf = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat)
        base_tf = tgt_tf @ np.linalg.inv(self._tip_loc_tf())
        self.set_pos_rotmat(base_tf[:3, 3], base_tf[:3, :3])
        if activate:
            self.activate()
        return base_tf

    def attach(self, child, offset_tf=None):
        """Attach object to tool"""
        if not self.is_activated:
            raise RuntimeError("Cannot attach: end effector not activated")
        parent_tf = self.runtime_root_lnk.tf @ self._tip_loc_tf()
        if offset_tf is None:
            loc_tf = np.linalg.inv(parent_tf) @ child.tf
        else:
            loc_tf = np.linalg.inv(parent_tf @ offset_tf) @ child.tf
        self.mount(child, self.runtime_root_lnk, loc_tf)

    def detach(self, child):
        """Detach object from tool."""
        self.unmount(child)
        self.deactivate()

    def _set_activation_state(self, state):
        """Override in subclass to implement actual activation logic."""
        self._is_activated = state

    @property
    def is_activated(self):
        return getattr(self, '_is_activated', False)
