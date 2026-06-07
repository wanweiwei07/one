import numpy as np
import one.utils.math as oum


# End effectors are plain MechBase subclasses + a behavior mixin (GripperMixin /
# PointMixin / DexHandMixin). The working point is a registered tcp
# ('grasp_center' for grippers, 'tcp' for point tools, '<primitive>_center' for
# dexterous hands), so there is no EndEffectorBase: positioning an EE goes
# through cross-object ik, e.g.
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


class DexHandMixin:
    """Behavior for a multi-finger dexterous hand (a MechBase EE).

    Where a parallel gripper closes via a single scalar ``set_jaw_width``, a hand
    closes via *named grasp synergies*: each primitive maps one scalar
    ``amount`` in [0, 1] (open -> closed) onto a coordinated set of finger joint
    targets. The concrete hand supplies the data; this mixin supplies the verbs.

    The concrete hand must define:
    - ``self.grasp_synergies``: ``{primitive_name: {joint_name: closed_q}}`` --
      the joint targets at ``amount=1.0`` (fully closed) for each primitive.
      Joints absent from a primitive stay where they are. Typical primitives:
      ``'pinch'`` (thumb+index), ``'tripod'`` (thumb+index+middle),
      ``'power'`` (whole-hand envelope).
    - center tcps named ``'<primitive>_center'`` for the primitives it wants to
      position via ik / the ``*_at`` helpers (e.g. ``'pinch_center'``,
      ``'power_grasp_center'``).
    """

    # primitive -> the center-tcp name used to position it. 'power' maps to
    # 'power_grasp_center'; pinch-family primitives share 'pinch_center'.
    _CENTER_TCP = {
        'pinch': 'pinch_center',
        'tripod': 'pinch_center',
        'power': 'power_grasp_center',
    }

    def _jnt_qidx(self, name):
        cache = getattr(self, '_jnt_qidx_cache', None)
        if cache is None:
            cache = {j.name: i for i, j in enumerate(self.structure.jnts)}
            self._jnt_qidx_cache = cache
        return cache[name]

    def _apply_synergy(self, primitive, amount):
        """Drive the primitive's joints to ``amount`` of their closed targets,
        leaving every other joint untouched."""
        amount = float(np.clip(amount, 0.0, 1.0))
        targets = self.grasp_synergies[primitive]
        qs = self.qs.copy()
        for jname, closed_q in targets.items():
            qs[self._jnt_qidx(jname)] = closed_q * amount
        self.fk(qs)

    # ---- shape primitives (no positioning) -----------------------------
    def open_hand(self):
        """Fully extend every finger (all joints to 0)."""
        self.fk(qs=np.zeros(self.ndof, dtype=np.float32))

    def pinch(self, amount=1.0):
        """Precision pinch: thumb opposes index."""
        self._apply_synergy('pinch', amount)

    def tripod(self, amount=1.0):
        """Tripod grip: thumb opposes index + middle (more stable than pinch)."""
        self._apply_synergy('tripod', amount)

    def power_grasp(self, amount=1.0):
        """Whole-hand enveloping (power) grasp."""
        self._apply_synergy('power', amount)

    # ---- attach / detach an object -------------------------------------
    def grasp(self, child, primitive='power', amount=1.0):
        """Close ``primitive`` and rigidly mount ``child`` on the hand."""
        self._apply_synergy(primitive, amount)
        parent = self.runtime_root_lnk
        loc_tf = np.linalg.inv(parent.tf) @ child.tf
        self.mount(child, parent, loc_tf)

    def release(self, child, reopen=True):
        if reopen:
            self.open_hand()
        self.unmount(child)

    # ---- position a center tcp, then close -----------------------------
    def grasp_at(self, tgt_pos, tgt_rotmat, primitive='power', amount=1.0,
                 center=None):
        """Move the primitive's center tcp to the target pose, then close to
        ``amount``. Returns the resulting hand base tf."""
        center = center or self._CENTER_TCP[primitive]
        loc_tf = self.tcp(center).loc_tf
        tgt_tf = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat)
        base_tf = tgt_tf @ np.linalg.inv(loc_tf)
        self.set_pos_rotmat(base_tf[:3, 3], base_tf[:3, :3])
        self._apply_synergy(primitive, amount)
        return base_tf

    def pinch_at(self, tgt_pos, tgt_rotmat, amount=1.0):
        return self.grasp_at(tgt_pos, tgt_rotmat, 'pinch', amount)

    def power_grasp_at(self, tgt_pos, tgt_rotmat, amount=1.0):
        return self.grasp_at(tgt_pos, tgt_rotmat, 'power', amount)
