from collections import namedtuple

import numpy as np
import one.utils.math as oum
import one.robots.base.tcp as orbt


# A dexterous-hand grasp definition, resolved to concrete (prefixed) joint /
# link names by the hand's ``grasp_spec``:
#   preshape: {joint: q} held FIXED at every closure (e.g. the thumb swing that
#     brings it across to oppose the fingers -- must not scale with amount or it
#     under-swings when the grip is wide).
#   closing:  {joint: q} the flexion that shuts the grip, SCALED by amount.
#   tuck:     {joint: q} non-participating fingers, held out of the way.
#   thumb_pad / opp_pads: distal link names of the moving thumb 'jaw' and the
#     opposing finger pad(s). opp_pads is None when the grasp is not a
#     parallel-jaw opposition (e.g. a power envelope) and so cannot be presented
#     to antipodal.
DexGraspSpec = namedtuple(
    'DexGraspSpec', ['preshape', 'closing', 'tuck', 'thumb_pad', 'opp_pads'])


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
    closes via *named grasp primitives* (pinch / tripod / power): each maps one
    scalar ``amount`` in [0, 1] (open -> closed) onto a coordinated finger pose.
    The concrete hand supplies the data; this mixin supplies the verbs.

    The concrete hand must define:
    - ``grasp_spec(primitive) -> DexGraspSpec``: the preshape / closing / tuck
      joint targets (and pad links) for a primitive, in this hand's joint/link
      names. One definition feeds BOTH the shape primitives and the parallel-jaw
      planning view, so there is no second copy of the grasp model.
    - center tcps named ``'<primitive>_center'`` for the primitives it wants to
      position via ik / the ``*_at`` helpers (e.g. ``'pinch_center'``,
      ``'power_center'``).

    A pinch/tripod (a thumb-vs-finger(s) opposition) can also be presented to the
    parallel-jaw grasp planner ``antipodal`` via ``spawn_jaw(primitive)``, which
    returns a calibrated, immutably-bound clone exposing the gripper interface
    (jaw_range / set_jaw_width / open_dir_at / grasp_center_at / eval_grasp_tcp /
    grip_at).
    """

    # closure samples for the jaw calibration sweep, and the representative
    # closure at which the 'grasp_center' tcp / reference open_dir are taken.
    _CAL_AMOUNTS = np.linspace(0.3, 1.0, 20)
    _REF_AMOUNT = 0.7

    def grasp_spec(self, primitive):
        """Return the DexGraspSpec for ``primitive`` (concrete joint/link names).
        Implemented by the concrete hand from its naming convention."""
        raise NotImplementedError

    def __getattr__(self, name):
        # Only fires when normal lookup fails -- i.e. the hand has NOT been bound
        # as a parallel jaw via spawn_jaw (the jaw attrs are created there). Turn
        # the otherwise cryptic AttributeError antipodal raises into a fix-it
        # message. Bound hands have these attrs, so this never runs for them.
        if name in ('contact_pattern', 'jaw_range', 'open_dir', '_spec'):
            raise AttributeError(
                f"{type(self).__name__} is a dexterous hand, not bound as a "
                f"parallel jaw. Call hand.spawn_jaw('pinch') (or 'tripod') and "
                f"pass that to antipodal, e.g. "
                f"antipodal(hand.spawn_jaw('pinch'), obj).")
        raise AttributeError(name)

    def _jnt_qidx(self, name):
        cache = getattr(self, '_jnt_qidx_cache', None)
        if cache is None:
            cache = {j.name: i for i, j in enumerate(self.structure.jnts)}
            self._jnt_qidx_cache = cache
        return cache[name]

    def _pose_grasp(self, spec, amount):
        """Pose the hand at ``amount`` in [0, 1] per ``spec``: preshape held
        fixed, closing scaled by amount, the rest tucked. Only independent joints
        are driven; mimic joints follow in fk."""
        amount = float(np.clip(amount, 0.0, 1.0))
        qs = np.zeros(self._compiled.n_jnts, dtype=np.float32)
        for j, v in spec.preshape.items():
            qs[self._jnt_qidx(j)] = v
        for j, v in spec.closing.items():
            qs[self._jnt_qidx(j)] = v * amount
        for j, v in spec.tuck.items():
            qs[self._jnt_qidx(j)] = v
        self.fk(qs=qs)

    # ---- shape primitives (no positioning) -----------------------------
    def open_hand(self):
        """Fully extend every finger (all joints to 0)."""
        self.fk(qs=np.zeros(self.ndof, dtype=np.float32))

    def pinch(self, amount=1.0):
        """Precision pinch: thumb opposes index."""
        self._pose_grasp(self.grasp_spec('pinch'), amount)

    def tripod(self, amount=1.0):
        """Tripod grip: thumb opposes index + middle (more stable than pinch)."""
        self._pose_grasp(self.grasp_spec('tripod'), amount)

    def power_grasp(self, amount=1.0):
        """Whole-hand enveloping (power) grasp."""
        self._pose_grasp(self.grasp_spec('power'), amount)

    # ---- attach / detach an object -------------------------------------
    def grasp(self, child, primitive='power', amount=1.0):
        """Close ``primitive`` and rigidly mount ``child`` on the hand."""
        self._pose_grasp(self.grasp_spec(primitive), amount)
        parent = self.runtime_root_lnk
        loc_tf = np.linalg.inv(parent.tf) @ child.tf
        self.mount(child, parent, loc_tf)

    def release(self, child, reopen=True):
        if reopen:
            self.open_hand()
        self.unmount(child)

    # ---- position a center tcp, then close -----------------------------
    def grasp_at(self, tgt_pos, tgt_rotmat, primitive='power', amount=1.0):
        """Move the primitive's center tcp to the target pose, then close to
        ``amount``. The center tcp follows the convention ``f'{primitive}_center'``
        (e.g. 'pinch' -> 'pinch_center', 'power' -> 'power_center'). Returns the
        resulting hand base tf."""
        loc_tf = self.tcp(f'{primitive}_center').loc_tf
        tgt_tf = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat)
        base_tf = tgt_tf @ np.linalg.inv(loc_tf)
        self.set_pos_rotmat(base_tf[:3, 3], base_tf[:3, :3])
        self._pose_grasp(self.grasp_spec(primitive), amount)
        return base_tf

    def pinch_at(self, tgt_pos, tgt_rotmat, amount=1.0):
        return self.grasp_at(tgt_pos, tgt_rotmat, 'pinch', amount)

    def power_grasp_at(self, tgt_pos, tgt_rotmat, amount=1.0):
        return self.grasp_at(tgt_pos, tgt_rotmat, 'power', amount)

    # ---- parallel-jaw view for antipodal planning ----------------------
    def spawn_jaw(self, primitive):
        """Return a CLONE of this hand immutably bound to ``primitive`` as a
        parallel-jaw target for ``antipodal``. It calibrates the pad gap, the
        opposition axis and the grasp center against closure ONCE, then exposes
        the gripper interface antipodal consumes. The source hand is untouched::

            grasps = antipodal(hand.spawn_jaw('pinch'), obj)

        Raises if ``primitive`` is not a parallel-jaw opposition (e.g. 'power').
        """
        spec = self.grasp_spec(primitive)
        if spec.opp_pads is None:
            raise ValueError(
                f"grasp '{primitive}' is not a parallel-jaw opposition (no "
                f"opposing pads); it cannot be presented to antipodal")
        new = self.clone()
        new._grasp = primitive
        new._spec = spec
        new._calibrate_jaw()
        return new

    def _world_vs(self, lnk_name):
        """World-frame vertices of a link's first visual mesh."""
        c = self._compiled
        lnk = self.runtime_lnks[c.lidx_map[self.structure.lnk_map[lnk_name]]]
        v = lnk.visuals[0]
        vs = np.asarray(v.geom.vs, dtype=np.float64).reshape(-1, 3)
        m = (lnk.tf @ v._tf).astype(np.float64)
        return vs @ m[:3, :3].T + m[:3, 3]

    def _pad_contacts(self, spec, amount):
        """The two opposing pad contact points (world frame) at ``amount``: the
        thumb pad, and the opposing 'jaw'. With one opposing finger this is the
        closest thumb<->finger vertex pair; with several (tripod) it is the mean
        of each finger's closest pair, so the opposing pad sits between them."""
        self._pose_grasp(spec, amount)
        thumb = self._world_vs(spec.thumb_pad)
        t_pts, o_pts = [], []
        for pad in spec.opp_pads:
            b = self._world_vs(pad)
            d = np.linalg.norm(thumb[:, None, :] - b[None, :, :], axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            t_pts.append(thumb[i])
            o_pts.append(b[j])
        return np.mean(t_pts, axis=0), np.mean(o_pts, axis=0)

    def _calibrate_jaw(self):
        """Sweep closure and measure (gap, opposition axis, center) at the pad
        contact, in the hand-base frame. The pinch is a curling motion, so all
        three are nonlinear in amount; the tables let set_jaw_width / open_dir_at
        / grasp_center_at invert and interpolate them per object."""
        spec = self._spec
        b_inv = np.linalg.inv(self.runtime_root_lnk.tf)
        gaps, dirs, mids = [], [], []
        for a in self._CAL_AMOUNTS:
            pad_t, pad_o = self._pad_contacts(spec, a)
            ct = b_inv[:3, :3] @ pad_t + b_inv[:3, 3]
            co = b_inv[:3, :3] @ pad_o + b_inv[:3, 3]
            d = co - ct
            gaps.append(np.linalg.norm(d))
            dirs.append(d / (np.linalg.norm(d) + oum.eps))
            mids.append((ct + co) * 0.5)
        gaps = np.array(gaps, dtype=np.float32)
        # keep only the monotonic-decreasing prefix: near full closure the
        # curling fingers slide past each other and the closest-point gap gets
        # noisy / non-monotonic, which would break the gap <-> amount inversion.
        n = 1
        while n < len(gaps) and gaps[n] < gaps[n - 1] - 1e-4:
            n += 1
        amts = self._CAL_AMOUNTS
        self._cal_amount = amts[:n]
        self._cal_gap = gaps[:n]
        self._cal_dir = np.array(dirs[:n], dtype=np.float32)
        self._cal_mid = np.array(mids[:n], dtype=np.float32)
        self.jaw_range = np.array([self._cal_gap[-1], self._cal_gap[0]],
                                  dtype=np.float32)
        # open_dir table: drop amounts where the pads ~touch (axis ill-defined)
        ok = self._cal_gap >= 0.005
        self._od_amount = self._cal_amount[ok]
        self._od_dir = self._cal_dir[ok]
        ref_idx = int(np.argmin(np.abs(self._cal_amount - self._REF_AMOUNT)))
        # 'grasp_center' tcp at a representative closure -- antipodal reads it
        # for the pre-grasp retreat distance. Keep the primitive tcp rotation so
        # the approach frame matches the hand's actual grasp frame.
        loc = self._grasp_center_loc_tf(self._cal_gap[ref_idx])
        if 'grasp_center' in self._tcps:
            self.tcp('grasp_center').set_loc_tf(loc)
        else:
            self.add_tcp('grasp_center', self.runtime_root_lnk, loc)
        self.open_dir = self._cal_dir[ref_idx]
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self._jaw_w = float(self.jaw_range[1])
        self.set_jaw_width(self._jaw_w)

    def _amount_for(self, jaw_width):
        # invert the pad-gap(amount) calibration (gap decreases as amount rises)
        w = np.clip(jaw_width, self.jaw_range[0], self.jaw_range[1])
        return np.interp(w, self._cal_gap[::-1], self._cal_amount[::-1])

    def open_dir_at(self, jaw_width):
        """Per-grasp opposition axis: the thumb-pad -> opposing-pad direction at
        the closure that grips a ``jaw_width``-wide object (hand-base frame).
        Scalar in -> (3,); array in -> (N, 3)."""
        amount = np.clip(self._amount_for(jaw_width),
                         self._od_amount[0], self._od_amount[-1])
        amount = np.atleast_1d(amount)
        od = np.stack([np.interp(amount, self._od_amount, self._od_dir[:, k])
                       for k in range(3)], axis=1)
        od = od / (np.linalg.norm(od, axis=1, keepdims=True) + oum.eps)
        od = od.astype(np.float32)
        return od[0] if np.ndim(jaw_width) == 0 else od

    def grasp_center_at(self, jaw_width):
        """Per-grasp center: the pad midpoint at the closure that grips a
        ``jaw_width``-wide object (hand-base frame)."""
        amount = float(np.clip(self._amount_for(jaw_width),
                               self._cal_amount[0], self._cal_amount[-1]))
        return np.array([np.interp(amount, self._cal_amount, self._cal_mid[:, k])
                         for k in range(3)], dtype=np.float32)

    def _grasp_center_loc_tf(self, jaw_width):
        """Per-width grasp center with the bound primitive tcp orientation."""
        if getattr(self, '_grasp', None) is None:
            raise AttributeError(
                f"{type(self).__name__} is a dexterous hand, not bound as a "
                f"parallel jaw. Call hand.spawn_jaw('pinch') (or 'tripod') "
                f"before using grip_at / eval_grasp_tcp.")
        tcp_name = f'{self._grasp}_center'
        rotmat = self.tcp(tcp_name).loc_tf[:3, :3]
        return oum.tf_from_pos_rotmat(pos=self.grasp_center_at(jaw_width),
                                      rotmat=rotmat)

    def eval_grasp_tcp(self, jaw_width):
        """A fresh TCP at the grasp center for ``jaw_width`` (no state mutation).
        Pass straight to ik: ``arm.ik(p, R, tcp=hand.eval_grasp_tcp(jw))``."""
        return orbt.TCP(self.runtime_root_lnk,
                        self._grasp_center_loc_tf(jaw_width))

    def set_jaw_width(self, w):
        w = float(np.clip(w, self.jaw_range[0], self.jaw_range[1]))
        # invert pad-gap(amount): _cal_gap decreases as amount rises, so reverse
        # both for np.interp (needs ascending x).
        amount = float(np.interp(w, self._cal_gap[::-1], self._cal_amount[::-1]))
        self._pose_grasp(self._spec, amount)
        self._jaw_w = w

    def grip_at(self, tgt_pos, tgt_rotmat, jaw_width):
        # Per-grasp center offset (pad midpoint at this closure) with the bound
        # primitive tcp orientation, so antipodal's approach frame maps onto the
        # hand's actual grasp frame instead of the hand base.
        loc = self._grasp_center_loc_tf(jaw_width)
        base = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat) @ np.linalg.inv(loc)
        self.set_pos_rotmat(base[:3, 3], base[:3, :3])
        self.set_jaw_width(jaw_width)

    def clone(self):
        new = super().clone()
        # carry the immutable grasp binding + calibration onto the clone (antipodal
        # clones the gripper internally, so this must survive).
        if getattr(self, '_grasp', None) is not None:
            new._grasp = self._grasp
            new._spec = self._spec
            new.jaw_range = self.jaw_range.copy()
            new.open_dir = self.open_dir.copy()
            new.contact_pattern = self.contact_pattern.copy()
            new._cal_amount = self._cal_amount.copy()
            new._cal_gap = self._cal_gap.copy()
            new._cal_dir = self._cal_dir.copy()
            new._cal_mid = self._cal_mid.copy()
            new._od_amount = self._od_amount.copy()
            new._od_dir = self._od_dir.copy()
            new.set_jaw_width(self._jaw_w)
        return new
