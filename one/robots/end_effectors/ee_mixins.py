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
# ('grasp_center' for grippers, 'tip' for point tools, '<primitive>_center' for
# dexterous hands), so there is no EndEffectorBase: positioning an EE goes
# through cross-object ik, e.g.
#   arm.ik(pos, rotmat, tcp=gripper.tcp('grasp_center'))
#
# Two orthogonal concerns are kept SEPARATE on purpose (their coupling is what
# leaked a re-derived jaw_width back into replay before):
#   * holding   -- attach(child[, qpos]) / detach(child): rigidly mount/unmount
#                  an object. Uniform across grippers and hands (EndEffectorMixin).
#   * closure   -- set_opening(w) (gripper) / pinch|tripod|power(amount) (hand):
#                  type-specific. Compose them: set the closure, then attach.
# The parallel-jaw abstraction the grasp planners consume is a third concern: a
# real gripper IS a jaw; a dexterous hand PRODUCES one via ``as_jaw`` (-> JawView)
# rather than carrying jaw methods on the hand class.


class EndEffectorMixin:
    """Uniform object-holding for any actuated end effector (gripper, hand).

    ``attach`` / ``detach`` rigidly mount / unmount a child object on the EE
    root link. Closure is a SEPARATE concern (``set_opening`` on a gripper,
    ``pinch`` etc. on a hand) -- compose them: set the closure, then attach.
    ``attach(child, qpos=...)`` is the canonical replay of a saved grasp: it
    poses the EE to the grasp's frozen ``qpos`` first, so no scalar jaw_width /
    primitive / amount is re-derived."""

    def attach(self, child, qpos=None):
        """Rigidly mount ``child`` on the EE at its current pose. If ``qpos`` is
        given, pose the EE to it first (e.g. a Grasp's frozen closure)."""
        if qpos is not None:
            self.fk(qs=qpos)
        parent = self.runtime_root_lnk
        loc_tf = np.linalg.inv(parent.tf) @ child.tf
        self.mount(child, parent, loc_tf)

    def detach(self, child):
        """Unmount ``child`` from the EE. Reopening (if wanted) is a separate,
        explicit closure call (``open`` / ``open_hand``)."""
        self.unmount(child)


class GripperMixin(EndEffectorMixin):
    """A parallel-jaw gripper: closes via a single scalar ``set_opening``, and
    natively IS a parallel jaw for the grasp planners (it implements the jaw
    protocol: ``jaw_range`` / ``open_dir`` / ``contact_pattern`` / ``set_opening``
    / ``grasp_center_tcp`` / ``grip_at`` / ``qpos`` / ``mode``). The concrete
    gripper supplies ``jaw_range`` / ``open_dir`` / ``contact_pattern`` and
    implements ``set_opening``."""

    def open(self):
        self.set_opening(self.jaw_range[1])

    def close(self):
        self.set_opening(self.jaw_range[0])

    def set_opening(self, width):
        """Set the jaw opening to ``width`` (concrete gripper implements)."""
        raise NotImplementedError

    # ---- parallel-jaw protocol (consumed by antipodal / polypodal) ---------
    def grip_at(self, tgt_pos, tgt_rotmat, width):
        """Move the grasp center to ``(tgt_pos, tgt_rotmat)`` and set the
        opening to ``width``; return the resulting base tf."""
        if width < self.jaw_range[0] or width > self.jaw_range[1]:
            raise ValueError(f"opening {width} out of range {self.jaw_range}")
        loc = self.tcp('grasp_center').loc_tf
        base_tf = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat) @ np.linalg.inv(loc)
        self.set_pos_rotmat(base_tf[:3, 3], base_tf[:3, :3])
        self.set_opening(width)
        return base_tf

    def grasp_center_tcp(self, width=None):
        """The grasp-center TCP. A parallel gripper's grasp_center is FIXED
        (independent of the opening), so ``width`` is ignored and the registered
        'grasp_center' tcp is returned. (A dexterous hand's JawView shifts it
        with the closure.)"""
        return self.tcp('grasp_center')

    @property
    def qpos(self):
        """The gripper's full joint config -- snapshotted into a Grasp's
        ``qpos`` at plan time, replayed via ``attach(child, qpos)``."""
        return self.qs

    @property
    def mode(self):
        """The grasp primitive that disambiguates the tcp. A parallel jaw has a
        single fixed grasp_center, so there is no mode -- ``None``. (A dexterous
        hand's JawView returns its bound primitive.) Recorded as a Grasp's
        non-authoritative ``provenance['mode']``; replay never depends on it."""
        return None

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
        """Attach object to the tool tip (requires activation first). A point
        tool holds at the TIP frame with an activation gate, so it keeps its own
        attach/detach rather than the root-relative EndEffectorMixin pair."""
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


class DexHandMixin(EndEffectorMixin):
    """Behavior for a multi-finger dexterous hand (a MechBase EE).

    A hand closes via *named grasp primitives* (pinch / tripod / power): each
    maps one scalar ``amount`` in [0, 1] (open -> closed) onto a coordinated
    finger pose. The concrete hand supplies the data; this mixin supplies the
    verbs. Holding (``attach`` / ``detach``) is inherited and uniform with the
    parallel gripper; closure (``pinch`` etc.) is hand-specific -- compose them.

    The concrete hand must define:
    - ``grasp_spec(primitive) -> DexGraspSpec``: the preshape / closing / tuck
      joint targets (and pad links) for a primitive, in this hand's joint/link
      names. One definition feeds BOTH the shape primitives and the parallel-jaw
      planning view, so there is no second copy of the grasp model.
    - center tcps named ``'<primitive>_center'`` for the primitives it wants to
      position via ik / the ``*_at`` helpers (e.g. ``'pinch_center'``,
      ``'power_center'``).

    A pinch/tripod (a thumb-vs-finger(s) opposition) is presented to the
    parallel-jaw grasp planner ``antipodal`` via ``as_jaw(primitive)``, which
    returns a :class:`JawView` -- a calibrated adapter over a bound clone. The
    hand itself carries NO jaw methods; the jaw lives in the JawView.
    """

    def grasp_spec(self, primitive):
        """Return the DexGraspSpec for ``primitive`` (concrete joint/link names).
        Implemented by the concrete hand from its naming convention."""
        raise NotImplementedError

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

    def _world_vs(self, lnk_name):
        """World-frame vertices of a link's first visual mesh."""
        c = self._compiled
        lnk = self.runtime_lnks[c.lidx_map[self.structure.lnk_map[lnk_name]]]
        v = lnk.visuals[0]
        vs = np.asarray(v.geom.vs, dtype=np.float64).reshape(-1, 3)
        m = (lnk.tf @ v._tf).astype(np.float64)
        return vs @ m[:3, :3].T + m[:3, 3]

    # ---- shape primitives (no positioning) -----------------------------
    def open_hand(self):
        """Fully extend every finger (all joints to 0)."""
        self.fk(qs=np.zeros(self.ndof, dtype=np.float32))

    def open(self):
        """Uniform 'fully open' across end effectors (mirrors a parallel
        gripper's ``open``); for a hand this extends every finger. Lets
        EE-agnostic callers (e.g. ``gen_pick_place``) get the open config the
        same way for a gripper or a hand."""
        self.open_hand()

    def grip(self, primitive, amount=1.0):
        """Close a named grasp ``primitive`` to ``amount`` in [0, 1] -- the
        generic form behind ``pinch`` / ``tripod`` / ``power``. Holding is
        separate: compose with ``attach(child)`` to pick an object up."""
        self._pose_grasp(self.grasp_spec(primitive), amount)

    def pinch(self, amount=1.0):
        """Precision pinch: thumb opposes index."""
        self.grip('pinch', amount)

    def tripod(self, amount=1.0):
        """Tripod grip: thumb opposes index + middle (more stable than pinch)."""
        self.grip('tripod', amount)

    def power(self, amount=1.0):
        """Whole-hand enveloping (power) grasp."""
        self.grip('power', amount)

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

    def power_at(self, tgt_pos, tgt_rotmat, amount=1.0):
        return self.grasp_at(tgt_pos, tgt_rotmat, 'power', amount)

    # ---- parallel-jaw view for antipodal planning ----------------------
    def as_jaw(self, primitive):
        """Present this hand to ``antipodal`` as a parallel jaw bound to
        ``primitive``::

            grasps = antipodal(hand.as_jaw('pinch'), obj)

        Returns a :class:`JawView` wrapping a calibrated clone of this hand --
        the hand itself is untouched and carries no jaw state. Raises if
        ``primitive`` is not a parallel-jaw opposition (e.g. 'power')."""
        spec = self.grasp_spec(primitive)
        if spec.opp_pads is None:
            raise ValueError(
                f"grasp '{primitive}' is not a parallel-jaw opposition (no "
                f"opposing pads); it cannot be presented to antipodal")
        return JawView(self.clone(), primitive, spec)


class JawView:
    """A parallel-jaw view of a dexterous hand, bound to one opposition
    primitive (pinch / tripod). It adapts a hand to the jaw protocol that
    ``antipodal`` / ``polypodal`` consume -- WITHOUT polluting the hand class --
    by wrapping a calibrated clone of the hand:

        jaw_range / open_dir / contact_pattern / mode   (attributes)
        set_opening(w) / open_dir_at(w) / grasp_center_tcp(w) / grip_at(...) /
        qpos / runtime_lnks / clone()                   (methods)

    The pinch is a curling motion, so the pad gap, opposition axis and grasp
    center are all nonlinear in ``amount``; calibration sweeps closure once and
    the tables let ``set_opening`` / ``open_dir_at`` / ``grasp_center_tcp``
    invert and interpolate them per object. A real parallel gripper implements
    the same protocol natively (see :class:`GripperMixin`), so the planners take
    either uniformly. The wrapped hand is exposed as ``hand``; ``runtime_lnks``
    forwards to it so the EE-vs-target collision detector works unchanged.
    """

    # closure samples for the calibration sweep, and the representative closure
    # at which the reference open_dir / contact pattern are taken.
    _CAL_AMOUNTS = np.linspace(0.3, 1.0, 20)
    _REF_AMOUNT = 0.7

    def __init__(self, hand, primitive, spec, _calibrate=True):
        self.hand = hand
        self.mode = primitive
        self._spec = spec
        if _calibrate:
            self._calibrate()

    # ---- the wrapped mech, forwarded for collision / posing ----------------
    @property
    def runtime_lnks(self):
        return self.hand.runtime_lnks

    @property
    def qpos(self):
        """The hand's full joint config at the current opening -- snapshotted
        into a Grasp's ``qpos`` at plan time."""
        return self.hand.qs

    # ---- rendering convenience (a JawView is shown as its wrapped hand) -----
    def attach_to(self, scene):
        return self.hand.attach_to(scene)

    @property
    def rgb(self):
        return self.hand.rgb

    @rgb.setter
    def rgb(self, value):
        self.hand.rgb = value

    @property
    def alpha(self):
        return self.hand.alpha

    @alpha.setter
    def alpha(self, value):
        self.hand.alpha = value

    def clone(self):
        new = JawView(self.hand.clone(), self.mode, self._spec, _calibrate=False)
        new.jaw_range = self.jaw_range.copy()
        new.open_dir = self.open_dir.copy()
        new.contact_pattern = self.contact_pattern.copy()
        new._cal_amount = self._cal_amount.copy()
        new._cal_gap = self._cal_gap.copy()
        new._cal_dir = self._cal_dir.copy()
        new._cal_mid = self._cal_mid.copy()
        new._od_amount = self._od_amount.copy()
        new._od_dir = self._od_dir.copy()
        new.set_opening(self._jaw_w)
        return new

    # ---- calibration -------------------------------------------------------
    def _pad_contacts(self, amount):
        """The two opposing pad contact points (world frame) at ``amount``: the
        thumb pad, and the opposing 'jaw'. With one opposing finger this is the
        closest thumb<->finger vertex pair; with several (tripod) it is the mean
        of each finger's closest pair, so the opposing pad sits between them."""
        spec = self._spec
        self.hand._pose_grasp(spec, amount)
        thumb = self.hand._world_vs(spec.thumb_pad)
        t_pts, o_pts = [], []
        for pad in spec.opp_pads:
            b = self.hand._world_vs(pad)
            d = np.linalg.norm(thumb[:, None, :] - b[None, :, :], axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            t_pts.append(thumb[i])
            o_pts.append(b[j])
        return np.mean(t_pts, axis=0), np.mean(o_pts, axis=0)

    def _calibrate(self):
        """Sweep closure and measure (gap, opposition axis, center) at the pad
        contact, in the hand-base frame."""
        b_inv = np.linalg.inv(self.hand.runtime_root_lnk.tf)
        gaps, dirs, mids = [], [], []
        for a in self._CAL_AMOUNTS:
            pad_t, pad_o = self._pad_contacts(a)
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
        self.open_dir = self._cal_dir[ref_idx]
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self._jaw_w = float(self.jaw_range[1])
        self.set_opening(self._jaw_w)

    def _amount_for(self, width):
        # invert the pad-gap(amount) calibration (gap decreases as amount rises)
        w = np.clip(width, self.jaw_range[0], self.jaw_range[1])
        return np.interp(w, self._cal_gap[::-1], self._cal_amount[::-1])

    # ---- jaw protocol ------------------------------------------------------
    def set_opening(self, width):
        width = float(np.clip(width, self.jaw_range[0], self.jaw_range[1]))
        # invert pad-gap(amount): _cal_gap decreases as amount rises, so reverse
        # both for np.interp (needs ascending x).
        amount = float(np.interp(width, self._cal_gap[::-1],
                                 self._cal_amount[::-1]))
        self.hand._pose_grasp(self._spec, amount)
        self._jaw_w = width

    def open_dir_at(self, width):
        """Per-grasp opposition axis: the thumb-pad -> opposing-pad direction at
        the closure that grips a ``width``-wide object (hand-base frame).
        Scalar in -> (3,); array in -> (N, 3)."""
        amount = np.clip(self._amount_for(width),
                         self._od_amount[0], self._od_amount[-1])
        amount = np.atleast_1d(amount)
        od = np.stack([np.interp(amount, self._od_amount, self._od_dir[:, k])
                       for k in range(3)], axis=1)
        od = od / (np.linalg.norm(od, axis=1, keepdims=True) + oum.eps)
        od = od.astype(np.float32)
        return od[0] if np.ndim(width) == 0 else od

    def grasp_center_at(self, width):
        """Per-grasp center: the pad midpoint at the closure that grips a
        ``width``-wide object (hand-base frame)."""
        amount = float(np.clip(self._amount_for(width),
                               self._cal_amount[0], self._cal_amount[-1]))
        return np.array([np.interp(amount, self._cal_amount, self._cal_mid[:, k])
                         for k in range(3)], dtype=np.float32)

    def _grasp_center_loc_tf(self, width):
        """Per-width grasp center with the bound primitive's tcp orientation."""
        rotmat = self.hand.tcp(f'{self.mode}_center').loc_tf[:3, :3]
        return oum.tf_from_pos_rotmat(pos=self.grasp_center_at(width),
                                      rotmat=rotmat)

    def grasp_center_tcp(self, width):
        """A fresh grasp-center TCP for ``width`` (no state mutation). Its origin
        shifts with the closure (the pad midpoint) and its rotation is the bound
        primitive's. Pass straight to ik / freeze into a Grasp."""
        return orbt.TCP(self.hand.runtime_root_lnk,
                        self._grasp_center_loc_tf(width))

    def grip_at(self, tgt_pos, tgt_rotmat, width):
        """Pose the wrapped hand so its grasp center sits at ``(tgt_pos,
        tgt_rotmat)`` at opening ``width`` (antipodal's collision placement)."""
        loc = self._grasp_center_loc_tf(width)
        base = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat) @ np.linalg.inv(loc)
        self.hand.set_pos_rotmat(base[:3, 3], base[:3, :3])
        self.set_opening(width)
