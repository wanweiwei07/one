"""Antipodal grasp planning for the Linkerbot O6 dexterous hand's PINCH.

``antipodal`` is written for a parallel-jaw gripper, but a thumb-index pinch is
exactly a 2-point opposing grasp, so we present the pinch to it through a thin
adapter (``O6Pinch``) that exposes the gripper interface antipodal needs:
``grasp_center`` tcp, ``open_dir``, ``jaw_range``, ``contact_pattern``,
``set_jaw_width`` (-> pinch synergy, with the other 3 fingers tucked away) and
``grip_at``.

The adapter is parameterized by ``_OPPOSING`` (which finger(s) face the thumb),
so ``test_o6_tripod`` reuses it with index + middle. Shows the best pinch on a
small bunny solid, plus the next few as translucent ghosts. Keys: N = next grasp
as the solid one, R = re-plan.
"""
import os

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.geom.loader as ogl
import one.scene.render_model as osrm
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.robots.end_effectors.linkerbot.o6.o6 as o6
from one.grasp.antipodal import antipodal

BUNNY_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'bunny_small.stl')
TUCK = 1.3   # mcp curl for the non-pinching middle/ring/pinky (dip mimics mcp)


class O6Pinch(o6.O6Left):
    """O6 left hand adapted to the parallel-jaw interface ``antipodal`` expects.
    The thumb opposes one or more fingers (``_OPPOSING``); for the plain pinch
    that is the index alone, so the opposing 'jaw' is the index pad and the other
    three fingers are tucked away. ``jaw_width`` is the thumb-vs-opposing opening.
    Subclass and widen ``_OPPOSING`` for other oppositions (e.g. tripod = index +
    middle)."""

    # opposing finger(s) -> mcp flexion at full close; the rest are tucked away.
    # Per-finger pitch (not a shared scalar) so each finger's pad can be brought
    # to the SAME distance from the thumb: the index and middle sit at different
    # palm positions, so a single pitch would leave the middle ~10 mm short and
    # never contacting (decorative). For the tripod the middle is flexed deeper
    # (1.15 vs index 0.9) so thumb-index and thumb-middle gaps track within a few
    # mm across the closure -- both fingers genuinely oppose the thumb.
    _OPPOSING = {'index': 0.9}
    REF_AMOUNT = 0.7   # representative half-pinch where an object sits between
                       # the pads (full pinch is degenerate: the pads touch)
    # Closing model: a fixed thumb-swing preshape (THUMB_YAW) brings the thumb
    # across to oppose the finger(s) at every opening (a single linear synergy
    # under-swings it when wide); the flexion joints then shut the grip, scaled
    # with amount. Only the INDEPENDENT joints are driven -- thumb_ip and the
    # *_dip joints follow via the URDF <mimic> couplings (thumb_ip =
    # 2.29*cmc_pitch, dip = 0.89*mcp).
    _THUMB_YAW = 0.9     # preshape: thumb swing across (fixed, all openings)
    _THUMB_PITCH = 0.45  # thumb flexion at full close (capped so the mimic
                         # thumb_ip = 2.29*cmc_pitch stays within its 1.08 limit)

    def __init__(self):
        super().__init__()
        b_inv = np.linalg.inv(self.runtime_root_lnk.tf)   # world -> hand base
        # Calibrate the pinch once: the thumb-index pinch is a curling motion, so
        # both the pad separation AND the pad-opposition axis are nonlinear
        # functions of the synergy amount. Sample (amount -> gap, dir) so that
        #   set_jaw_width(w) closes to the amount where the pads are w apart, and
        #   open_dir_at(w)  returns the real opposition axis at that closure.
        # All measured at the pad contact (closest points between thumb & index
        # distal meshes), in the hand-base / grasp_center frame.
        amts = np.linspace(0.3, 1.0, 20)
        gaps, dirs, mids = [], [], []
        for a in amts:
            pad_t, pad_i = self._pad_contacts(a)
            ct = b_inv[:3, :3] @ pad_t + b_inv[:3, 3]
            ci = b_inv[:3, :3] @ pad_i + b_inv[:3, 3]
            d = ci - ct
            gaps.append(np.linalg.norm(d))
            dirs.append(d / (np.linalg.norm(d) + oum.eps))
            mids.append((ct + ci) * 0.5)
        gaps = np.array(gaps, dtype=np.float32)
        # keep only the monotonic-decreasing prefix: near full closure the
        # curling fingers slide past each other and the closest-point gap gets
        # noisy / non-monotonic, which would break the gap <-> amount inversion.
        n = 1
        while n < len(gaps) and gaps[n] < gaps[n - 1] - 1e-4:
            n += 1
        self._cal_amount = amts[:n]
        self._cal_gap = gaps[:n]
        self._cal_dir = np.array(dirs[:n], dtype=np.float32)
        self._cal_mid = np.array(mids[:n], dtype=np.float32)
        # achievable pad separations (gap shrinks as the pinch closes)
        self.jaw_range = np.array([self._cal_gap[-1], self._cal_gap[0]],
                                  dtype=np.float32)
        # open_dir table: drop amounts where the pads ~touch (axis ill-defined)
        ok = self._cal_gap >= 0.005
        self._od_amount = self._cal_amount[ok]
        self._od_dir = self._cal_dir[ok]
        ref_idx = int(np.argmin(np.abs(self._cal_amount - self.REF_AMOUNT)))
        # 'grasp_center' tcp at a representative closure -- antipodal reads it for
        # the pre-grasp retreat distance; the actual per-grasp center comes from
        # grasp_center_at() inside grip_at.
        self.add_tcp('grasp_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(pos=self._cal_mid[ref_idx]))
        # reference open_dir (fallback / contact_depth); antipodal calls
        # open_dir_at() per candidate. axis = thumb pad -> index pad, hand frame.
        self.open_dir = self._cal_dir[ref_idx]
        self.contact_pattern = np.zeros((1, 3), dtype=np.float32)
        self._jaw_w = self.jaw_range[1]
        self.set_jaw_width(self._jaw_w)

    def _world_vs(self, lnk_name):
        c = self._compiled
        lnk = self.runtime_lnks[c.lidx_map[self.structure.lnk_map[lnk_name]]]
        v = lnk.visuals[0]
        vs = np.asarray(v.geom.vs, dtype=np.float64).reshape(-1, 3)
        m = (lnk.tf @ v._tf).astype(np.float64)
        return vs @ m[:3, :3].T + m[:3, 3]

    def _pose_pinch(self, amount):
        """Pose the grip at ``amount`` in [0,1]: thumb opposition (preshape)
        fixed, the flexion joints close (thumb + the opposing fingers), the rest
        tucked away. Only independent joints are set; mimic (dip / thumb_ip)
        follow in fk."""
        amount = float(np.clip(amount, 0.0, 1.0))
        qs = np.zeros(self._compiled.n_jnts, dtype=np.float32)   # full per-joint
        nm = {j.name: i for i, j in enumerate(self.structure.jnts)}
        qs[nm['lh_thumb_cmc_yaw']] = self._THUMB_YAW             # fixed opposition
        qs[nm['lh_thumb_cmc_pitch']] = self._THUMB_PITCH * amount
        for f in ('index', 'middle', 'ring', 'pinky'):
            if f in self._OPPOSING:
                qs[nm[f'lh_{f}_mcp_pitch']] = self._OPPOSING[f] * amount
            else:
                qs[nm[f'lh_{f}_mcp_pitch']] = TUCK              # dip follows mimic
        self.fk(qs=qs)

    def _pad_contacts(self, amount):
        """The two opposing pad contact points (world frame) at the given grip
        amount: the thumb pad and the opposing 'jaw'. With a single opposing
        finger this is the closest thumb<->finger vertex pair; with several (e.g.
        tripod) it is the mean of each finger's closest pair, so the opposing pad
        sits between them."""
        self._pose_pinch(amount)
        thumb = self._world_vs('lh_thumb_distal')
        t_pts, o_pts = [], []
        for f in self._OPPOSING:
            b = self._world_vs(f'lh_{f}_distal')
            d = np.linalg.norm(thumb[:, None, :] - b[None, :, :], axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            t_pts.append(thumb[i])
            o_pts.append(b[j])
        return np.mean(t_pts, axis=0), np.mean(o_pts, axis=0)

    def _pad_gap(self, amount):
        a, b = self._pad_contacts(amount)
        return float(np.linalg.norm(a - b))

    def _amount_for(self, jaw_width):
        # invert the pad-gap(amount) calibration (gap decreases as amount rises)
        w = np.clip(jaw_width, self.jaw_range[0], self.jaw_range[1])
        return np.interp(w, self._cal_gap[::-1], self._cal_amount[::-1])

    def open_dir_at(self, jaw_width):
        """Per-grasp opposition axis: the thumb-pad -> index-pad direction at the
        closure that grips a ``jaw_width``-wide object (hand-base frame). Scalar
        in -> (3,); array in -> (N, 3). antipodal calls this per candidate so the
        real pinch axis aligns with each object's antipodal contact pair."""
        amount = np.clip(self._amount_for(jaw_width),
                         self._od_amount[0], self._od_amount[-1])
        amount = np.atleast_1d(amount)
        od = np.stack([np.interp(amount, self._od_amount, self._od_dir[:, k])
                       for k in range(3)], axis=1)
        od = (od / (np.linalg.norm(od, axis=1, keepdims=True) + oum.eps))
        od = od.astype(np.float32)
        return od[0] if np.ndim(jaw_width) == 0 else od

    def grasp_center_at(self, jaw_width):
        """Per-grasp center: the pad midpoint at the closure that grips a
        ``jaw_width``-wide object (hand-base frame), so the object sits between
        the pads at the actual closure rather than at a fixed reference."""
        amount = float(np.clip(self._amount_for(jaw_width),
                               self._cal_amount[0], self._cal_amount[-1]))
        return np.array([np.interp(amount, self._cal_amount, self._cal_mid[:, k])
                         for k in range(3)], dtype=np.float32)

    def set_jaw_width(self, w):
        w = float(np.clip(w, self.jaw_range[0], self.jaw_range[1]))
        # invert pad-gap(amount): _cal_gap decreases as amount increases, so
        # reverse both for np.interp (needs ascending x).
        amount = float(np.interp(w, self._cal_gap[::-1], self._cal_amount[::-1]))
        self._pose_pinch(amount)
        self._jaw_w = w

    def grip_at(self, tgt_pos, tgt_rotmat, jaw_width):
        # per-grasp center offset (pad midpoint at this closure), not the fixed
        # tcp -> the object ends up between the pads at the actual closure.
        loc = oum.tf_from_pos_rotmat(pos=self.grasp_center_at(jaw_width))
        base = oum.tf_from_pos_rotmat(tgt_pos, tgt_rotmat) @ np.linalg.inv(loc)
        self.set_pos_rotmat(base[:3, 3], base[:3, :3])
        self.set_jaw_width(jaw_width)

    def clone(self):
        new = super().clone()
        new.open_dir = self.open_dir.copy()
        new.jaw_range = self.jaw_range.copy()
        new.contact_pattern = self.contact_pattern.copy()
        new._cal_amount = self._cal_amount.copy()
        new._cal_gap = self._cal_gap.copy()
        new._cal_dir = self._cal_dir.copy()
        new._cal_mid = self._cal_mid.copy()
        new._od_amount = self._od_amount.copy()
        new._od_dir = self._od_dir.copy()
        new.set_jaw_width(self._jaw_w)
        return new


def make_bunny():
    bunny = osso.SceneObject(collision_type=ouc.CollisionType.MESH, is_free=True)
    bunny.add_visual(
        osrm.RenderModel(geom=ogl.load_geometry(BUNNY_STL), rgb=(0.85, 0.7, 0.6)),
        auto_make_collision=True)
    return bunny


def main(grasper_cls=O6Pinch, label='pinch'):
    base = ovw.World(cam_pos=(0.28, 0.22, 0.18), cam_lookat_pos=(0.0, 0.0, 0.02))
    # length_scale shrinks the shaft; radius_scale must shrink the head too,
    # else the (unscaled) head fills the whole arrow and only tips show.
    ossop.frame(length_scale=0.2, radius_scale=0.25).attach_to(base.scene)

    grasper = grasper_cls()
    bunny = make_bunny()
    bunny.attach_to(base.scene)

    state = {'grasps': [], 'cur': 0, 'solid': None, 'ghosts': []}

    def clear():
        if state['solid'] is not None:
            state['solid'].detach_from(base.scene)
            state['solid'] = None
        for gh in state['ghosts']:
            gh.detach_from(base.scene)
        state['ghosts'] = []

    def show(idx):
        clear()
        grasps = state['grasps']
        if not grasps:
            return
        idx %= len(grasps)
        state['cur'] = idx
        # a few lower-ranked candidates as translucent ghosts
        for pose, _pre, jw, _sc in grasps[1:6]:
            gh = grasper.clone()
            gh.grip_at(pose[:3, 3], pose[:3, :3], jw)
            gh.alpha = 0.2
            gh.attach_to(base.scene)
            state['ghosts'].append(gh)
        # the selected grasp solid
        pose, _pre, jw, _sc = grasps[idx]
        solid = grasper.clone()
        solid.grip_at(pose[:3, 3], pose[:3, :3], jw)
        solid.attach_to(base.scene)
        state['solid'] = solid

    def replan():
        grasps = antipodal(grasper, bunny, density=0.0008, normal_tol_deg=25,
                           roll_step_deg=30, max_grasps=40, clearance=0.003)
        state['grasps'] = grasps
        print(f'antipodal {label} grasps: {len(grasps)} '
              f'(N = next, R = re-plan)')
        show(0)

    replan()

    import pyglet.window.key as key

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.N):
            show(state['cur'] + 1)
        if base.input_manager.is_key_pressed_edge(key.R):
            replan()

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == '__main__':
    main()
