import os

import numpy as np

import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_base as orbmb
import one.robots.base.urdf_loader as orul
import one.robots.end_effectors.ee_mixins as oremx


_URDF_DIR = os.path.join(os.path.dirname(__file__), 'urdf')


def prepare_mechstruct(side, collision_type=ouc.CollisionType.MESH):
    """Load one side of the Linkerbot O6 dexterous hand from its standalone
    URDF (carved out of the L1 h0602 body URDF) into a MechStruct."""
    urdf_path = os.path.abspath(os.path.join(_URDF_DIR, f'o6_{side}.urdf'))
    urdf_dir = os.path.dirname(urdf_path)
    urdf = orul.load_robot_from_xacro(urdf_path, base_dir=urdf_dir)
    return orul.urdf_to_mechstruct(
        urdf, urdf_dir,
        collision_type=collision_type,
        res_dir=os.path.dirname(__file__))


class _O6Hand(oremx.DexHandMixin, orbmb.MechBase):
    """Linkerbot O6 dexterous hand (one side): a mountable MechBase EE with the
    DexHandMixin grasp behaviors (open_hand / pinch / tripod / power_grasp,
    grasp / release, *_at positioning, and ``spawn_jaw`` for antipodal planning).

    DexHandMixin is listed FIRST so its ``clone`` (which carries the jaw
    calibration) is reached before MechBase's in the MRO and chains up via super.

    Carries two grasp-center tcps (offsets on the hand base link, tunable):

    - ``power_center``: center of a whole-hand power/enveloping grasp,
      sitting in the palm cup.
    - ``pinch_center``: thumb-index pinch point, for precision picks (shared by
      the tripod primitive).

    Finger chains + fingertip ik are not modelled; the hand closes via the named
    grasp primitives (a scalar ``amount`` -> coordinated finger qs) and is
    positioned as a rigid EE through a center tcp via cross-object ik, e.g.
    ``arm.ik(p, R, chain='left_arm', tcp=hand.tcp('power_center'))``.
    """

    # palmar y sign: thumb opposes the fingers across this side of the palm.
    # left hand mirrors right exactly in y, so the tcp y-offsets flip with _Y
    # while x (palm normal, +x = palmar/grasping side) and z (along the hand)
    # stay the same. Offsets derived from mesh geometry: power_center =
    # centroid of the five fingertips curled into an enveloping grasp;
    # pinch_center = thumb/index fingertip contact point in a precision pinch.
    _Y = None
    _PREFIX = None   # 'lh_' / 'rh_' joint-name prefix (set per side)

    _TUCK = 1.3   # mcp curl for fingers not participating in the grasp

    # Single per-primitive grasp definition (joint BASEnames; prefixed per side
    # in grasp_spec). Drives BOTH the shape primitives and the parallel-jaw
    # planning view -- one source, no second copy.
    #   preshape: held fixed at every closure (thumb swing that opposes; must not
    #             scale with amount or it under-swings a wide grip)
    #   closing:  scaled by amount (the flexion that shuts the grip)
    #   pads:     (thumb, [opposing fingers]) -> presentable to antipodal as a
    #             parallel jaw; None for the power envelope.
    # Only INDEPENDENT joints are listed -- thumb_ip and the *_dip joints are
    # URDF <mimic> couplings (thumb_ip = 2.29*cmc_pitch, dip = 0.89*mcp) and
    # follow in fk. cmc_pitch <= 0.47 so the mimic thumb_ip stays within 1.08.
    _GRASP_TABLE = {
        'pinch': {
            'preshape': {'thumb_cmc_yaw': 0.9},
            'closing': {'thumb_cmc_pitch': 0.45, 'index_mcp_pitch': 0.9},
            'pads': ('thumb', ['index']),
        },
        'tripod': {
            # middle flexed deeper (1.15 vs index 0.9): it sits farther from the
            # thumb, so matched this way both pads track the thumb within a few
            # mm across the closure -- a genuine third contact, not decorative.
            'preshape': {'thumb_cmc_yaw': 0.9},
            'closing': {'thumb_cmc_pitch': 0.45, 'index_mcp_pitch': 0.9,
                        'middle_mcp_pitch': 1.15},
            'pads': ('thumb', ['index', 'middle']),
        },
        'power': {   # all five fingers envelop -- not a parallel jaw
            'preshape': {},
            'closing': {'thumb_cmc_yaw': 0.8, 'thumb_cmc_pitch': 0.4,
                        'index_mcp_pitch': 1.0, 'middle_mcp_pitch': 1.0,
                        'ring_mcp_pitch': 1.0, 'pinky_mcp_pitch': 1.0},
            'pads': None,
        },
    }

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)   # is_free=True until mounted
        y = self._Y
        self.add_tcp('power_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(
                         pos=np.array([0.038, 0.005 * y, 0.122], dtype=np.float32)))
        self.add_tcp('pinch_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(
                         pos=np.array([0.055, 0.030 * y, 0.120], dtype=np.float32)))

    def grasp_spec(self, primitive):
        """Resolve _GRASP_TABLE[primitive] to a DexGraspSpec with this side's
        prefixed joint / link names. Tuck = every finger this grasp doesn't
        drive."""
        p = self._PREFIX
        t = self._GRASP_TABLE[primitive]
        preshape = {p + j: v for j, v in t['preshape'].items()}
        closing = {p + j: v for j, v in t['closing'].items()}
        tuck = {f'{p}{f}_mcp_pitch': self._TUCK
                for f in ('index', 'middle', 'ring', 'pinky')
                if f'{p}{f}_mcp_pitch' not in closing}
        pads = t['pads']
        thumb_pad = None if pads is None else f'{p}{pads[0]}_distal'
        opp_pads = None if pads is None else [f'{p}{f}_distal' for f in pads[1]]
        return oremx.DexGraspSpec(
            preshape=preshape, closing=closing, tuck=tuck,
            thumb_pad=thumb_pad, opp_pads=opp_pads)


class O6Left(_O6Hand):
    _Y = -1.0
    _PREFIX = 'lh_'

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct('left')


class O6Right(_O6Hand):
    _Y = 1.0
    _PREFIX = 'rh_'

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct('right')


if __name__ == '__main__':
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(0.45, 0.0, 0.25), cam_lookat_pos=(0.0, 0.0, 0.08))
    ossop.frame().attach_to(base.scene)

    # place the two hands apart in y so they don't overlap
    left = O6Left(pos=np.array([0.0, 0.12, 0.0], dtype=np.float32))
    right = O6Right(pos=np.array([0.0, -0.12, 0.0], dtype=np.float32))
    left.pinch(1.0)          # thumb-index precision pinch
    right.power_grasp(1.0)   # whole-hand envelope
    for hand in (left, right):
        hand.attach_to(base.scene)
        # keep arrows thin and short (head must stay < shaft length): arrow
        # length = 0.2*length_scale = 0.03 m, head = 0.04*radius_scale = 0.01 m.
        hand.toggle_tcp('power_center', length_scale=0.15, radius_scale=0.25)
        hand.toggle_tcp('pinch_center', length_scale=0.15, radius_scale=0.25)
    base.run()
