import os

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_base as orbmb
import one.robots.base.urdf_loader as orul
import one.robots.end_effectors.ee_mixins as oremx


_URDF_PATH = os.path.join(os.path.dirname(__file__), 'xhand_right.urdf')


def prepare_mechstruct(collision_type=ouc.CollisionType.MESH):
    """Load the XHand right (12-dof dexterous hand) from its URDF into a
    MechStruct. Kinematics, link meshes, colors and joint limits all come from
    ``xhand_right.urdf`` -- no hand-coded link/joint table."""
    urdf_path = os.path.abspath(_URDF_PATH)
    urdf_dir = os.path.dirname(urdf_path)
    urdf = orul.load_robot_from_xacro(urdf_path, base_dir=urdf_dir)
    return orul.urdf_to_mechstruct(
        urdf, urdf_dir,
        collision_type=collision_type,
        res_dir=os.path.dirname(__file__))


class XHandRight(oremx.DexHandMixin, orbmb.MechBase):
    """XHand right: a 12-dof dexterous hand as a mountable MechBase EE with the
    DexHandMixin grasp behaviors. Closure is the ``pinch`` / ``tripod`` /
    ``power`` (alias ``grip``) primitives and ``open_hand``; holding a child is
    the uniform ``attach`` / ``detach``; ``as_jaw`` returns a parallel-jaw
    ``JawView`` for antipodal planning.

    DexHandMixin is listed FIRST so its ``clone`` (which carries the jaw
    calibration) is reached before MechBase's in the MRO and chains up via super.

    Joint order (the qs / conf vector), straight from the URDF:
      thumb  joint0..2 -> conf[0:3]   (swing across palm, then 2 flexion)
      index  joint0..2 -> conf[3:6]   (abduction, then 2 flexion)
      middle joint0..1 -> conf[6:8]   (2 flexion)
      ring   joint0..1 -> conf[8:10]
      pinky  joint0..1 -> conf[10:12]

    Carries two grasp-center tcps (offsets on the palm link, tunable):
    - ``power_center``: center of a whole-hand enveloping grasp (palm cup).
    - ``pinch_center``: thumb-index pinch point (shared by the tripod primitive).

    NOTE: the grasp-table closures and the tcp offsets below are a first pass
    derived from the joint axes / mesh layout. Verify and tune them in the
    viewer (run this module) before relying on the *_at helpers or as_jaw.
    """

    _TUCK = 1.92  # flexion curl for fingers not participating in the grasp

    # Single per-primitive grasp definition (concrete URDF joint / link names).
    # Drives BOTH the shape primitives and the parallel-jaw planning view.
    #   preshape: held FIXED at every closure (the thumb swing that opposes the
    #             fingers; must not scale with amount or it under-swings a wide
    #             grip).
    #   closing:  scaled by amount (the flexion that shuts the grip).
    #   pads:     (thumb_distal, [opposing_distal, ...]) presented to antipodal
    #             as a parallel jaw; None for the non-opposition power envelope.
    # Closures tuned against measured fingertip positions (the WRS set_flange
    # offsets): at full pinch the thumb / index fingertips close to ~20 mm
    # (pads just touching) at a contact near (0.036, -0.088, 0.121). Both finger
    # nodes (joint1 proximal + joint2 distal) curl, not just the proximal.
    _GRASP_TABLE = {
        'pinch': {
            'preshape': {'thumb_joint0': 1.35},
            'closing': {'thumb_joint1': 0.9, 'thumb_joint2': 0.4,
                        'index_joint1': 1.0, 'index_joint2': 1.0},
            'pads': ('thumb_rota_link2', ['index_rota_link2']),
        },
        'tripod': {
            'preshape': {'thumb_joint0': 1.5},
            'closing': {'thumb_joint1': 0.9, 'thumb_joint2': 0.4,
                        'index_joint1': 1.0, 'index_joint2': 1.0,
                        'middle_joint0': 1.0, 'middle_joint1': 1.0},
            'pads': ('thumb_rota_link2', ['index_rota_link2', 'mid_link2']),
        },
        'power': {   # all five fingers envelop -- not a parallel jaw
            'preshape': {},
            'closing': {'thumb_joint0': 1.75, 'thumb_joint1': 0.7,
                        'thumb_joint2': 0.5,
                        'index_joint1': 1.2, 'index_joint2': 1.0,
                        'middle_joint0': 1.2, 'middle_joint1': 1.0,
                        'ring_joint0': 1.2, 'ring_joint1': 1.0,
                        'pinky_joint0': 1.2, 'pinky_joint1': 1.0},
            'pads': None,
        },
    }

    # Flexion joints used to tuck non-participants. XHand joints are independent,
    # so curl both phalanges instead of relying on mimic coupling.
    _FINGER_TUCK = {
        'index': ('index_joint1', 'index_joint2'),
        'middle': ('middle_joint0', 'middle_joint1'),
        'ring': ('ring_joint0', 'ring_joint1'),
        'pinky': ('pinky_joint0', 'pinky_joint1'),
    }

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    # the hand's grasp frame orientation (WRS acting-center rotmat): +approach
    # and opening axes align with how the fingers close toward -y.
    _GRASP_ROT = oum.rotmat_from_euler(oum.pi / 2, oum.pi / 2, 0)

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)   # is_floating=True until mounted
        # power_center: WRS whole-hand acting center (object sits in the palm
        # cup). pinch_center: measured thumb-index pad contact at full pinch.
        self.add_tcp('power_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(
                         pos=np.array([0.0, -0.075, 0.075], dtype=np.float32),
                         rotmat=self._GRASP_ROT))
        self.add_tcp('pinch_center', self.runtime_root_lnk,
                     oum.tf_from_pos_rotmat(
                         pos=np.array([0.03, -0.085, 0.12], dtype=np.float32),
                         rotmat=self._GRASP_ROT))

    def grasp_spec(self, primitive):
        """Resolve _GRASP_TABLE[primitive] to a DexGraspSpec. Tuck = every
        finger this grasp doesn't already drive."""
        t = self._GRASP_TABLE[primitive]
        preshape = dict(t['preshape'])
        closing = dict(t['closing'])
        tuck = {}
        for joints in self._FINGER_TUCK.values():
            if not any(j in closing for j in joints):
                tuck.update({j: self._TUCK for j in joints})
        pads = t['pads']
        thumb_pad = None if pads is None else pads[0]
        opp_pads = None if pads is None else list(pads[1])
        return oremx.DexGraspSpec(
            preshape=preshape, closing=closing, tuck=tuck,
            thumb_pad=thumb_pad, opp_pads=opp_pads)

    # ---- convenience kept from the original model ----------------------
    def goto_given_conf(self, conf):
        self.fk(qs=conf)

    def rand_conf(self):
        lo = self._compiled.jlmt_low_by_idx
        hi = self._compiled.jlmt_high_by_idx
        return np.random.uniform(lo, hi).astype(np.float32)


if __name__ == '__main__':
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    # one hand per grasp primitive, spread along y so they don't overlap.
    # (name, action, which center tcp to draw)
    PREVIEW = [
        ('open',   lambda h: h.open_hand(),      None),
        ('pinch',  lambda h: h.pinch(1.0),       'pinch_center'),
        ('tripod', lambda h: h.tripod(1.0),      'pinch_center'),
        ('power',  lambda h: h.power(1.0),       'power_center'),
    ]
    dy = 0.22   # y spacing between hands

    base = ovw.World(cam_pos=(0.6, -0.4, 0.5),
                     cam_lookat_pos=(0.0, -0.5 * dy * (len(PREVIEW) - 1), 0.08))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    hands = []
    for i, (name, action, tcp) in enumerate(PREVIEW):
        hand = XHandRight(pos=np.array([0.0, -i * dy, 0.0], dtype=np.float32))
        action(hand)
        hand.attach_to(base.scene)
        # small frame at the hand base to mark each one
        ossop.frame(pos=hand.runtime_root_lnk.tf[:3, 3],
                    rotmat=hand.runtime_root_lnk.tf[:3, :3]).attach_to(base.scene)
        if tcp is not None:
            hand.toggle_tcp(tcp, length_scale=0.15, radius_scale=0.25)
        hands.append(hand)
        print(f"hand {i}: {name}  (y = {-i * dy:.2f})")
    builtins.hands = hands
    base.run()
