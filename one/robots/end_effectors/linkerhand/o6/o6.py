import os

import numpy as np

import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_base as orbmb
import one.robots.base.urdf_loader as orul


_URDF_DIR = os.path.join(os.path.dirname(__file__), 'urdf')


def prepare_mechstruct(side, collision_type=ouc.CollisionType.MESH):
    """Load one side of the LinkerHand O6 dexterous hand from its standalone
    URDF (carved out of the L1 h0602 body URDF) into a MechStruct."""
    urdf_path = os.path.abspath(os.path.join(_URDF_DIR, f'o6_{side}.urdf'))
    urdf_dir = os.path.dirname(urdf_path)
    urdf = orul.load_robot_from_xacro(urdf_path, base_dir=urdf_dir)
    return orul.urdf_to_mechstruct(
        urdf, urdf_dir,
        collision_type=collision_type,
        res_dir=os.path.dirname(__file__))


class _O6Hand(orbmb.MechBase):
    """LinkerHand O6 dexterous hand (one side) as a mountable MechBase EE.

    Carries two grasp-center tcps (offsets on the hand base link, tunable):

    - ``power_grasp_center``: center of a whole-hand power/enveloping grasp,
      sitting in the palm cup.
    - ``pinch_center``: thumb-index pinch point, for precision picks.

    The 11 finger joints are present in the structure (FK / visualisation work),
    but finger chains + fingertip ik are not modelled yet -- the hand is used as
    a rigid EE mounted on an arm, positioned via a center tcp through cross-object
    ik (e.g. ``arm.ik('left_arm', hand.tcp('power_grasp_center'), R, p)``).
    """

    # palmar y sign: thumb opposes the fingers across this side of the palm.
    # left hand mirrors right exactly in y, so the tcp y-offsets flip with _Y
    # while x (palm normal, +x = palmar/grasping side) and z (along the hand)
    # stay the same. Offsets derived from mesh geometry: power_grasp_center =
    # centroid of the five fingertips curled into an enveloping grasp;
    # pinch_center = thumb/index fingertip contact point in a precision pinch.
    _Y = None

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)   # is_free=True until mounted
        y = self._Y
        self.add_tcp('power_grasp_center', self.runtime_root_lnk,
                     oum.tf_from_rotmat_pos(
                         pos=np.array([0.038, 0.005 * y, 0.122], dtype=np.float32)))
        self.add_tcp('pinch_center', self.runtime_root_lnk,
                     oum.tf_from_rotmat_pos(
                         pos=np.array([0.055, 0.030 * y, 0.120], dtype=np.float32)))


class O6Left(_O6Hand):
    _Y = -1.0

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct('left')


class O6Right(_O6Hand):
    _Y = 1.0

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct('right')


if __name__ == '__main__':
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(0.3, 0.25, 0.25), cam_lookat_pos=(0.0, 0.0, 0.08))
    hand = O6Left()
    hand.attach_to(base.scene)
    ossop.frame().attach_to(base.scene)
    # keep arrows thin and short (head must stay < shaft length): arrow length
    # = 0.2*length_scale = 0.03 m, head length = 0.04*radius_scale = 0.01 m.
    hand.toggle_tcp('power_grasp_center', length_scale=0.15, radius_scale=0.25)
    hand.toggle_tcp('pinch_center', length_scale=0.15, radius_scale=0.25)
    base.run()
