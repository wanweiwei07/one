import os

import numpy as np

import one.utils.constant as ouc
import one.robots.base.mech_base as orbmb
import one.robots.base.urdf_loader as orul
import one.robots.end_effectors.linkerhand.o6.o6 as oello6


_H0602_URDF = os.path.join(
    os.path.dirname(__file__),
    'urdf',
    'h0602.urdf',
)


def prepare_mechstruct(collision_type=ouc.CollisionType.MESH):
    """Load the Linx L1/H0602 upper-body URDF into a MechStruct, via the shared
    urdf_loader (no bespoke XML parsing)."""
    urdf_path = os.path.abspath(_H0602_URDF)
    urdf_dir = os.path.dirname(urdf_path)
    urdf = orul.load_robot_from_xacro(urdf_path, base_dir=urdf_dir)
    return orul.urdf_to_mechstruct(
        urdf, urdf_dir,
        collision_type=collision_type,
        res_dir=os.path.dirname(__file__))


class L1(orbmb.MechBase):
    """Linx L1 upper-body humanoid model based on the H0602 URDF."""

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None, home_qs=None, is_free=True):
        super().__init__(rotmat=rotmat, pos=pos,
                         home_qs=home_qs, is_free=is_free)
        lm = self.structure.lnk_map
        # named chains (which joints move) -- base/tip are structure links
        self.add_chain('left_arm', lm['waist_link2'], lm['left_arm_link_6'])
        self.add_chain('right_arm', lm['waist_link2'], lm['right_arm_link_6'])
        self.add_chain('left_arm_waist', lm['waist_link1'], lm['left_arm_link_6'])
        self.add_chain('right_arm_waist', lm['waist_link1'], lm['right_arm_link_6'])
        self.add_chain('neck', lm['waist_link2'], lm['neck_link2'])
        # dexterous hands: standalone O6 EEs mounted on each arm flange (the
        # *_linkerhand_mount_joint, xyz=(0,0,0.034), that used to live in the
        # body URDF). grasp targets come from the hands' center tcps via
        # cross-object ik, e.g.
        #   robot.ik('left_arm', robot.left_hand.tcp('power_grasp_center'), R, p)
        mount_tf = oum.tf_from_rotmat_pos(
            pos=np.array([0.0, 0.0, 0.034], dtype=np.float32))
        self.left_hand = oello6.O6Left()
        self.right_hand = oello6.O6Right()
        self.mount(self.left_hand, self.lnk('left_arm_link_6'), mount_tf, update=True)
        self.mount(self.right_hand, self.lnk('right_arm_link_6'), mount_tf, update=True)

    def clone(self):
        new = super().clone()
        # MechBase.clone deep-copies the mounted hands but not these instance
        # handles; re-bind them to the cloned children (by EE class -> side).
        for child in new._mountings:
            if isinstance(child, oello6.O6Left):
                new.left_hand = child
            elif isinstance(child, oello6.O6Right):
                new.right_hand = child
        return new

    def lnk(self, name):
        lidx = self.structure.compiled.lidx_map[self.structure.lnk_map[name]]
        return self.runtime_lnks[lidx]

    def jnt(self, name):
        return self.structure.jnt_map[name]


if __name__ == '__main__':
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(2.2, 1.4, 1.6),
                     cam_lookat_pos=(0.0, 0.0, 0.9))
    robot = L1()
    robot.attach_to(base.scene)
    # arm / hand / manipulator all share the same call; grasp targets are the
    # mounted hands' center tcps (cross-object ik):
    #   robot.ik('left_arm', robot.left_hand.tcp('power_grasp_center'), R, p)
    for hand in (robot.left_hand, robot.right_hand):
        hand.toggle_tcp('power_grasp_center', length_scale=0.15, radius_scale=0.25)
        hand.toggle_tcp('pinch_center', length_scale=0.15, radius_scale=0.25)
    base.run()
