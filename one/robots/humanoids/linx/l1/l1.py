import os

import numpy as np

import one.utils.constant as ouc
import one.utils.math as oum
import one.robots.base.mech_base as orbmb
import one.robots.base.urdf_loader as orul
import one.robots.base.kine.anaik as orbka
import one.robots.end_effectors.linkerbot.o6.o6 as oello6


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
    """Linx L1 upper-body humanoid, bare (no end-effectors). Arms tip at the
    ``*_arm_link_6`` flanges; mount hands/grippers there yourself, or use
    :class:`L1O6` for the version pre-fitted with two O6 hands."""

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None, home_qs=None, is_floating=True):
        super().__init__(rotmat=rotmat, pos=pos,
                         home_qs=home_qs, is_floating=is_floating)
        lm = self.structure.lnk_map
        # named chains (which joints move) -- base/tip are structure links
        self.add_chain('left_arm', lm['waist_link2'], lm['left_arm_link_6'],
                       solver=orbka.S456X12)
        self.add_chain('right_arm', lm['waist_link2'], lm['right_arm_link_6'],
                       solver=orbka.S456X12)
        self.add_chain('left_arm_waist', lm['waist_link1'], lm['left_arm_link_6'])
        self.add_chain('right_arm_waist', lm['waist_link1'], lm['right_arm_link_6'])
        self.add_chain('neck', lm['waist_link2'], lm['neck_link2'])

    def lnk(self, name):
        lidx = self.structure.compiled.lidx_map[self.structure.lnk_map[name]]
        return self.runtime_lnks[lidx]

    def jnt(self, name):
        return self.structure.jnt_map[name]


class L1O6(L1):
    """Linx L1 with two dexterous Linkerbot O6 hands mounted on the arm
    flanges. Grasp targets come from the hands' center tcps via cross-object
    ik, e.g.
        robot.ik(p, R, chain='left_arm', tcp=robot.left_hand.tcp('power_center'))
    """

    def __init__(self, rotmat=None, pos=None, home_qs=None, is_floating=True):
        super().__init__(rotmat=rotmat, pos=pos,
                         home_qs=home_qs, is_floating=is_floating)
        # O6 EEs mounted on each arm flange (the *_linkerbot_mount_joint,
        # xyz=(0,0,0.034), that used to live in the body URDF).
        mount_tf = oum.tf_from_pos_rotmat(
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


if __name__ == '__main__':
    import one.viewer.world as ovw
    import one.robots.base.kine_visualizer as orbkv

    base = ovw.World(cam_pos=(2.2, 1.4, 1.6),
                     cam_lookat_pos=(0.0, 0.0, 0.9))
    robot = L1O6()
    robot.attach_to(base.scene)
    # render the meshes semi-transparent so the kinematic skeleton shows through
    robot.alpha = 0.3
    robot.left_hand.alpha = 0.3
    robot.right_hand.alpha = 0.3
    # stick (skeleton) model of the whole mechanism (chain=None -> all joints)
    kv = orbkv.KineVisualizer(robot, alpha=0.9)
    kv.attach_to(base.scene)
    base.run()
