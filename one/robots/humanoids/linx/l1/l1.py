import os

import numpy as np

import one.utils.constant as ouc
import one.robots.base.mech_base as orbmb
import one.robots.base.urdf_loader as orul


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
        # tcps (what point to position) -- a tool point on each arm's last link
        tcp_tf = oum.tf_from_rotmat_pos(
            pos=np.array([0.04, 0.0, 0.12], dtype=np.float32))
        self.add_tcp('left_tcp', self.lnk('left_arm_link_6'), tcp_tf)
        self.add_tcp('right_tcp', self.lnk('right_arm_link_6'), tcp_tf)

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
    # arm / hand / manipulator all share the same call:
    #   robot.ik('left_arm', 'left_tcp', tgt_rotmat, tgt_pos)
    robot.toggle_tcp('left_tcp', length_scale=0.18, radius_scale=0.6)
    robot.toggle_tcp('right_tcp', length_scale=0.18, radius_scale=0.6)
    base.run()
