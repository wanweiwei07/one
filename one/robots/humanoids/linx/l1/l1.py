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

    def lnk(self, name):
        lidx = self.structure.compiled.lidx_map[self.structure.lnk_map[name]]
        return self.runtime_lnks[lidx]

    def jnt(self, name):
        return self.structure.jnt_map[name]

    def chain(self, base_lnk_name, tip_lnk_name):
        return self.structure.get_chain(
            self.structure.lnk_map[base_lnk_name],
            self.structure.lnk_map[tip_lnk_name],
        )

    @property
    def left_arm_chain(self):
        return self.chain('waist_link2', 'left_arm_link_6')

    @property
    def right_arm_chain(self):
        return self.chain('waist_link2', 'right_arm_link_6')

    @property
    def left_arm_waist_chain(self):
        return self.chain('waist_link1', 'left_arm_link_6')

    @property
    def right_arm_waist_chain(self):
        return self.chain('waist_link1', 'right_arm_link_6')

    @property
    def neck_chain(self):
        return self.chain('waist_link2', 'neck_link2')


if __name__ == '__main__':
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(2.2, 1.4, 1.6),
                     cam_lookat_pos=(0.0, 0.0, 0.9))
    robot = L1()
    robot.attach_to(base.scene)
    loc_tcp_tf = np.eye(4, dtype=np.float32)
    loc_tcp_tf[:3, 3] = np.array([0.04, 0.0, 0.12], dtype=np.float32)
    left_tcp = ossop.frame_from_tf(
        loc_tcp_tf,
        length_scale=0.18,
        radius_scale=0.6,
    )
    right_tcp = ossop.frame_from_tf(
        loc_tcp_tf,
        length_scale=0.18,
        radius_scale=0.6,
    )
    left_tcp.attach_to(robot.lnk('left_arm_link_6'))
    right_tcp.attach_to(robot.lnk('right_arm_link_6'))
    base.run()
