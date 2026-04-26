import os
import sys
import numpy as np

# allow running as `python kurabo/robot.py` from project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.math as oum
import one.utils.constant as ouc
# UR3E module is not yet implemented in `one`; using UR3 (same kinematic family)
import one.robots.manipulators.universal_robots.ur3.ur3 as orur3

from kurabo.grippers.krb_left.krb_left import KRBLeft
from kurabo.grippers.krb_right.krb_right import KRBRight


# Gripper base sits 3mm behind the UR flange face along the flange Z axis.
_KRB_ENGAGE_TF = oum.tf_from_rotmat_pos(pos=(0.0, 0.0, -0.003))


class LeftRobot(orur3.UR3):
    """UR3 with the Kurabo left hand mounted on the flange."""

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        self.gripper = KRBLeft()
        self.engage(self.gripper, engage_tf=_KRB_ENGAGE_TF)


class RightRobot(orur3.UR3):
    """UR3 with the Kurabo right hand mounted on the flange."""

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat=rotmat, pos=pos)
        self.gripper = KRBRight()
        self.engage(self.gripper, engage_tf=_KRB_ENGAGE_TF)


if __name__ == '__main__':
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=(1.6, 1.2, 1.0), cam_lookat_pos=(0.0, 0.0, 0.4))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    lft_robot = LeftRobot(pos=np.array([0.0, -0.3, 0.0], dtype=np.float32))
    rgt_robot = RightRobot(pos=np.array([0.0, 0.3, 0.0], dtype=np.float32))

    lft_robot.attach_to(base.scene)
    lft_robot.gripper.attach_to(base.scene)
    rgt_robot.attach_to(base.scene)
    rgt_robot.gripper.attach_to(base.scene)

    for arm in (lft_robot, rgt_robot):
        tcp_tf = arm.gl_tcp_tf
        ossop.frame(pos=tcp_tf[:3, 3], rotmat=tcp_tf[:3, :3],
                    color_mat=ouc.CoordColor.MYC).attach_to(base.scene)
        ee_tf = arm.gripper.runtime_root_lnk.tf
        ossop.frame(pos=ee_tf[:3, 3], rotmat=ee_tf[:3, :3]).attach_to(base.scene)

    builtins.lft_robot = lft_robot
    builtins.rgt_robot = rgt_robot
    base.run()
