"""Inspect the L1/O6 collision model as convex hulls.

MuJoCo treats mesh geoms as convex for collision.  This script mirrors that
view in the one viewer by rebuilding the robot with CVXHULL collision shapes
and rendering those collision shapes over faint visual meshes.
"""
import os
import sys
import builtins

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc                              # noqa: E402
import one.utils.math as oum                                  # noqa: E402
import one.scene.scene_object_primitive as ossop              # noqa: E402
import one.viewer.world as ovw                                # noqa: E402
import one.robots.base.kine_visualizer as orbkv               # noqa: E402
import one.robots.end_effectors.linkerbot.o6.o6 as oello6     # noqa: E402
import one.robots.humanoids.linx.l1.l1 as l1mod               # noqa: E402


TABLE_ORIGIN = np.array([0.3, 0.0, 0.0], dtype=np.float32)
TABLE_LEN_Y = 1.2
TABLE_WID_X = 0.6
TABLE_TOP_Z = 0.9
TOP_THICK = 0.04
LEG = 0.05
TABLE_RGB = (0.55, 0.42, 0.30)


class L1CvxHull(l1mod.L1):
    """L1 body rebuilt with convex-hull collision shapes."""

    @classmethod
    def _build_structure(cls):
        return l1mod.prepare_mechstruct(collision_type=ouc.CollisionType.CVXHULL)


class O6LeftCvxHull(oello6.O6Left):
    """Left O6 hand rebuilt with convex-hull collision shapes."""

    @classmethod
    def _build_structure(cls):
        return oello6.prepare_mechstruct('left', collision_type=ouc.CollisionType.CVXHULL)


class O6RightCvxHull(oello6.O6Right):
    """Right O6 hand rebuilt with convex-hull collision shapes."""

    @classmethod
    def _build_structure(cls):
        return oello6.prepare_mechstruct('right', collision_type=ouc.CollisionType.CVXHULL)


class L1O6CvxHull(L1CvxHull):
    """L1 plus O6 hands, all rendered with convex-hull collision shapes."""

    def __init__(self, rotmat=None, pos=None, home_qs=None, is_free=True):
        super().__init__(rotmat=rotmat, pos=pos, home_qs=home_qs, is_free=is_free)
        mount_tf = oum.tf_from_pos_rotmat(
            pos=np.array([0.0, 0.0, 0.034], dtype=np.float32))
        self.left_hand = O6LeftCvxHull()
        self.right_hand = O6RightCvxHull()
        self.mount(self.left_hand, self.lnk('left_arm_link_6'), mount_tf, update=True)
        self.mount(self.right_hand, self.lnk('right_arm_link_6'), mount_tf, update=True)


def build_table():
    """Same tabletop + 4 legs used by l1picking.py."""
    ox, oy, _ = TABLE_ORIGIN
    top_cz = TABLE_TOP_Z - TOP_THICK / 2
    parts = [ossop.box(xyz_lengths=(TABLE_WID_X, TABLE_LEN_Y, TOP_THICK),
                       pos=(ox, oy, top_cz), rgb=TABLE_RGB,
                       collision_type=ouc.CollisionType.AABB)]
    leg_h = TABLE_TOP_Z - TOP_THICK
    inset_x = TABLE_WID_X / 2 - LEG / 2
    inset_y = TABLE_LEN_Y / 2 - LEG / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(ossop.box(
                xyz_lengths=(LEG, LEG, leg_h),
                pos=(ox + sx * inset_x, oy + sy * inset_y, leg_h / 2),
                rgb=TABLE_RGB, collision_type=ouc.CollisionType.AABB))
    return parts


def main():
    base = ovw.World(cam_pos=(1.2, 0.8, 1.5), cam_lookat_pos=(0.0, 0.0, 0.9))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    robot = L1O6CvxHull()
    robot.fk(qs=np.zeros(robot.qs.shape, dtype=np.float32))
    robot.alpha = 0.18
    robot.left_hand.alpha = 0.18
    robot.right_hand.alpha = 0.18
    robot.toggle_render_collision = True
    robot.left_hand.toggle_render_collision = True
    robot.right_hand.toggle_render_collision = True
    robot.attach_to(base.scene)

    table = build_table()
    for part in table:
        part.alpha = 0.35
        part.toggle_render_collision = True
        part.attach_to(base.scene)

    kv = orbkv.KineVisualizer(robot, alpha=0.55)
    kv.attach_to(base.scene)

    n_shapes = sum(len(lnk.collisions) for lnk in robot.runtime_lnks)
    n_shapes += sum(len(lnk.collisions) for lnk in robot.left_hand.runtime_lnks)
    n_shapes += sum(len(lnk.collisions) for lnk in robot.right_hand.runtime_lnks)
    n_table_shapes = sum(len(part.collisions) for part in table)
    print(f'L1O6 convex-hull collision shapes: {n_shapes}')
    print(f'table AABB collision shapes: {n_table_shapes}')
    print('Orange transparent meshes are CVXHULL collision models; visual meshes are faint.')

    base.run()


if __name__ == '__main__':
    main()
