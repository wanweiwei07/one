"""A plain, composable motion container.

``MotionData`` is the connective tissue between motion primitives: every
generator (RRT solve, cartesian descent, approach-depart) returns one, and they
compose with ``+`` so a pick = approach + grasp + depart is just three segments
added together. It is deliberately *pure data* -- three parallel lists, no robot
reference, no mesh generation, no backup/restore state dance. Rendering meshes
is the viewer's job (draw each config on demand); timing is a separate
``trajectory.time_param`` step.

Three parallel per-waypoint lists (all length ``len(self)``):
    robot_qpos_list -- robot / arm joint configs (full qs, the planners'
                       vocabulary).
    ee_qpos_list    -- end-effector joint config (the gripper / hand qs) at that
                       waypoint, or ``None`` to leave the EE as-is. Replay is
                       uniform across end effectors: ``ee.fk(qs=ee_qpos)`` -- a
                       parallel gripper's qs is e.g. ``[w/2, w/2]``, a dexterous
                       hand's is its full finger config. (Open -> closed marks a
                       grasp, closed -> open a release.)
    obj_pose_list   -- carried-object world tf at that waypoint (``None`` = none
                       held).

``a + b`` concatenates the lists; if ``b`` starts where ``a`` ends (same config)
the duplicate seam waypoint is dropped, so segments chain without a stutter.
"""
import numpy as np


def _qpos(v):
    return None if v is None else np.asarray(v, dtype=np.float32)


class MotionData:
    def __init__(self, robot_qpos_list=None, ee_qpos_list=None,
                 obj_pose_list=None):
        self.robot_qpos_list = [np.asarray(q, dtype=np.float32)
                                for q in (robot_qpos_list or [])]
        n = len(self.robot_qpos_list)
        self.ee_qpos_list = ([_qpos(e) for e in ee_qpos_list]
                             if ee_qpos_list is not None else [None] * n)
        self.obj_pose_list = (list(obj_pose_list)
                              if obj_pose_list is not None else [None] * n)
        if len(self.ee_qpos_list) != n or len(self.obj_pose_list) != n:
            raise ValueError(
                f"parallel lists must match robot_qpos_list ({n}): "
                f"ee={len(self.ee_qpos_list)}, obj={len(self.obj_pose_list)}")

    @classmethod
    def from_jpath(cls, robot_qpos_list, ee_qpos=None, obj_pose=None):
        """Build a segment from a robot joint path with a CONSTANT end-effector
        config and (optionally) a constant carried-object pose across all of its
        waypoints."""
        robot_qpos_list = list(robot_qpos_list)
        n = len(robot_qpos_list)
        return cls(robot_qpos_list,
                   [ee_qpos] * n if ee_qpos is not None else None,
                   [obj_pose] * n if obj_pose is not None else None)

    def __len__(self):
        return len(self.robot_qpos_list)

    def __iter__(self):
        return iter(self.robot_qpos_list)

    def __getitem__(self, i):
        return (self.robot_qpos_list[i], self.ee_qpos_list[i],
                self.obj_pose_list[i])

    def is_empty(self):
        return len(self.robot_qpos_list) == 0

    def copy(self):
        return MotionData(self.robot_qpos_list, self.ee_qpos_list,
                          self.obj_pose_list)

    def __add__(self, other):
        if not isinstance(other, MotionData):
            return NotImplemented
        if self.is_empty():
            return other.copy()
        if other.is_empty():
            return self.copy()
        # drop the seam waypoint if b starts exactly where a ended
        drop = 1 if np.allclose(self.robot_qpos_list[-1],
                                other.robot_qpos_list[0], atol=1e-4) else 0
        return MotionData(
            self.robot_qpos_list + other.robot_qpos_list[drop:],
            self.ee_qpos_list + other.ee_qpos_list[drop:],
            self.obj_pose_list + other.obj_pose_list[drop:])

    def __repr__(self):
        n_ee = sum(e is not None for e in self.ee_qpos_list)
        return f"MotionData(n={len(self)}, ee_qpos@{n_ee}/{len(self)})"
