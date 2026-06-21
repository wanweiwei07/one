"""A plain, composable motion container.

``MotionData`` is the connective tissue between motion primitives: every
generator (RRT solve, cartesian descent, approach-depart) returns one, and they
compose with ``+`` so a pick = approach + grasp + depart is just three segments
added together. It is deliberately *pure data* -- three parallel lists, no robot
reference, no mesh generation, no backup/restore state dance. Rendering meshes
is the viewer's job (draw each ``jv`` on demand); timing is a separate
``trajectory.time_param`` step.

Three parallel per-waypoint lists (all length ``len(self)``):
    jv_list       -- joint configs (full qs, the planners' vocabulary)
    ev_list       -- gripper opening at that waypoint (None = unspecified)
    obj_pose_list -- carried-object world tf at that waypoint (None = none held)

``a + b`` concatenates the two; if ``b`` starts where ``a`` ends (same config)
the duplicate seam waypoint is dropped, so segments chain without a stutter.
"""
import numpy as np


class MotionData:
    def __init__(self, jv_list=None, ev_list=None, obj_pose_list=None):
        self.jv_list = [np.asarray(q, dtype=np.float32)
                        for q in (jv_list or [])]
        n = len(self.jv_list)
        self.ev_list = list(ev_list) if ev_list is not None else [None] * n
        self.obj_pose_list = (list(obj_pose_list)
                              if obj_pose_list is not None else [None] * n)
        if len(self.ev_list) != n or len(self.obj_pose_list) != n:
            raise ValueError(
                f"parallel lists must match jv_list ({n}): "
                f"ev={len(self.ev_list)}, obj={len(self.obj_pose_list)}")

    @classmethod
    def from_jpath(cls, jv_list, ev_value=None, obj_pose=None):
        """Build a segment from a joint path with a CONSTANT gripper opening and
        (optionally) a constant carried-object pose across all its waypoints."""
        jv_list = list(jv_list)
        n = len(jv_list)
        return cls(jv_list,
                   [ev_value] * n if ev_value is not None else None,
                   [obj_pose] * n if obj_pose is not None else None)

    def __len__(self):
        return len(self.jv_list)

    def __iter__(self):
        return iter(self.jv_list)

    def __getitem__(self, i):
        return (self.jv_list[i], self.ev_list[i], self.obj_pose_list[i])

    def is_empty(self):
        return len(self.jv_list) == 0

    def copy(self):
        return MotionData(self.jv_list, self.ev_list, self.obj_pose_list)

    def __add__(self, other):
        if not isinstance(other, MotionData):
            return NotImplemented
        if self.is_empty():
            return other.copy()
        if other.is_empty():
            return self.copy()
        # drop the seam waypoint if b starts exactly where a ended
        drop = 1 if np.allclose(self.jv_list[-1], other.jv_list[0],
                                atol=1e-4) else 0
        return MotionData(
            self.jv_list + other.jv_list[drop:],
            self.ev_list + other.ev_list[drop:],
            self.obj_pose_list + other.obj_pose_list[drop:])

    def __repr__(self):
        evs = {e for e in self.ev_list if e is not None}
        return (f"MotionData(n={len(self)}, "
                f"ev={'∅' if not evs else sorted(evs)})")
