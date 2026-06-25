"""A manipulation Session: compose primitive verbs into a recipe.

Bind an arm + its collision world + constraints ONCE, then list steps -- the
free verbs ``moveto`` / ``linear`` (motion) and ``grasp`` / ``release`` (state).
Each step plans immediately against the current state; the Session threads the
start config between steps, tracks the gripper opening and the held object, and
accumulates the segments into one ``MotionData``. If any step cannot plan the
Session fails and ``result`` is None.

This is the ergonomic layer over the orthogonal arm verbs (which stay directly
usable): every higher-level shape -- pick-and-place, pick-and-hold, regrasp's
depart-then-approach -- is just a different SEQUENCE of the same four steps, so
the library needs no combinatorial family of named compositions.

    s = arm.plan(collider, constraints=[cable])        # or system.plan(arm, aux=...)
    s.moveto(pick_pose); s.grasp(obj, opening=jw); s.moveto(goal_pose)
    motion = s.result                                  # MotionData (holds at goal) or None
"""
import numpy as np

from one.motion.core.motion_data import MotionData


class Session:
    def __init__(self, arm, collider, *, constraints=(), tcp=None, start_qs=None):
        self.arm = arm
        self.collider = collider
        self.constraints = tuple(constraints)
        self._ee = arm.end_effector
        # default working frame: the EE grasp center (the frame poses target)
        self.tcp = tcp if tcp is not None else self._ee.tcp('grasp_center')
        self._start = (np.asarray(start_qs, dtype=np.float64).copy()
                       if start_qs is not None
                       else np.asarray(arm.body.qs, dtype=np.float64).copy())
        self._ee.open()                                # start open
        self._ee_qpos = np.asarray(self._ee.qs, dtype=np.float32).copy()
        self._held = None
        self._segs = []
        self._failed = False

    # ---- motion steps --------------------------------------------------------
    def moveto(self, goal, **kw):
        """Free RRT to ``goal`` (a config, or a tcp pose IK'd via the bound tcp),
        carrying whatever is held."""
        if not self._failed:
            self._run(self.arm.moveto(
                goal, collider=self.collider, constraints=self.constraints,
                tcp=self.tcp, start_qs=self._start, ee_qpos=self._ee_qpos, **kw))
        return self

    def linear(self, goal_pos, goal_rotmat, **kw):
        """Straight cartesian move of the bound tcp to ``(goal_pos, goal_rotmat)``
        (the mating / insertion leg)."""
        if not self._failed:
            self._run(self.arm.insert(
                goal_pos, goal_rotmat, collider=self.collider, tcp=self.tcp,
                start_qs=self._start, constraints=self.constraints,
                ee_qpos=self._ee_qpos, **kw))
        return self

    # ---- state steps ---------------------------------------------------------
    def grasp(self, obj, *, qpos=None, opening=None, exclude=False):
        """Take ``obj`` into the end effector at closed config ``qpos`` (or a jaw
        ``opening``). ``exclude`` keeps it out of collision (clearance enforced by
        a constraint instead). Subsequent legs carry it."""
        if not self._failed:
            # pose the arm at the CURRENT config so the mount captures the right
            # gripper<->object relative transform (the object must already sit at
            # its grasp pose, e.g. resting at the pick).
            self.arm.body.fk(qs=self._start)
            if qpos is None:
                self._ee.set_opening(opening) if opening is not None else \
                    self._ee.close()
                qpos = np.asarray(self._ee.qs, dtype=np.float32).copy()
            self.arm.grasp(obj, qpos, collider=self.collider, exclude=exclude)
            self._ee_qpos = np.asarray(qpos, dtype=np.float32).copy()
            self._held = obj
        return self

    def release(self, obj):
        """Drop ``obj`` (open + detach); subsequent legs are empty-handed."""
        if not self._failed:
            self.arm.release(obj, collider=self.collider)
            self._ee.open()
            self._ee_qpos = np.asarray(self._ee.qs, dtype=np.float32).copy()
            self._held = None
        return self

    # ---- result --------------------------------------------------------------
    @property
    def result(self):
        """The accumulated MotionData, or None if any step failed / nothing planned."""
        if self._failed or not self._segs:
            return None
        motion = self._segs[0]
        for seg in self._segs[1:]:
            motion = motion + seg
        return motion

    @property
    def last_qs(self):
        """The config the recipe currently ends at (e.g. a hold config to freeze
        this arm while another plans), or None if failed."""
        return None if self._failed else np.asarray(self._start, dtype=np.float32)

    def _run(self, seg):
        if seg is None:
            self._failed = True
            return
        qlist = seg.robot_qpos_list
        if self._held is not None:           # held object follows the gripper (FK)
            obj_list = []
            for q in qlist:
                self.arm.body.fk(qs=q)
                obj_list.append(np.asarray(self._held.tf, dtype=np.float32).copy())
        else:
            obj_list = [None] * len(qlist)
            self.arm.body.fk(qs=qlist[-1])   # leave the arm at the segment end so
                                             # a following grasp mounts correctly
        self._segs.append(MotionData(qlist, seg.ee_qpos_list, obj_list))
        self._start = np.asarray(qlist[-1], dtype=np.float64)
