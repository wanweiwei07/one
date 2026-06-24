"""The ``Grasp`` record -- a self-contained, gripper-agnostic grasp.

A grasp is FROZEN at planning time so that replaying it (place the hand, drive
IK, motion-plan) needs NO re-derivation from the live gripper's state. This is
the deliberate cure for the old ``(pose, pre_pose, jaw_width, score)`` tuple,
whose ``pose`` was only usable if the consumer re-derived the tcp from the
gripper -- ambiguous the moment one opening (``jaw_width``) can map to several
grasp-center frames (a dexterous hand's pinch vs. tripod share an opening but
not a tcp orientation).

What is frozen, and why each piece is needed:

  ``pose`` / ``pre_pose`` -- the IK target frame (the tcp frame) in the OBJECT's
      LOCAL coordinates. Map to the world with ``obj_wd_tf @ pose``. This is what
      the planners consume; it is NOT the hand base.
  ``tcp`` -- the tcp's offset RELATIVE TO THE HAND ROOT LINK (a ``loc_tf``),
      snapshotted at plan time. Freezing it (rather than re-deriving via
      ``jaw.grasp_center_tcp``) makes the same tcp reconstruct regardless of
      the gripper's current mode / closure. ``pose @ inv(tcp)`` recovers the
      hand base; ``make_tcp(gripper)`` rebuilds the IK tcp.
  ``qpos`` / ``pre_qpos`` -- the hand's FULL joint configuration (closed on the
      object at the grasp, opened part-way at the pre-grasp). Full qpos, not a
      scalar ``jaw_width``, is the gripper-agnostic truth: a dexterous hand's
      tripod and pinch differ at the same nominal opening. Replacing the
      width->qpos lambda the consumers used to carry.
  ``score`` -- planner score, best first.
  ``provenance`` -- OPTIONAL, NON-authoritative metadata (e.g.
      ``{'mode': 'pinch', 'jaw_width': 0.04}``). Replay never depends on it;
      it exists for humans, for filtering by grasp type, and for live closure
      control (``set_opening`` needs the mode bound). The geometry above is
      the single source of truth.

The grasp is bound to a specific gripper (``tcp`` / ``qpos`` are that hand's
data); ``serialize`` records the gripper name for a load-time sanity check.
Producers build these via :meth:`Grasp.from_jaw` (opposing-jaw / dexterous
pinch) or :meth:`Grasp.from_tool` (single-contact suction / tip tools).
"""
from dataclasses import dataclass, field

import numpy as np

import one.robots.base.tcp as orbt


def _tf(a):
    return np.asarray(a, dtype=np.float32).reshape(4, 4)


def _vec(a):
    return np.asarray(a, dtype=np.float32).reshape(-1)


@dataclass
class Grasp:
    """A frozen, gripper-agnostic grasp record (see module docstring)."""

    pose: np.ndarray                 # (4,4) tcp frame in object-local coords
    pre_pose: np.ndarray             # (4,4) pre-grasp tcp frame, object-local
    tcp: np.ndarray                  # (4,4) tcp loc_tf relative to hand root link
    qpos: np.ndarray                 # (k,) hand joint config at the grasp
    pre_qpos: np.ndarray             # (k,) hand joint config at the pre-grasp
    score: float = 0.0
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pose = _tf(self.pose)
        self.pre_pose = _tf(self.pre_pose)
        self.tcp = _tf(self.tcp)
        self.qpos = _vec(self.qpos)
        self.pre_qpos = _vec(self.pre_qpos)
        self.score = float(self.score)

    # ---- reconstruction helpers (no gripper state re-derivation) -----------
    def make_tcp(self, gripper):
        """Rebuild the IK tcp on ``gripper`` from the frozen ``tcp`` loc_tf.

        Identical for a parallel jaw and a dexterous hand -- the loc_tf is data,
        not a function of the hand's current mode / closure."""
        return orbt.TCP(gripper.runtime_root_lnk, self.tcp.copy())

    def base_pose(self, obj_wd_tf):
        """The hand ROOT-link world pose that realizes this grasp on an object
        at ``obj_wd_tf``: ``(obj @ pose) @ inv(tcp)``."""
        world = _tf(obj_wd_tf) @ self.pose
        return world @ np.linalg.inv(self.tcp.astype(np.float64))

    def transformed(self, tf):
        """A copy with ``pose`` / ``pre_pose`` mapped by world transform ``tf``
        (object-local -> world). ``tcp`` / ``qpos`` are hand-relative, untouched."""
        tf = _tf(tf)
        return Grasp(tf @ self.pose, tf @ self.pre_pose, self.tcp.copy(),
                     self.qpos.copy(), self.pre_qpos.copy(), self.score,
                     dict(self.provenance))

    # ---- (de)serialization -------------------------------------------------
    def to_dict(self):
        return {
            "pose": self.pose.tolist(),
            "pre_pose": self.pre_pose.tolist(),
            "tcp": self.tcp.tolist(),
            "qpos": self.qpos.tolist(),
            "pre_qpos": self.pre_qpos.tolist(),
            "score": self.score,
            "provenance": _jsonable(self.provenance),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["pose"], d["pre_pose"], d["tcp"], d["qpos"],
                   d["pre_qpos"], d.get("score", 0.0),
                   dict(d.get("provenance", {})))

    # ---- producer-side constructors ---------------------------------------
    @classmethod
    def from_jaw(cls, jaw, pose, pre_pose, jaw_width, pre_jaw_width,
                 score=0.0, extra_provenance=None):
        """Snapshot a grasp from a parallel jaw (a real gripper, or a dexterous
        hand's ``as_jaw`` :class:`~one.robots.end_effectors.ee_mixins.JawView`).

        ``pose`` / ``pre_pose`` are the grasp-center (tcp) frames the producer
        computed. The tcp loc_tf is frozen from ``jaw.grasp_center_tcp`` and the
        full qpos at the grasp and pre-grasp openings are snapshotted from
        ``jaw.qpos`` (so a dexterous hand's per-closure config is captured, not
        just a scalar width). ``provenance`` records the grasp ``mode`` (a
        dexterous hand's bound primitive, else None) and the ``jaw_width``."""
        tcp = np.asarray(jaw.grasp_center_tcp(jaw_width).loc_tf,
                         dtype=np.float32)
        jaw.set_opening(jaw_width)
        qpos = np.asarray(jaw.qpos, dtype=np.float32).copy()
        jaw.set_opening(pre_jaw_width)
        pre_qpos = np.asarray(jaw.qpos, dtype=np.float32).copy()
        prov = {"jaw_width": float(jaw_width),
                "ee": type(getattr(jaw, "hand", jaw)).__name__}
        mode = getattr(jaw, "mode", None)
        if mode is not None:
            prov["mode"] = mode
        if extra_provenance:
            prov.update(extra_provenance)
        return cls(pose, pre_pose, tcp, qpos, pre_qpos, score, prov)

    @classmethod
    def from_tool(cls, tool, pose, pre_pose, tcp_name="tip", score=0.0,
                  extra_provenance=None):
        """Snapshot a single-contact grasp from a suction / tip ``tool``.

        There is no opposing closure: ``qpos`` / ``pre_qpos`` are the tool's own
        (typically fixed) joint config, and the tcp loc_tf is the named contact
        tcp. ``provenance`` records the ``tcp_name``."""
        tcp = np.asarray(tool.tcp(tcp_name).loc_tf, dtype=np.float32)
        qpos = np.asarray(tool.qs, dtype=np.float32).copy()
        prov = {"tcp_name": tcp_name, "ee": type(tool).__name__}
        if extra_provenance:
            prov.update(extra_provenance)
        return cls(pose, pre_pose, tcp, qpos, qpos.copy(), score, prov)


def _jsonable(obj):
    """Recursively convert numpy scalars / arrays in provenance to plain JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
