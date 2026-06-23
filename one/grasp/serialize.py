"""Save / load / world-transform a list of :class:`~one.grasp.grasp.Grasp`.

Grasps are authored and saved in the object's LOCAL frame; place them in the
world with :func:`transform_grasps` (the dual of :func:`load_grasps`) given the
object's world transform. The on-disk record is fully self-contained -- it
freezes the tcp loc_tf and the hand qpos -- so a reload needs no gripper-state
re-derivation (see the ``Grasp`` module docstring). The gripper / object names
are metadata only, for a load-time sanity check.
"""
import json

import numpy as np

from one.grasp.grasp import Grasp


def save_grasps(grasps, path, gripper_name=None, object_name=None):
    """Save planned grasps to a JSON file.

    grasps: iterable of :class:`Grasp` (as returned by ``antipodal`` /
            ``polypodal`` / ``monocontact``).
    path: output file path.
    gripper_name: optional gripper class name (metadata).
    object_name: optional object STL file name (metadata).
    """
    payload = {
        "metadata": {"gripper": gripper_name, "object": object_name},
        "grasps": [g.to_dict() for g in grasps],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_grasps(path):
    """Load grasps previously saved by :func:`save_grasps`.

    Returns a list of :class:`Grasp`, in the object's local frame.
    """
    with open(path, "r") as f:
        payload = json.load(f)
    entries = payload["grasps"] if isinstance(payload, dict) else payload
    return [Grasp.from_dict(e) for e in entries]


def transform_grasps(grasps, tf):
    """Map object-LOCAL grasps into the world by an object world transform.

    Given the object's world transform (e.g. ``scene_obj.wd_tf``) this places
    each grasp where the object actually stands -- ``pose`` / ``pre_pose`` are
    transformed; the hand-relative ``tcp`` / ``qpos`` and the ``score`` /
    ``provenance`` pass through untouched.

    grasps: iterable of :class:`Grasp`.
    tf: (4, 4) object world transform.

    Returns a list of transformed :class:`Grasp`.
    """
    tf = np.asarray(tf, dtype=np.float32)
    return [g.transformed(tf) for g in grasps]
