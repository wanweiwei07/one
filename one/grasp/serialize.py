import json
import numpy as np


def save_grasps(grasps, path, gripper_name=None, object_name=None):
    """Save planned grasps to a JSON file.

    grasps: iterable of (pose_tf, pre_pose_tf, jaw_width, score) as returned
            by `one.grasp.antipodal.antipodal`. pose_tf and pre_pose_tf are
            (4, 4) numpy arrays.
    path: output file path.
    gripper_name: optional gripper class name.
    object_name: optional object STL file name.
    """
    grasp_entries = []
    for pose, pre_pose, jaw_width, score in grasps:
        grasp_entries.append({
            "pose": np.asarray(pose, dtype=np.float32).tolist(),
            "pre_pose": np.asarray(pre_pose, dtype=np.float32).tolist(),
            "jaw_width": float(jaw_width),
            "score": float(score),
        })
    payload = {
        "metadata": {
            "gripper": gripper_name,
            "object": object_name,
        },
        "grasps": grasp_entries,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_grasps(path):
    """Load grasps previously saved by `save_grasps`.

    Returns a list of (pose_tf, pre_pose_tf, jaw_width, score) tuples
    matching the layout produced by `one.grasp.antipodal.antipodal`.
    """
    with open(path, "r") as f:
        payload = json.load(f)
    entries = payload["grasps"] if isinstance(payload, dict) else payload
    grasps = []
    for entry in entries:
        pose = np.asarray(entry["pose"], dtype=np.float32)
        pre_pose = np.asarray(entry["pre_pose"], dtype=np.float32)
        grasps.append((pose, pre_pose, float(entry["jaw_width"]),
                       float(entry["score"])))
    return grasps
