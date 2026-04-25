import json
import numpy as np


def save_grasps(grasps, path):
    """Save planned grasps to a JSON file.

    grasps: iterable of (pose_tf, pre_pose_tf, jaw_width, score) as returned
            by `one.grasp.antipodal.antipodal`. pose_tf and pre_pose_tf are
            (4, 4) numpy arrays.
    path: output file path.
    """
    payload = []
    for pose, pre_pose, jaw_width, score in grasps:
        payload.append({
            "pose": np.asarray(pose, dtype=np.float32).tolist(),
            "pre_pose": np.asarray(pre_pose, dtype=np.float32).tolist(),
            "jaw_width": float(jaw_width),
            "score": float(score),
        })
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_grasps(path):
    """Load grasps previously saved by `save_grasps`.

    Returns a list of (pose_tf, pre_pose_tf, jaw_width, score) tuples
    matching the layout produced by `one.grasp.antipodal.antipodal`.
    """
    with open(path, "r") as f:
        payload = json.load(f)
    grasps = []
    for entry in payload:
        pose = np.asarray(entry["pose"], dtype=np.float32)
        pre_pose = np.asarray(entry["pre_pose"], dtype=np.float32)
        grasps.append((pose, pre_pose, float(entry["jaw_width"]),
                       float(entry["score"])))
    return grasps
