"""Visualize an assembly + dual-arm grasp pairing.

Loads the assembly description from `kurabo/assembly/<name>.json`,
loads the saved per-object grasps from `kurabo/grasps/`, places the
two parts at the assembled configuration, then enumerates every
(left_grasp, right_grasp) pair and keeps those that pass a multi-body
collision check (left vs right gripper, left vs mounted part, right
vs base part). SPACE cycles through the surviving pairs.
"""
import os
import sys
import json
import builtins

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.collider.cpu_simd as occs            # noqa: E402
import one.collider.gpu_simd_batch as ocgcb     # noqa: E402
import one.scene.scene_object as osso           # noqa: E402
import one.utils.constant as ouc                # noqa: E402
import one.utils.math as oum                    # noqa: E402
import one.viewer.world as ovw                  # noqa: E402
from one.grasp.serialize import load_grasps     # noqa: E402

from kurabo.grippers.krb_left.krb_left import KRBLeft    # noqa: E402
from kurabo.grippers.krb_right.krb_right import KRBRight  # noqa: E402

ASSEMBLY_PATH = os.path.join(
    _PROJECT_ROOT, "kurabo", "assembly", "con_fe_con_ma.json")
GRASPS_LFT = os.path.join(
    _PROJECT_ROOT, "kurabo", "grasps", "con_fe.json")
GRASPS_RGT = os.path.join(
    _PROJECT_ROOT, "kurabo", "grasps", "con_ma.json")


def _load_assembly(path):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    base_path = os.path.join(_PROJECT_ROOT, "kurabo", spec["base"])
    mount_path = os.path.join(_PROJECT_ROOT, "kurabo", spec["mount"])
    t = np.asarray(
        spec["mount_pose_in_base"]["translation"], dtype=np.float32)
    q = np.asarray(
        spec["mount_pose_in_base"]["quaternion"], dtype=np.float32)
    R = oum.rotmat_from_quat(q).astype(np.float32)
    return base_path, mount_path, R, t


def _build_pair_collision_batch(lft_gripper, rgt_gripper, base_obj, mount_obj):
    """Pairs to check (the 'within-grasp' gripper-vs-own-target are
    already verified during grasp generation):
        - left gripper links vs mounted part
        - right gripper links vs base part
        - left gripper links vs right gripper links
    """
    lft_lnks = lft_gripper.runtime_lnks[1:]
    rgt_lnks = rgt_gripper.runtime_lnks[1:]
    n_lft = len(lft_lnks)
    n_rgt = len(rgt_lnks)
    items = list(lft_lnks) + list(rgt_lnks) + [base_obj, mount_obj]
    base_idx = n_lft + n_rgt
    mount_idx = base_idx + 1
    pairs = []
    for i in range(n_lft):
        pairs.append((i, mount_idx))
        for j in range(n_rgt):
            pairs.append((i, n_lft + j))
    for j in range(n_rgt):
        pairs.append((n_lft + j, base_idx))
    try:
        detector = ocgcb.create_detector()
        batch = ocgcb.build_batch(items, pairs)
        print("assembly collision detector: GPU")
    except Exception as e:
        print(f"assembly collision detector: CPU fallback ({e})")
        detector = occs.create_detector()
        batch = occs.build_batch(items, pairs)
    return detector, batch


def main():
    import pyglet.window.key as pkey
    import one.scene.scene_object_primitive as ossop

    world = ovw.World(cam_pos=(0.15, 0.15, 0.15),
                      cam_lookat_pos=(0.0, 0.0, 0.0))
    builtins.base = world

    base_path, mount_path, mount_R, mount_t = _load_assembly(ASSEMBLY_PATH)
    mount_tf = oum.tf_from_rotmat_pos(mount_R, mount_t).astype(np.float32)

    base_obj = osso.SceneObject.from_file(
        base_path, collision_type=ouc.CollisionType.MESH, is_free=True)
    base_obj.pos = np.zeros(3, dtype=np.float32)
    base_obj.rotmat = np.eye(3, dtype=np.float32)
    base_obj.rgb = (0.85, 0.6, 0.3)

    mount_obj = osso.SceneObject.from_file(
        mount_path, collision_type=ouc.CollisionType.MESH, is_free=True)
    mount_obj.pos = mount_t
    mount_obj.rotmat = mount_R
    mount_obj.rgb = (0.5, 0.7, 0.85)

    lft_grasps = load_grasps(GRASPS_LFT)
    rgt_grasps = load_grasps(GRASPS_RGT)
    print(f"left grasps: {len(lft_grasps)}, "
          f"right grasps: {len(rgt_grasps)}")

    lft = KRBLeft()
    rgt = KRBRight()

    detector, batch = _build_pair_collision_batch(
        lft, rgt, base_obj, mount_obj)
    lft_lo, lft_hi = lft.jaw_range
    rgt_lo, rgt_hi = rgt.jaw_range

    # Enumerate all (li, ri) pairs and keep collision-free ones.
    valid = []
    for li, (lp, _lpre, ljw, _lsc) in enumerate(lft_grasps):
        lft.grip_at(lp[:3, 3], lp[:3, :3],
                    float(np.clip(ljw, lft_lo, lft_hi)))
        for ri, (rp, _rpre, rjw, _rsc) in enumerate(rgt_grasps):
            rp_world = mount_tf @ rp
            rgt.grip_at(rp_world[:3, 3], rp_world[:3, :3],
                        float(np.clip(rjw, rgt_lo, rgt_hi)))
            if detector.detect_collision_batch(batch) is None:
                valid.append((li, ri))
    print(f"collision-free pairs: {len(valid)}")
    if not valid:
        return

    ossop.frame().attach_to(world.scene)
    base_obj.attach_to(world.scene)
    mount_obj.attach_to(world.scene)
    lft.attach_to(world.scene)
    rgt.attach_to(world.scene)

    cursor = [0]

    def _show(i):
        li, ri = valid[i]
        lp, _lpre, ljw, _lsc = lft_grasps[li]
        rp, _rpre, rjw, _rsc = rgt_grasps[ri]
        lft.grip_at(lp[:3, 3], lp[:3, :3],
                    float(np.clip(ljw, lft_lo, lft_hi)))
        rp_world = mount_tf @ rp
        rgt.grip_at(rp_world[:3, 3], rp_world[:3, :3],
                    float(np.clip(rjw, rgt_lo, rgt_hi)))
        msg = (f"pair {i + 1}/{len(valid)}  "
               f"lft={li} rgt={ri}  "
               f"jaw_l={ljw * 1000:.1f}mm jaw_r={rjw * 1000:.1f}mm")
        world.set_caption(msg)
        print(msg)

    _show(0)

    REPEAT_DELAY = 0.4
    held_for = [0.0]

    def _step(dt):
        im = world.input_manager
        if im.is_key_pressed_edge(pkey.SPACE):
            cursor[0] = (cursor[0] + 1) % len(valid)
            _show(cursor[0])
            held_for[0] = 0.0
        elif im.is_key_pressed(pkey.SPACE):
            held_for[0] += dt
            if held_for[0] >= REPEAT_DELAY:
                cursor[0] = (cursor[0] + 1) % len(valid)
                _show(cursor[0])
        else:
            held_for[0] = 0.0

    world.schedule_interval(_step, interval=0.05)
    world.run()


if __name__ == "__main__":
    main()
