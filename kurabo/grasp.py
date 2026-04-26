"""Compute antipodal grasps for one of the two kurabo hand/object pairs
and dump the result to JSON. Toggle the active configuration by
commenting the unwanted line in the CONFIG block below.

Mapping:
    LEFT  hand (KRBLeft)  -> con_fe.stl  -> grasps/con_fe.json
    RIGHT hand (KRBRight) -> con_ma.stl  -> grasps/con_ma.json
"""
import os
import sys

import numpy as np

# Inject project root before any cross-package imports.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.geom.geometry as ogg  # noqa: E402
import one.scene.geometry_ops as osgop  # noqa: E402
import one.scene.render_model as osrm  # noqa: E402
import one.scene.scene_object as osso  # noqa: E402
import one.utils.constant as ouc  # noqa: E402
from one.grasp.antipodal import antipodal  # noqa: E402
from one.grasp.polypodal import polypodal  # noqa: E402
from one.grasp.serialize import save_grasps  # noqa: E402

from kurabo.grippers.krb_left.krb_left import KRBLeft  # noqa: E402
from kurabo.grippers.krb_right.krb_right import KRBRight  # noqa: E402


def _obj(name):
    return os.path.join(_PROJECT_ROOT, "kurabo", "objects", f"{name}.stl")


def _out(name):
    return os.path.join(_PROJECT_ROOT, "kurabo", "grasps", f"{name}.json")


def _pre_pose_from_grasp(gripper, pose):
    pre_pose = pose.copy()
    retreat_dist = 0.5 * np.linalg.norm(gripper.loc_tcp_tf[:3, 3])
    pre_pose[:3, 3] = pose[:3, 3] - retreat_dist * pose[:3, 2]
    return pre_pose


def plan_grasps(gripper, obj, config):
    pattern = np.asarray(gripper.contact_pattern, dtype=np.float32)
    if pattern.ndim != 2 or pattern.shape[1] != 3:
        raise ValueError("gripper.contact_pattern must be (N, 3)")
    if pattern.shape[0] == 1:
        return antipodal(
            gripper=gripper, target_sobj=obj,
            exclude_regions=config.get("exclude_regions"),
            **config["params"])

    poly_params = config.get("polypodal_params", {})
    raw_grasps = polypodal(
        gripper=gripper, target_sobj=obj,
        exclude_regions=config.get("exclude_regions"),
        **poly_params)
    return [
        (pose, _pre_pose_from_grasp(gripper, pose), jaw_width, 1.0)
        for pose, jaw_width in raw_grasps
    ]


# Per-hand antipodal params. `clearance` is per-side gap; KRB finger
# contact faces sit ~2mm inside the link origin, so a small NEGATIVE
# clearance keeps the finger flush with the object surface.
LEFT_CONFIG = dict(
    gripper_cls=KRBLeft,
    obj_path=_obj("con_fe"),
    out_path=_out("con_fe"),
    params=dict(
        density=0.003, normal_tol_deg=0, roll_step_deg=20,
        clearance=0.00075, max_grasps=300),
    polypodal_params=dict(
        n_samples=5000, normal_tol_deg=15, distance_tol=0.001,
        surface_density_factor=1, clearance=0.0003, min_thickness=0.001,
        max_thickness=0.05),
    # Keep only z ∈ [-22mm, -5mm] AND remove the two side flanges
    # (|x| > 5.9mm with y < 2mm) of con_fe.
    exclude_regions=[
        [((0.0, 0.0, -0.005), (0.0, 0.0, -1.0))],   # cut: z > -5mm
        [((0.0, 0.0, -0.022), (0.0, 0.0,  1.0))],   # cut: z < -22mm
        [                                            # cut: x > 5.9 AND y < 2
            ((0.0059, 0.0,   0.0), (-1.0, 0.0, 0.0)),
            ((0.0,    0.002, 0.0), ( 0.0, 1.0, 0.0)),
        ],
        [                                            # cut: x < -5.9 AND y < 2
            ((-0.0059, 0.0,   0.0), (1.0, 0.0, 0.0)),
            (( 0.0,    0.002, 0.0), (0.0, 1.0, 0.0)),
        ],
        [                                            # cut: x > 5.9 AND y > 9.6
            ((0.0059, 0.0,    0.0), (-1.0,  0.0, 0.0)),
            ((0.0,    0.0096, 0.0), ( 0.0, -1.0, 0.0)),
        ],
        [                                            # cut: x < -5.9 AND y > 9.6
            ((-0.0059, 0.0,    0.0), (1.0,  0.0, 0.0)),
            (( 0.0,    0.0096, 0.0), (0.0, -1.0, 0.0)),
        ],
        [                                            # cut: -6.2 < x < 6.2 AND 1.8 < y < 10.1
            (( 0.0062, 0.0,    0.0), ( 1.0,  0.0, 0.0)),
            ((-0.0062, 0.0,    0.0), (-1.0,  0.0, 0.0)),
            (( 0.0,    0.0018, 0.0), ( 0.0, -1.0, 0.0)),
            (( 0.0,    0.0101, 0.0), ( 0.0,  1.0, 0.0)),
        ],
    ],
)
RIGHT_CONFIG = dict(
    gripper_cls=KRBRight,
    obj_path=_obj("con_ma"),
    out_path=_out("con_ma"),
    params=dict(
        density=0.003, normal_tol_deg=0, roll_step_deg=20,
        clearance=0.0003, max_grasps=300),
    polypodal_params=dict(
        n_samples=10000, normal_tol_deg=15, distance_tol=0.001,
        surface_density_factor=1, clearance=0.0003, min_thickness=0.001,
        max_thickness=0.05),
    # Keep only y ∈ [1mm, 9mm] of con_ma.
    exclude_regions=[
        [((0.0, 0.001, 0.0), (0.0,  1.0, 0.0))],   # cut: y < 1mm
        [((0.0, 0.009, 0.0), (0.0, -1.0, 0.0))],   # cut: y > 9mm
    ],
)

# === toggle which hand/object pair to compute ===
CONFIG = LEFT_CONFIG
# CONFIG = RIGHT_CONFIG


if __name__ == "__main__":
    import builtins
    import pyglet.window.key as pkey
    import one.viewer.world as ovw

    print(f"Computing antipodal grasps on {CONFIG['obj_path']}...")
    gripper = CONFIG["gripper_cls"]()
    obj = osso.SceneObject.from_file(
        CONFIG["obj_path"], collision_type=ouc.CollisionType.MESH,
        is_free=True)
    obj.pos = np.zeros(3, dtype=np.float32)
    obj.rotmat = np.eye(3, dtype=np.float32)
    grasps = plan_grasps(gripper, obj, CONFIG)
    print(f"  found {len(grasps)} grasps")
    save_grasps(
        grasps, CONFIG["out_path"],
        gripper_name=type(gripper).__name__,
        object_name=os.path.basename(CONFIG["obj_path"]))
    print(f"Wrote {CONFIG['out_path']}")
    if not grasps:
        sys.exit(0)

    # Visualize: object at origin, single ghost gripper.
    # Press SPACE to step to the next grasp candidate.
    base = ovw.World(cam_pos=(0.25, 0.25, 0.2),
                     cam_lookat_pos=(0.0, 0.0, 0.0))
    builtins.base = base
    obj.rgb = (0.85, 0.6, 0.3)
    obj.alpha = 0.4
    obj.attach_to(base.scene)

    # Red overlay: the surviving triangles that antipodal sampled on.
    regs = CONFIG.get("exclude_regions")
    if regs:
        full_geom = obj.collisions[0].geom
        kept_vs, kept_fs = osgop.clip_mesh(full_geom.vs, full_geom.fs, regs)
        if len(kept_fs):
            kept_geom = ogg.gen_geom_from_raw(
                kept_vs.astype(np.float32),
                kept_fs.astype(np.int32))
            kept_obj = osso.SceneObject()
            kept_obj.add_visual(
                osrm.RenderModel(geom=kept_geom, rgb=(1.0, 0.0, 0.0)),
                auto_make_collision=False)
            kept_obj.attach_to(base.scene)

    ghost = gripper.clone()
    ghost.attach_to(base.scene)
    cursor = [0]

    def show(i):
        pose, _pre, jw, sc = grasps[i]
        ghost.grip_at(pose[:3, 3], pose[:3, :3], jw)
        print(f"[grasp {i+1}/{len(grasps)}] jaw={jw*1000:.1f}mm "
              f"score={sc:.3f}")

    show(0)

    # Keyboard-style repeat: tap = +1, hold > REPEAT_DELAY = auto-repeat.
    REPEAT_DELAY = 0.4
    held_for = [0.0]

    def step(dt):
        im = base.input_manager
        if im.is_key_pressed_edge(pkey.SPACE):
            cursor[0] = (cursor[0] + 1) % len(grasps)
            show(cursor[0])
            held_for[0] = 0.0
        elif im.is_key_pressed(pkey.SPACE):
            held_for[0] += dt
            if held_for[0] >= REPEAT_DELAY:
                cursor[0] = (cursor[0] + 1) % len(grasps)
                show(cursor[0])
        else:
            held_for[0] = 0.0

    base.schedule_interval(step, interval=0.05)
    base.run()
