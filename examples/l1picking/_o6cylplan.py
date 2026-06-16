"""Shared O6 cylinder grasp-planning core, used by o6cylstlplanning.py (mesh
loaded from cylinder.stl) and o6scnobjplanning.py (an ossop.cylinder primitive
of the same size). Both call plan_save_show with the SAME parameters so the only
variable is the cylinder source -- letting us check the grasp results match.

cylinder.stl is dia 0.025 (radius 0.0125), height 0.075, base at z=0.
"""
import os
import sys
import builtins

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.scene.scene_object_primitive as ossop              # noqa: E402
from one.robots.end_effectors.linkerbot.o6.o6 import O6Left   # noqa: E402
from one.grasp.antipodal import antipodal                     # noqa: E402
from one.grasp.serialize import save_grasps                   # noqa: E402

# same dims as cylinder.stl
CYL_RADIUS = 0.0125
CYL_HEIGHT = 0.075

# identical planning parameters for both sources
PLAN_KW = dict(density=0.0015, normal_tol_deg=25, roll_step_deg=30,
               max_grasps=60, clearance=0.003)


def plan_save_show(cyl, out_json, label, primitive='pinch'):
    hand = O6Left()
    jaw = hand.spawn_jaw(primitive)
    grasps = antipodal(jaw, cyl, **PLAN_KW)
    pos = np.array([g[0][:3, 3] for g in grasps]) if grasps else np.zeros((0, 3))
    print(f"[{label}] antipodal: {len(grasps)} {primitive} grasps")
    if len(pos):
        print(f"[{label}] grasp-pose pos bbox  x{np.round([pos[:,0].min(),pos[:,0].max()],4)}"
              f"  y{np.round([pos[:,1].min(),pos[:,1].max()],4)}"
              f"  z{np.round([pos[:,2].min(),pos[:,2].max()],4)}")
    if not grasps:
        raise RuntimeError(f"[{label}] no antipodal grasp found")
    save_grasps(grasps, out_json, gripper_name="O6Left", object_name=label)
    print(f"[{label}] saved {len(grasps)} grasps -> {os.path.basename(out_json)}")

    if os.environ.get("ONE_HEADLESS"):
        return grasps

    import one.utils.constant as ouc
    import one.viewer.world as ovw
    import pyglet.window.key as key

    base = ovw.World(cam_pos=(0.18, 0.0, 0.06), cam_lookat_pos=(0.0, 0.0, 0.04))
    builtins.base = base
    ossop.frame(length_scale=0.3).attach_to(base.scene)
    cyl.attach_to(base.scene)
    jaw_open = float(jaw.jaw_range[1])
    jaw_pose = hand.spawn_jaw(primitive)
    jaw_pre = hand.spawn_jaw(primitive)
    jaw_pose.rgb = (0.20, 0.85, 0.25)
    jaw_pre.rgb = (0.95, 0.85, 0.15)
    jaw_pose.attach_to(base.scene)
    jaw_pre.attach_to(base.scene)
    state = {"i": 0}

    def show(i):
        pose, pre, jw, score = grasps[i]
        wpose = cyl.wd_tf @ pose
        wpre = cyl.wd_tf @ pre
        jaw_pose.grip_at(wpose[:3, 3], wpose[:3, :3], jw)
        jaw_pre.grip_at(wpre[:3, 3], wpre[:3, :3], jaw_open)
        jaw_pose.rgb = (0.20, 0.85, 0.25)
        jaw_pre.rgb = (0.95, 0.85, 0.15)
        base.scene.dirty = True
        base.set_caption(f"[{label}] pair {i}/{len(grasps)}  green=pose "
                         f"yellow=pre  jaw={jw*1000:.1f}mm   N: next")

    show(0)

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.N):
            state["i"] = (state["i"] + 1) % len(grasps)
            show(state["i"])

    base.schedule_interval(tick, interval=0.05)
    base.run()
    return grasps
