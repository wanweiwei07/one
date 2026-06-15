"""Plan O6 LEFT-hand antipodal grasps of an upright cylinder (dia 0.05, height
0.3) and save them to JSON for the L1 picking demo (l1picking.py).

The O6 hand is presented to ``antipodal`` as a parallel jaw via
``spawn_jaw('pinch')`` (the only opposition wide enough -- the pinch opens to
~0.082 m, comfortably over the 0.05 m cylinder; the 'power' grasp is an envelope
with no opposing pads so it cannot be antipodal-planned). Each grasp is a pose
of the pinch *grasp center* in the cylinder's LOCAL frame plus a jaw width;
l1picking.py maps these onto the cylinder wherever it stands on the table.

Viewer shows each grasp as a pair: GREEN jaw at the grasp ``pose`` and YELLOW
jaw at the ``pre`` pre-grasp (approach) pose, simultaneously.
Keys (viewer):  N = next pose/pre pair   (Q/Esc closes)
Run headless (just plan + save, no window):  ONE_HEADLESS=1
"""
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import one.utils.constant as ouc                              # noqa: E402
import one.scene.scene_object_primitive as ossop              # noqa: E402
from one.robots.end_effectors.linkerbot.o6.o6 import O6Left   # noqa: E402
from one.grasp.antipodal import antipodal                     # noqa: E402
from one.grasp.serialize import save_grasps                   # noqa: E402

# Cylinder: dia 0.05 (radius 0.025), height 0.3, upright (along +Z), centred at
# its own origin -- the same local frame l1picking.py transforms onto the table.
CYL_RADIUS = 0.025
CYL_HEIGHT = 0.30
OUT_JSON = os.path.join(_THIS, "o6_cylinder_grasps.json")


def make_cylinder():
    return ossop.cylinder(
        spos=(0.0, 0.0, -CYL_HEIGHT / 2), epos=(0.0, 0.0, CYL_HEIGHT / 2),
        radius=CYL_RADIUS, segments=24,
        collision_type=ouc.CollisionType.MESH, is_free=True,
        rgb=(0.6, 0.7, 0.5))


def main(primitive='pinch'):
    headless = bool(os.environ.get("ONE_HEADLESS"))
    hand = O6Left()
    cyl = make_cylinder()

    jaw = hand.spawn_jaw(primitive)
    grasps = antipodal(jaw, cyl, density=0.0015, normal_tol_deg=25,
                       roll_step_deg=30, max_grasps=60, clearance=0.003)
    print(f"antipodal: {len(grasps)} {primitive} grasps on the cylinder "
          f"(dia {2 * CYL_RADIUS}, h {CYL_HEIGHT})")
    if not grasps:
        raise RuntimeError(f"no antipodal {primitive} grasp found on the cylinder")

    save_grasps(grasps, OUT_JSON, gripper_name="O6Left",
                object_name=f"cylinder_d{2 * CYL_RADIUS}_h{CYL_HEIGHT}")
    print(f"saved {len(grasps)} {primitive} grasps -> {OUT_JSON}")

    if headless:
        return

    import builtins
    import one.viewer.world as ovw
    import pyglet.window.key as key

    base = ovw.World(cam_pos=(0.35, 0.0, 0.18), cam_lookat_pos=(0.0, 0.0, 0.0))
    builtins.base = base
    ossop.frame(length_scale=0.5).attach_to(base.scene)
    cyl.attach_to(base.scene)

    # Two independent jaw views, shown at the same time: GREEN at the grasp
    # ``pose`` (closed to the grasp width) and YELLOW at the ``pre`` pre-grasp
    # pose (opened, the approach stand-off). N cycles to the next pose/pre pair.
    jaw_open = float(jaw.jaw_range[1])
    jaw_pose = hand.spawn_jaw(primitive)          # green = final grasp
    jaw_pre = hand.spawn_jaw(primitive)           # yellow = pre-grasp / approach
    jaw_pose.rgb = (0.20, 0.85, 0.25)
    jaw_pre.rgb = (0.95, 0.85, 0.15)
    jaw_pose.attach_to(base.scene)
    jaw_pre.attach_to(base.scene)
    print(f"showing pair 0/{len(grasps)} (green=pose, yellow=pre); press N")

    state = {"i": 0}

    def show(i):
        pose, pre, jw, score = grasps[i]
        # antipodal plans in the cylinder's LOCAL (zero-pose) frame and returns
        # local grasps (that is what we save for l1picking). Map them onto the
        # cylinder's actual placement before gripping -- the cylinder here sits
        # at cyl.wd_tf (spos=-h/2), so without this the jaws would be ~0.15 off.
        wpose = cyl.wd_tf @ pose
        wpre = cyl.wd_tf @ pre
        jaw_pose.grip_at(wpose[:3, 3], wpose[:3, :3], jw)
        jaw_pre.grip_at(wpre[:3, 3], wpre[:3, :3], jaw_open)
        jaw_pose.rgb = (0.20, 0.85, 0.25)       # re-assert after re-grip
        jaw_pre.rgb = (0.95, 0.85, 0.15)
        base.scene.dirty = True
        base.set_caption(f"pair {i}/{len(grasps)}  green=pose yellow=pre  "
                         f"jaw={jw * 1000:.1f}mm  score={score:.2f}   N: next")

    show(0)

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.N):
            state["i"] = (state["i"] + 1) % len(grasps)
            show(state["i"])

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == "__main__":
    main('power')
