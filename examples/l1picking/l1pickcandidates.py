"""Visualize the loaded O6 pinch grasp candidates in the L1 picking scene.

Same scene as l1picking.py (robot at home, table, upright cylinder), but NO
motion planning: the robot stays put and we only show one grasp candidate at a
time as a pair of jaws -- GREEN at the grasp ``pose`` (closed to the grasp
width) and YELLOW at the ``pre`` approach pose (jaw open). Press N to cycle to
the next candidate.
"""
import os
import sys
import builtins

import pyglet.window.key as key

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import one.scene.scene_object_primitive as ossop              # noqa: E402
import one.viewer.world as ovw                                # noqa: E402

from l1picking import build_scene, load_world_grasps, GRASPS_JSON  # noqa: E402


def main(primitive='pinch'):
    robot, table, cyl, ground = build_scene()
    grasps = load_world_grasps(cyl)
    print(f"loaded {len(grasps)} grasps from {os.path.basename(GRASPS_JSON)}")
    if not grasps:
        raise RuntimeError("no grasps loaded")

    base = ovw.World(cam_pos=(1.6, 1.0, 1.7), cam_lookat_pos=(0.35, 0.1, 0.95))
    builtins.base = base
    ossop.frame().attach_to(base.scene)
    for e in (robot, *table, cyl, ground):
        e.attach_to(base.scene)

    # One ghost pair, re-gripped on each N press (robot never moves).
    # ``primitive`` picks which jaw to spawn (e.g. 'pinch', 'tripod').
    jaw_open = float(robot.left_hand.as_jaw(primitive).jaw_range[1])
    ghost_pose = robot.left_hand.as_jaw(primitive)
    ghost_pre = robot.left_hand.as_jaw(primitive)
    ghost_pose.attach_to(base.scene)
    ghost_pre.attach_to(base.scene)

    state = {"i": 0}

    def show(i):
        g = grasps[i]
        pose, pre = g.pose, g.pre_pose
        jw, score = g.provenance["jaw_width"], g.score
        ghost_pose.grip_at(pose[:3, 3], pose[:3, :3], jw)
        ghost_pose.rgb = (0.20, 0.85, 0.25)     # green = grasp pose
        ghost_pre.grip_at(pre[:3, 3], pre[:3, :3], jaw_open)
        ghost_pre.rgb = (0.95, 0.85, 0.15)      # yellow = pre-grasp / approach
        base.scene.dirty = True
        base.set_caption(
            f"candidate {i}/{len(grasps)}  green=pose yellow=pre  "
            f"jaw={jw * 1000:.1f}mm  score={score:.2f}   N: next")
        print(f"candidate {i}/{len(grasps)}  "
              f"jaw={jw * 1000:.1f}mm  score={score:.2f}")

    show(0)
    print("N: next candidate")

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.N):
            state["i"] = (state["i"] + 1) % len(grasps)
            show(state["i"])

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == "__main__":
    main('power')
