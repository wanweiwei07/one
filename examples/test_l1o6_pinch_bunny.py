"""L1O6 humanoid pinch-and-place of a small bunny with the LEFT arm, via the
high-level ``one.manipulation.pick_place``.

Grasps come from ANTIPODAL planning: the O6 left hand is presented as a parallel
jaw (``hand.as_jaw('pinch')``), and ``gen_pick_place`` reasons a grasp feasible
at BOTH the pick and the place pose, then composes home -> pick -> lift ->
transfer -> place -> retreat -- moving ONLY the 'left_arm_waist' chain (the torso
and the other arm stay frozen). The dexterous hand is just another end effector
here: its per-grasp tcp and closure ride in the returned MotionData as
``ee_qpos`` and are replayed with ``hand.fk(qs=ee_qpos)`` -- the same code path a
parallel gripper uses. No hand-rolled IK / motion: that lives in the library.

Run:        py -3.12 examples/test_l1o6_pinch_bunny.py
Headless:   ONE_HEADLESS=1 py -3.12 examples/test_l1o6_pinch_bunny.py
Keys:       F = step one frame   G = play/pause   R = reset
"""
import os

import numpy as np

import one.utils.constant as ouc
import one.utils.math as oum
import one.viewer.world as ovw
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.robots.humanoids.linx.l1.l1 as l1
from one.grasp.antipodal import antipodal

BUNNY_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'bunny_small.stl')   # repo-root asset
CHAIN = 'left_arm_waist'   # waist + left arm (the torso/other arm stay frozen)
TABLE_TOP = 0.93
GRASP = np.array([0.40, 0.18, TABLE_TOP + 0.03], np.float32)   # bunny pick center
PLACE = np.array([0.34, 0.26, TABLE_TOP + 0.03], np.float32)   # bunny place center
LIFT = 0.18                                                    # straight up (m)


def build_scene():
    robot = l1.L1O6()
    # AABB collision: the table is axis-aligned, so its box bounds are exact.
    table = ossop.box(xyz_lengths=(0.5, 0.8, 0.04),
                      pos=(0.45, 0.1, TABLE_TOP - 0.02), rgb=(0.6, 0.45, 0.3),
                      collision_type=ouc.CollisionType.AABB)
    # small bunny (~0.05 m) so it fits a pinch -- pre-scaled mesh on disk
    bunny = osso.SceneObject.from_file(
        BUNNY_STL, collision_type=ouc.CollisionType.MESH,
        rgb=(0.85, 0.7, 0.6), is_floating=True)
    bunny.pos = GRASP.copy()
    ground = ossop.plane(pos=(0, 0, 0.0))
    return robot, table, bunny, ground


def main():
    headless = os.environ.get('ONE_HEADLESS')
    if headless:
        np.random.seed(0)   # reproducible antipodal sampling + RRT for the assert
    robot, table, bunny, ground = build_scene()

    # ONE end-effector binds both steps: grasps are planned for ``hand`` (as a
    # parallel jaw) and gen_pick_place is given the SAME ``hand`` -- the grasps'
    # frozen tcp / qpos only make sense on the EE they were planned for, so
    # sourcing both from one variable avoids a silent mismatch (gen_pick_place
    # also guards against it via each grasp's provenance).
    hand = robot.left_hand

    # antipodal pinch grasps in the bunny's LOCAL frame -- gen_pick_place maps
    # them onto the pick / place poses itself.
    grasps = antipodal(hand.as_jaw('pinch'), bunny, density=0.0008,
                       normal_tol_deg=25, roll_step_deg=30, max_grasps=40,
                       clearance=0.003)
    if not grasps:
        raise RuntimeError('antipodal found no pinch grasps on the bunny')

    pick_pose = bunny.wd_tf                                   # bunny at GRASP
    place_pose = oum.tf_from_pos_rotmat(PLACE, bunny.rotmat)  # same rest pose, moved

    # The dexterous hand is just the end effector. gen_pick_place plans ONLY the
    # 'left_arm_waist' chain (torso/other arm frozen), reasons a grasp feasible at
    # both poses, and returns a MotionData carrying the hand's tcp + closure qpos.
    # ``robot.left_arm`` IS the left arm/manipulator; its end_effector is the
    # left hand (== ``hand``), so the grasps match (gen_pick_place guards it). It
    # plans only its own chain ('left_arm_waist').
    motion = robot.left_arm.pick_place(bunny, grasps, pick_pose, place_pose,
                                       statics=[table, ground], lift_height=LIFT)
    if motion is None:
        raise RuntimeError('no feasible pinch pick-and-place for the left arm')
    print(f'{len(grasps)} pinch grasps -> {len(motion)} waypoints')

    if headless:
        pp = np.array(motion.robot_qpos_list)
        moved = set(np.where(np.abs(pp.max(0) - pp.min(0)) > 1e-6)[0].tolist())
        chain = set(robot.chain(CHAIN).active_jnt_ids.tolist())
        assert moved <= chain, f'non-chain joints moved: {sorted(moved - chain)}'
        assert all(e is not None for e in motion.ee_qpos_list), 'ee_qpos gap'
        print(f'headless OK: only chain {CHAIN} moves ({sorted(moved)}); '
              f'dexterous-hand pick-place via gen_pick_place')
        return

    import pyglet.window.key as key

    base = ovw.World(cam_pos=(2.0, 1.2, 1.6), cam_lookat_pos=(0.3, 0.1, 0.95))
    ossop.frame().attach_to(base.scene)
    for e in (robot, table, bunny, ground):
        e.attach_to(base.scene)
    ossop.frame(pos=pick_pose[:3, 3], rotmat=pick_pose[:3, :3]).attach_to(base.scene)
    ossop.frame(pos=place_pose[:3, 3], rotmat=place_pose[:3, :3]).attach_to(base.scene)
    print('F = step one frame   G = play/pause   R = reset')

    hand = robot.left_hand
    state = {'i': 0, 'playing': False}

    def show(i):
        robot.fk(qs=motion.robot_qpos_list[i])
        ee_qpos = motion.ee_qpos_list[i]
        if ee_qpos is not None:
            hand.fk(qs=ee_qpos)               # replay the hand closure (uniform)
        op = motion.obj_pose_list[i]
        if op is not None:
            bunny.pos, bunny.rotmat = op[:3, 3], op[:3, :3]
        base.scene.dirty = True

    def step_one():
        if state['i'] >= len(motion):
            state['playing'] = False
            return
        show(state['i'])
        state['i'] += 1

    def reset_play():
        state['i'], state['playing'] = 0, False
        show(0)

    reset_play()

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.R):
            reset_play()
            return
        if base.input_manager.is_key_pressed_edge(key.G):
            state['playing'] = not state['playing']
        if base.input_manager.is_key_pressed_edge(key.F):
            state['playing'] = False
            step_one()
        if state['playing']:
            step_one()

    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == '__main__':
    main()
