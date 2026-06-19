"""L1O6 humanoid pinch-and-place of a small bunny with the LEFT arm, where the
grasp poses come from ANTIPODAL planning (not a hand-picked top-down pose).

The O6 left hand is presented to ``antipodal`` as a parallel jaw via
``spawn_jaw('pinch')``; each returned grasp is a world-frame pose of the pinch
center + a jaw width. For a chosen grasp we solve the pick-and-place keyframes
(IK targeting a per-grasp center tcp) and plan the free-space hops:

    home -> pre-grasp -> (approach) grasp -> pinch+attach -> lift
         -> carry -> (approach) place -> release -> retreat -> home

Everything moves only the 'left_arm_waist' chain (other joints frozen). Keys:
    N = advance to the next reachable grasp (re-solves + re-plans its motion)
    R = replay the current grasp's motion

Run headless validation (no window):  set ONE_HEADLESS=1
"""
import os

import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.collider.mj_collider as ocm
import one.motion.core.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
import one.motion.interpolation.joint as omij
import one.robots.base.tcp as orbt
import one.robots.humanoids.linx.l1.l1 as l1
import one.grasp._common as ogc
from one.grasp.antipodal import antipodal

BUNNY_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'bunny_small.stl')   # repo-root asset
CHAIN = 'left_arm_waist'   # waist + left arm (vs 'left_arm' for arm only)
# weight of the most-proximal chain joint (the waist) in the nearest-solution
# metric: high -> prefer redundant solutions that keep the torso still and let
# the arm do the reaching (like a person), instead of twisting the waist.
WAIST_WEIGHT = 6.0
TABLE_TOP = 0.93
GRASP = np.array([0.40, 0.18, TABLE_TOP + 0.03], np.float32)   # bunny center
PLACE = np.array([0.34, 0.26, TABLE_TOP + 0.03], np.float32)   # drop center
UP = np.array([0.0, 0.0, 0.18], np.float32)   # standoff above grasp/place
STANDOFF = 0.10   # pre-grasp back-off along the grasp approach axis
PRE_OPEN = 0.5    # pre-grasp jaw openness (fraction to max); matches antipodal


# ----------------------------------------------------------------------------
def build_scene():
    robot = l1.L1O6()
    # AABB collision so the table carries an explicit box mesh -- the exact
    # hand-vs-table filter reads .collisions (one's tri-tri detector needs it).
    # AABB is exact here: the table is axis-aligned (rotmat = identity), matching
    # the box's own xyz definition. (MESH/CVXHULL break MJCF: no mesh name/file.)
    table = ossop.box(xyz_lengths=(0.5, 0.8, 0.04),
                      pos=(0.45, 0.1, TABLE_TOP - 0.02), rgb=(0.6, 0.45, 0.3),
                      collision_type=ouc.CollisionType.AABB)
    # small bunny (~0.05 m) so it fits a pinch -- pre-scaled mesh on disk
    bunny = osso.SceneObject.from_file(
        BUNNY_STL, collision_type=ouc.CollisionType.MESH,
        rgb=(0.85, 0.7, 0.6), is_free=True)
    bunny.pos = GRASP.copy()
    ground = ossop.plane(pos=(0, 0, 0.0))
    return robot, table, bunny, ground


def make_collider(robot, table, ground):
    """Collider for MOTION planning: robot vs table + ground + self. The bunny is
    the grasp TARGET (hand contact with it is intended), so it is deliberately NOT
    in this collider -- otherwise hand-vs-bunny would read as a collision. The
    precise hand-vs-environment gate is ``filter_grasps`` (exact tri-tri)."""
    mjc = ocm.MJCollider()
    for e in (robot, table, ground):
        mjc.append(e)
    mjc.actors = [robot]
    mjc.compile(margin=0.0, auto_acm=True)   # disable resting self-collisions
    return mjc


def chain_planning_context(robot, mjc, chain_name):
    """A PlanningContext over the robot's full qs with every joint NOT on
    ``chain_name`` frozen at its home value -> the planner only explores that
    chain, while the collider still gets a valid full-body configuration."""
    c = robot._compiled
    chain = robot.chain(chain_name)
    lo = c.jlmt_low_by_idx.astype(np.float64).copy()
    hi = c.jlmt_high_by_idx.astype(np.float64).copy()
    home = robot.qs.astype(np.float64).copy()
    free = np.zeros(c.n_jnts, dtype=bool)
    free[chain.active_jnt_ids] = True
    lo[~free] = home[~free]
    hi[~free] = home[~free]
    return omppc.PlanningContext(collider=mjc, joint_limits=(lo, hi))


def plan_grasps(robot, bunny):
    """Antipodal pinch grasps on the bunny, best score first. antipodal plans in
    the object's LOCAL frame (its collision mesh), so we map every pose by
    ``bunny.wd_tf`` to world before returning -- that is the frame IK needs.
    Returns (jaw, grasps): the spawned parallel-jaw view (holds the jw->closure
    calibration) and the list of world-frame (pose, pre_pose, jw, score)."""
    jaw = robot.left_hand.spawn_jaw('pinch')
    local = antipodal(jaw, bunny, density=0.0008, normal_tol_deg=25,
                      roll_step_deg=30, max_grasps=40, clearance=0.003)
    tf = bunny.wd_tf
    grasps = [(tf @ pose, tf @ pre, jw, sc) for pose, pre, jw, sc in local]
    return jaw, grasps


def filter_grasps(jaw, grasps, env_sobj):
    """Keep only grasps whose HAND clears the environment (the table) at both the
    grasp pose (closed to jw) and the pre-grasp pose (opened to pre_jw).

    Uses the SAME exact tri-tri detector as antipodal -- the hand is non-convex
    and MuJoCo would collide-test its convex hull (fingertips' gaps filled),
    falsely rejecting valid grasps; tri-tri is exact. It tests the FREE hand
    alone (arm-independent), so it is cheap and pre-screens grasps before the
    expensive IK / motion planning. The bunny (grasp target) is not the
    environment, so intended hand-bunny contact is never counted here."""
    det, batch = ogc.build_ee_target_detector(jaw, env_sobj)
    jaw_max = float(jaw.jaw_range[1])
    kept = []
    for pose, pre, jw, sc in grasps:
        pre_jw = jw + PRE_OPEN * (jaw_max - jw)
        jaw.grip_at(pose[:3, 3], pose[:3, :3], jw)
        if det.detect_collision_batch(batch) is not None:
            continue
        jaw.grip_at(pre[:3, 3], pre[:3, :3], pre_jw)
        if det.detect_collision_batch(batch) is not None:
            continue
        kept.append((pose, pre, jw, sc))
    return kept


def ik_config(robot, ctx, pos, rotmat, tcp, collision_free=True, ref=None):
    """IK for ``tcp`` at (pos, rotmat). Among the solutions (optionally the
    collision-free ones), return the one CLOSEST in joint space to ``ref``
    (defaults home), weighting the waist heavily so the torso stays still.
    Returns full qs (n_jnts,) or None."""
    chain = robot.chain(CHAIN)
    if ref is None:
        ref = robot.qs
    ref_active = chain.extract_active_qs(ref)
    w = np.ones_like(ref_active)
    w[0] = WAIST_WEIGHT   # active_jnt_ids order -> [0] is the proximal waist joint
    sols = robot.ik(pos, rotmat, chain=CHAIN, tcp=tcp,
                    ref_qs=ref_active, max_solutions=8)
    best, best_d = None, None
    for s in sols:
        if collision_free and not ctx.is_state_valid(s.astype(np.float64)):
            continue
        d = float(np.linalg.norm(w * (chain.extract_active_qs(s) - ref_active)))
        if best_d is None or d < best_d:
            best, best_d = s.astype(np.float64), d
    return best


# ----------------------------------------------------------------------------
def build_motion(robot, ctx, planner, jaw, grasp):
    """Build the pick-and-place trajectory for one antipodal ``grasp`` =
    (pose, pre_pose, jw, score). Returns (traj, grasp_idx, release_idx, amount)
    or None if any keyframe is unreachable / colliding."""
    pose, _pre_pose, jw, _sc = grasp
    rot = pose[:3, :3]
    g = pose[:3, 3]
    # per-grasp center tcp on the MOUNTED hand: the calibrated pinch center for
    # this closure (hand-base offset), so IK puts the hand where the pads will
    # close around the object. +z of the grasp pose is the approach axis.
    center_tcp = orbt.TCP(robot.left_hand.runtime_root_lnk,
                          oum.tf_from_pos_rotmat(pos=jaw.grasp_center_at(jw)))
    home = robot.qs.astype(np.float64).copy()
    pre_pos = g - STANDOFF * rot[:, 2]            # back off along the approach
    pre = ik_config(robot, ctx, pre_pos, rot, center_tcp)
    grasp_q = ik_config(robot, ctx, g, rot, center_tcp,
                        collision_free=False, ref=pre)
    lift = ik_config(robot, ctx, g + UP, rot, center_tcp,
                     collision_free=False, ref=grasp_q)
    carry = ik_config(robot, ctx, PLACE + UP, rot, center_tcp, ref=lift)
    place = ik_config(robot, ctx, PLACE, rot, center_tcp,
                      collision_free=False, ref=carry)
    retreat = ik_config(robot, ctx, PLACE + UP, rot, center_tcp,
                        collision_free=False, ref=place)
    if any(x is None for x in (pre, grasp_q, lift, carry, place, retreat)):
        return None

    approach = planner.solve(home, pre, max_iters=4000)   # planned free-space
    if not approach:
        return None
    traj = list(approach)
    traj += list(omij.interp_by_step(pre, grasp_q))    # approach
    grasp_idx = len(traj) - 1
    traj += list(omij.interp_by_step(grasp_q, lift))   # depart (carrying)
    carry_hop = planner.solve(lift, carry, max_iters=4000)    # planned carry
    if not carry_hop:
        return None
    traj += carry_hop
    traj += list(omij.interp_by_step(carry, place))    # approach
    release_idx = len(traj) - 1
    traj += list(omij.interp_by_step(place, retreat))  # depart
    return_hop = planner.solve(retreat, home, max_iters=4000)  # planned return
    if not return_hop:
        return None
    traj += return_hop
    amount = float(jaw._amount_for(jw))           # jw -> pinch closure amount
    return traj, grasp_idx, release_idx, amount


def first_reachable(robot, ctx, planner, jaw, grasps, start=0):
    """Build the motion for the first reachable grasp at/after ``start`` (wrap
    around). Returns (grasp_index, motion) or (None, None)."""
    for k in range(len(grasps)):
        idx = (start + k) % len(grasps)
        motion = build_motion(robot, ctx, planner, jaw, grasps[idx])
        if motion is not None:
            return idx, motion
    return None, None


# ----------------------------------------------------------------------------
def main():
    headless = os.environ.get('ONE_HEADLESS')
    if headless:
        np.random.seed(0)   # reproducible antipodal sampling for the assert
    robot, table, bunny, ground = build_scene()
    mjc = make_collider(robot, table, ground)   # bunny excluded (grasp target)
    ctx = chain_planning_context(robot, mjc, CHAIN)
    planner = ompr.RRTConnectPlanner(pln_ctx=ctx, extend_step_size=np.pi / 36,
                                     goal_bias=0.3)
    jaw, grasps = plan_grasps(robot, bunny)
    if not grasps:
        raise RuntimeError('antipodal found no pinch grasps on the bunny')
    n_all = len(grasps)
    grasps = filter_grasps(jaw, grasps, table)   # exact hand-vs-table pre-screen
    print(f'antipodal: {n_all} pinch grasps, {len(grasps)} clear of the table')
    if not grasps:
        raise RuntimeError('no antipodal grasp clears the table')

    gi, motion = first_reachable(robot, ctx, planner, jaw, grasps)
    if motion is None:
        raise RuntimeError('no antipodal grasp is reachable by the left arm')
    traj, grasp_idx, release_idx, _amount = motion
    print(f'grasp {gi} (score {grasps[gi][3]:.2f}): {len(traj)} waypoints '
          f'(pinch@{grasp_idx}, release@{release_idx})')

    if headless:
        chain = robot.chain(CHAIN)
        pp = np.array(traj)
        moved = set(np.where(np.abs(pp.max(0) - pp.min(0)) > 1e-6)[0].tolist())
        assert moved <= set(chain.active_jnt_ids.tolist()), \
            f'non-chain joints moved: {sorted(moved - set(chain.active_jnt_ids.tolist()))}'
        print(f'headless OK: only chain {CHAIN} joints move ({sorted(moved)}); '
              f'antipodal-planned pick-place')
        return

    import pyglet.window.key as key

    base = ovw.World(cam_pos=(2.0, 1.2, 1.6), cam_lookat_pos=(0.3, 0.1, 0.95))
    ossop.frame().attach_to(base.scene)
    for e in (robot, table, bunny, ground):
        e.attach_to(base.scene)
    bunny_home = (bunny.pos.copy(), bunny.rotmat.copy())
    print('N = next grasp, R = replay current')

    state = {'gi': gi, 'motion': motion, 'i': 0, 'held': False}

    def reset_play():
        if state['held']:
            robot.left_hand.release(bunny)   # unmount + reopen
            state['held'] = False
        robot.left_hand.open_hand()
        bunny.pos, bunny.rotmat = bunny_home[0].copy(), bunny_home[1].copy()
        robot.fk(qs=state['motion'][0][0])
        state['i'] = 0

    def select_next():
        reset_play()
        gi2, motion2 = first_reachable(robot, ctx, planner, jaw, grasps,
                                       start=state['gi'] + 1)
        if motion2 is None:
            print('no other reachable grasp'); return
        state['gi'], state['motion'] = gi2, motion2
        t, gidx, ridx, _ = motion2
        print(f'grasp {gi2} (score {grasps[gi2][3]:.2f}): {len(t)} waypoints')
        reset_play()

    reset_play()

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.N):
            select_next(); return
        if base.input_manager.is_key_pressed_edge(key.R):
            reset_play(); return
        traj_i, grasp_idx_i, release_idx_i, amount = state['motion']
        i = state['i']
        if i >= len(traj_i):
            return
        robot.fk(qs=traj_i[i])
        if i == grasp_idx_i and not state['held']:
            robot.left_hand.grasp(bunny, primitive='pinch', amount=amount)
            state['held'] = True
        if i == release_idx_i and state['held']:
            robot.left_hand.release(bunny)
            state['held'] = False
        state['i'] += 1

    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == '__main__':
    main()
