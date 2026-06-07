"""L1O6 humanoid pinch-and-place of a small bunny with the LEFT arm.

Pipeline (everything moves only the 'left_arm' chain; waist / right arm / neck /
hands stay put -- chain-restricted planning by freezing the other joints):

    home -> pre-grasp -> (approach) grasp -> pinch + attach -> lift
         -> carry -> (approach) place -> release -> retreat -> home

Free-space hops (home->pre-grasp, lift->carry, retreat->home) are planned with
RRTConnect against an MJ collider whose resting self-collisions are auto-ACM'd;
the short straight approaches/departs are joint-space interpolations.

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
import one.motion.probabilistic.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
import one.robots.humanoids.linx.l1.l1 as l1

BUNNY_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'bunny_small.stl')   # repo-root asset
CHAIN = 'left_arm_waist'   # waist + left arm (vs 'left_arm' for arm only)
TCP = 'pinch_center'
# approach orientation = the hand's rest orientation rolled this little about
# its own finger axis: the pure rest orientation can't reach the table targets,
# and +30 deg is the smallest reachable deviation (vs ~100 deg for an arbitrary
# top-down), so the wrist barely changes from its natural pose.
APPROACH_ROLL = np.radians(30.0)
# weight of the most-proximal chain joint (the waist) in the nearest-solution
# metric: high -> prefer redundant solutions that keep the torso still and let
# the arm do the reaching (like a person), instead of twisting the waist.
WAIST_WEIGHT = 6.0
TABLE_TOP = 0.93
GRASP = np.array([0.40, 0.18, TABLE_TOP + 0.03], np.float32)   # bunny center
PLACE = np.array([0.34, 0.26, TABLE_TOP + 0.03], np.float32)   # drop center
UP = np.array([0.0, 0.0, 0.18], np.float32)   # standoff above grasp/place


# ----------------------------------------------------------------------------
def build_scene():
    robot = l1.L1O6()
    table = ossop.box(xyz_lengths=(0.5, 0.8, 0.04),
                      pos=(0.45, 0.1, TABLE_TOP - 0.02), rgb=(0.6, 0.45, 0.3))
    # small bunny (~0.05 m) so it fits a pinch -- pre-scaled mesh on disk
    bunny = osso.SceneObject.from_file(
        BUNNY_STL, collision_type=ouc.CollisionType.MESH,
        rgb=(0.85, 0.7, 0.6), is_free=True)
    bunny.pos = GRASP.copy()
    ground = ossop.plane(pos=(0, 0, 0.0))
    return robot, table, bunny, ground


def make_collider(robot, table, bunny, ground):
    mjc = ocm.MJCollider()
    for e in (robot, table, bunny, ground):
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


def ik_config(robot, ctx, pos, approach, collision_free=True, ref=None):
    """IK for the pinch tcp at (pos, approach). Among the solutions (optionally
    the collision-free ones), return the one CLOSEST in joint space to ``ref``
    (defaults to the home posture) -- with a redundant chain (waist + arm) there
    are far branches (waist twisted, wrist flipped, reaching from behind); the
    nearest one keeps the robot near its current posture. Returns full qs (20,)
    or None."""
    chain = robot.chain(CHAIN)
    if ref is None:
        ref = robot.qs   # home while keyframes are being solved
    ref_active = chain.extract_active_qs(ref)
    w = np.ones_like(ref_active)
    w[0] = WAIST_WEIGHT   # active_jnt_ids order -> [0] is the proximal waist joint
    sols = robot.ik(pos, approach, chain=CHAIN, tcp=robot.left_hand.tcp(TCP),
                    ref_qs=ref_active, max_solutions=8)
    best, best_d = None, None
    for s in sols:
        if collision_free and not ctx.is_state_valid(s.astype(np.float64)):
            continue
        d = float(np.linalg.norm(w * (chain.extract_active_qs(s) - ref_active)))
        if best_d is None or d < best_d:
            best, best_d = s.astype(np.float64), d
    return best


def jtraj(q0, q1, step=np.deg2rad(3.0)):
    n = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / step)))
    return [q0 + (q1 - q0) * t for t in np.linspace(0, 1, n)]


def plan_segment(planner, q0, q1):
    path = planner.solve(q0, q1, max_iters=4000)
    return path if path else jtraj(q0, q1)   # fall back to interpolation


# ----------------------------------------------------------------------------
def build_motion(robot, ctx):
    """Return (trajectory, grasp_index, release_index): a list of full-qs
    waypoints plus the indices at which to pinch+attach / release the bunny."""
    planner = ompr.RRTConnectPlanner(pln_ctx=ctx, extend_step_size=np.pi / 36,
                                     goal_bias=0.3)
    home = robot.qs.astype(np.float64).copy()
    # approach orientation: the hand's rest orientation (robot is at home here)
    # rolled APPROACH_ROLL about its own finger axis -- reachable while keeping
    # the wrist near its natural pose (no flip / reaching from behind).
    rest_R = robot.left_hand.tcp(TCP).rotmat.copy()
    appr = oum.rotmat_from_axangle(rest_R[:, 2], APPROACH_ROLL) @ rest_R

    pre = ik_config(robot, ctx, GRASP + UP, appr)
    grasp = ik_config(robot, ctx, GRASP, appr, collision_free=False, ref=pre)
    lift = ik_config(robot, ctx, GRASP + UP, appr, collision_free=False, ref=grasp)
    carry = ik_config(robot, ctx, PLACE + UP, appr, ref=lift)
    place = ik_config(robot, ctx, PLACE, appr, collision_free=False, ref=carry)
    retreat = ik_config(robot, ctx, PLACE + UP, appr, collision_free=False, ref=place)
    names = ['pre', 'grasp', 'lift', 'carry', 'place', 'retreat']
    got = [pre, grasp, lift, carry, place, retreat]
    if any(g is None for g in got):
        raise RuntimeError('IK failed for: ' +
                           ', '.join(n for n, g in zip(names, got) if g is None))

    traj = []
    traj += plan_segment(planner, home, pre)     # planned free-space
    traj += jtraj(pre, grasp)                     # approach
    grasp_idx = len(traj) - 1
    traj += jtraj(grasp, lift)                    # depart (carrying)
    traj += plan_segment(planner, lift, carry)    # planned carry
    traj += jtraj(carry, place)                   # approach
    release_idx = len(traj) - 1
    traj += jtraj(place, retreat)                 # depart
    traj += plan_segment(planner, retreat, home)  # planned return
    return traj, grasp_idx, release_idx


# ----------------------------------------------------------------------------
def main():
    headless = os.environ.get('ONE_HEADLESS')
    robot, table, bunny, ground = build_scene()
    mjc = make_collider(robot, table, bunny, ground)
    ctx = chain_planning_context(robot, mjc, CHAIN)
    traj, grasp_idx, release_idx = build_motion(robot, ctx)
    print(f'trajectory: {len(traj)} waypoints '
          f'(pinch@{grasp_idx}, release@{release_idx})')

    if headless:
        # sanity: no joint OUTSIDE the planned chain ever moves
        chain = robot.chain(CHAIN)
        pp = np.array(traj)
        moved = set(np.where(np.abs(pp.max(0) - pp.min(0)) > 1e-6)[0].tolist())
        assert moved <= set(chain.active_jnt_ids.tolist()), \
            f'non-chain joints moved: {sorted(moved - set(chain.active_jnt_ids.tolist()))}'
        print(f'headless OK: only chain {CHAIN} joints move '
              f'({sorted(moved)}); full pick-place planned')
        return

    import pyglet.window.key as key

    base = ovw.World(cam_pos=(2.0, 1.2, 1.6), cam_lookat_pos=(0.3, 0.1, 0.95))
    ossop.frame().attach_to(base.scene)
    for e in (robot, table, bunny, ground):
        e.attach_to(base.scene)
    robot.left_hand.toggle_tcp(TCP, length_scale=0.12, radius_scale=0.25)
    bunny_home = (bunny.pos.copy(), bunny.rotmat.copy())
    print('press R to replay')

    state = {'i': 0, 'held': False}

    def reset():
        if state['held']:
            robot.left_hand.release(bunny)   # unmount + reopen
            state['held'] = False
        robot.left_hand.open_hand()
        bunny.pos, bunny.rotmat = bunny_home[0].copy(), bunny_home[1].copy()
        robot.fk(qs=traj[0])
        state['i'] = 0

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.R):
            reset()
            return
        i = state['i']
        if i >= len(traj):
            return
        robot.fk(qs=traj[i])
        if i == grasp_idx and not state['held']:
            robot.left_hand.pinch(1.0)
            robot.left_hand.grasp(bunny, primitive='pinch')
            state['held'] = True
        if i == release_idx and state['held']:
            robot.left_hand.release(bunny)
            state['held'] = False
        state['i'] += 1

    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == '__main__':
    main()
