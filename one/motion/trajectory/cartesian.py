import warnings
import numpy as np
import one.utils.math as oum


def cartesian_to_jtraj(
    robot,
    start_rotmat,
    start_pos,
    goal_rotmat=None,
    goal_pos=None,
    pos_step=0.01,
    rot_step=np.deg2rad(2.0),
    ref_qs=None,
    chain='main',
    tcp='flange',
    ctx=None,
):
    """
    Convert Cartesian straight-line trajectory to joint trajectory using IK.

    Parameters
    ----------
    robot : MechBase
        Must provide `ik(chain, tcp, ...)`, `chain(name)` and `qs`.
    chain, tcp : str
        Names of the registered chain / tcp to solve against (default the
        single-arm convention 'main' / 'flange').
    start_rotmat : array-like, shape (3, 3)
    start_pos : array-like, shape (3,)
    goal_rotmat : array-like, shape (3, 3), optional
    goal_pos : array-like, shape (3,), optional
    pos_step : float
    rot_step : float
    ref_qs : array-like, optional
        Initial IK seed. If None, use robot.qs.
    ctx : PlanningContext, optional
        When given, each newly solved config is checked against the previous one
        with `ctx.is_motion_valid` (densified at `ctx.cd_step_size`). The first
        segment that collides (or leaves bounds) aborts and returns `None` for
        `q_seq`. A joint-space line between two IK-valid endpoints does NOT keep
        the tcp on the straight Cartesian path, so this rejects segments whose
        densified motion clips an obstacle. `qs` produced by `robot.ik` must live
        in the same state space as `ctx` (same convention the planners rely on).
        Default None disables checking (pure geometry + IK).

    Returns
    -------
    q_seq : np.ndarray | None
        Joint sequence with shape (N, n_jnts) if successful, otherwise None.
    pose_seq : tuple[np.ndarray, np.ndarray]
        Interpolated `(pos_seq, rotmat_seq)` used for IK.
    """
    pos_seq, rotmat_seq = line_by_step_size(
        start_rotmat=start_rotmat,
        start_pos=start_pos,
        goal_rotmat=goal_rotmat,
        goal_pos=goal_pos,
        pos_step=pos_step,
        rot_step=rot_step,
    )
    ik_chain = robot.chain(chain)
    if ref_qs is None:
        ref_qs = np.asarray(robot.qs, dtype=np.float32)
    else:
        ref_qs = np.asarray(ref_qs, dtype=np.float32)
    ref_qs_active = ik_chain.extract_active_qs(ref_qs)
    q_list = []
    prev_qs = None
    for pos, rotmat in zip(pos_seq, rotmat_seq):
        qs_list = robot.ik(pos, rotmat, chain=chain, tcp=tcp,
                           max_solutions=1, ref_qs=ref_qs_active)
        if not qs_list:
            return None, (pos_seq, rotmat_seq)
        qs = np.asarray(qs_list[0], dtype=np.float32)
        # gate each densified segment as soon as its endpoint is solved, so a
        # colliding move aborts before wasting IK on the rest of the line.
        if ctx is not None and prev_qs is not None \
                and not ctx.is_motion_valid(prev_qs, qs):
            return None, (pos_seq, rotmat_seq)
        q_list.append(qs)
        prev_qs = qs
        ref_qs_active = ik_chain.extract_active_qs(qs)
    return np.asarray(q_list, dtype=np.float32), (pos_seq, rotmat_seq)


def line_by_step_size(
    start_rotmat,
    start_pos,
    goal_rotmat=None,
    goal_pos=None,
    pos_step=0.01,
    rot_step=np.deg2rad(2.0),
):
    """
    Cartesian straight-line interpolation with automatic sample count.

    The number of samples is decided by both translation and rotation:
    N = max(ceil(||dp|| / pos_step), ceil(||drot|| / rot_step)) + 1.

    Parameters
    ----------
    start_rotmat : array-like, shape (3, 3)
    start_pos : array-like, shape (3,)
    goal_rotmat : array-like, shape (3, 3), optional
    goal_pos : array-like, shape (3,), optional
    pos_step : float
        Maximum translation per segment in meters.
    rot_step : float
        Maximum rotation per segment in radians.

    Returns
    -------
    pos_seq : np.ndarray, shape (N, 3), dtype float32
    rotmat_seq : np.ndarray, shape (N, 3, 3), dtype float32
    """
    if pos_step <= 0:
        raise ValueError(f"pos_step must be > 0, got {pos_step}")
    if rot_step <= 0:
        raise ValueError(f"rot_step must be > 0, got {rot_step}")
    start_rotmat = np.asarray(start_rotmat, dtype=np.float32)
    start_pos = np.asarray(start_pos, dtype=np.float32)
    goal_rotmat, goal_pos, is_single = _resolve_goal(
        start_rotmat=start_rotmat,
        start_pos=start_pos,
        goal_rotmat=goal_rotmat,
        goal_pos=goal_pos,
    )
    if is_single:
        return (
            start_pos.reshape(1, 3).astype(np.float32),
            start_rotmat.reshape(1, 3, 3).astype(np.float32),
        )
    pos_dist = float(np.linalg.norm(goal_pos - start_pos))
    rot_dist = float(
        np.linalg.norm(oum.delta_rotvec_between_rotmats(start_rotmat, goal_rotmat))
    )
    n_pos = int(np.ceil(pos_dist / float(pos_step)))
    n_rot = int(np.ceil(rot_dist / float(rot_step)))
    n_steps = max(n_pos, n_rot) + 1
    n_steps = max(n_steps, 2)
    return line_by_n_steps(
        start_rotmat=start_rotmat,
        start_pos=start_pos,
        goal_rotmat=goal_rotmat,
        goal_pos=goal_pos,
        n_steps=n_steps,
    )


def line_by_n_steps(
    start_rotmat, start_pos, goal_rotmat=None, goal_pos=None, n_steps=2
):
    """
    Cartesian straight-line interpolation with fixed number of samples.

    Parameters
    ----------
    start_rotmat : array-like, shape (3, 3)
    start_pos : array-like, shape (3,)
    goal_rotmat : array-like, shape (3, 3), optional
    goal_pos : array-like, shape (3,), optional
    n_steps : int
        Number of interpolated samples (including both ends).

    Returns
    -------
    pos_seq : np.ndarray, shape (N, 3), dtype float32
    rotmat_seq : np.ndarray, shape (N, 3, 3), dtype float32
    """
    n_steps = int(n_steps)
    start_rotmat = np.asarray(start_rotmat, dtype=np.float32)
    start_pos = np.asarray(start_pos, dtype=np.float32)
    goal_rotmat, goal_pos, is_single = _resolve_goal(
        start_rotmat=start_rotmat,
        start_pos=start_pos,
        goal_rotmat=goal_rotmat,
        goal_pos=goal_pos,
    )
    if is_single:
        return (
            start_pos.reshape(1, 3).astype(np.float32),
            start_rotmat.reshape(1, 3, 3).astype(np.float32),
        )
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    pos_seq = np.linspace(start_pos, goal_pos, n_steps, dtype=np.float32)
    rotmat_seq = oum.rotmat_slerp(start_rotmat, goal_rotmat, n_steps)
    return pos_seq, rotmat_seq


def _resolve_goal(start_rotmat, start_pos, goal_rotmat, goal_pos):
    start_rotmat = np.asarray(start_rotmat, dtype=np.float32)
    start_pos = np.asarray(start_pos, dtype=np.float32)
    if goal_rotmat is None and goal_pos is None:
        warnings.warn(
            "Both goal_rotmat and goal_pos are None; return start pose only.",
            RuntimeWarning,
        )
        return start_rotmat, start_pos, True
    if goal_rotmat is None:
        goal_rotmat = start_rotmat.copy()
    else:
        goal_rotmat = np.asarray(goal_rotmat, dtype=np.float32)
    if goal_pos is None:
        goal_pos = start_pos.copy()
    else:
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
    return goal_rotmat, goal_pos, False
