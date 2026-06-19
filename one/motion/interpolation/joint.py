import numpy as np
import one.utils.math as oum


def interp_by_step(start_qs, goal_qs, step=np.deg2rad(3.0)):
    """
    Joint-space straight-line interpolation with automatic sample count.

    Sample count follows the largest per-joint motion: N = max(ceil(max|dq| /
    step), 2). Straight-line interpolation between two configs is a property of
    the (real-vector) configuration space, so the math is delegated to the
    single in-house source ``oum.interpolate_vectors``; this is just the
    joint-space, float32 entry point that mirrors ``cartesian.interp_by_step``.

    Parameters
    ----------
    start_qs : array-like, shape (n_jnts,)
    goal_qs : array-like, shape (n_jnts,)
    step : float
        Maximum per-joint change per segment in radians.

    Returns
    -------
    q_seq : np.ndarray, shape (N, n_jnts), dtype float32
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    return oum.interpolate_vectors(start_qs, goal_qs, step).astype(np.float32)


def interp_by_n(start_qs, goal_qs, n_steps=2):
    """
    Joint-space straight-line interpolation with fixed number of samples.

    Parameters
    ----------
    start_qs : array-like, shape (n_jnts,)
    goal_qs : array-like, shape (n_jnts,)
    n_steps : int
        Number of interpolated samples (including both ends).

    Returns
    -------
    q_seq : np.ndarray, shape (N, n_jnts), dtype float32
    """
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    start_qs = np.asarray(start_qs, dtype=np.float32)
    goal_qs = np.asarray(goal_qs, dtype=np.float32)
    return np.linspace(start_qs, goal_qs, n_steps, dtype=np.float32)


def linear_path(start_qs, goal_qs, step=np.deg2rad(3.0), ctx=None):
    """
    Joint-space straight-line path between two configs, optionally collision-gated.

    The joint-space counterpart of ``cartesian.linear_to_jpath``: a straight line
    in joint space needs no IK, so this is just ``interp_by_step`` plus an optional
    validity gate. With ``ctx`` given, the whole densified segment (endpoints +
    bounds + collision at ``ctx.cd_step_size``) is checked via
    ``ctx.is_motion_valid``; if it is blocked, returns None instead of an unsafe
    path. ``ctx=None`` degrades to the raw interpolation. Returning a path here
    (not a timed trajectory) matches the probabilistic planners' vocabulary;
    timing is a separate ``trajectory.time_param`` step.

    Parameters
    ----------
    start_qs : array-like, shape (n_jnts,)
    goal_qs : array-like, shape (n_jnts,)
    step : float
        Maximum per-joint change per segment in radians.
    ctx : PlanningContext, optional
        When given, the segment is rejected (None) if not motion-valid.

    Returns
    -------
    jpath : np.ndarray | None
        Joint path with shape (N, n_jnts), or None when ``ctx`` rejects it.
    """
    if ctx is not None and not ctx.is_motion_valid(start_qs, goal_qs):
        return None
    return interp_by_step(start_qs, goal_qs, step=step)
