"""Time-optimal retiming of a joint waypoint path (Kunz-Stilman style).

The path is treated as ONE curve in joint space, parameterized by arc length
``s``. Timing is then a single scalar function ``s(t)``, so every joint shares
one path speed ``sdot`` and gets its share through the segment's unit tangent:
``qd = sdot * u``. Two sweeps over the waypoints -- forward (accelerate as hard
as allowed) and backward (stay slow enough to still stop) -- give the fastest
feasible ``sdot`` at every waypoint; the pointwise minimum of the two is the
time-optimal profile.

This is what a per-segment trapezoidal retimer cannot do: solving each segment
as an independent rest-to-rest move makes a waypoint-dense path pay a full stop
at every point. Here a waypoint only costs speed when the direction
actually turns -- collinear waypoints are passed through untouched, and the
result depends on the path's GEOMETRY, not on how finely it was sampled.

Corners are handled without deviating from the path: crossing a waypoint at
path speed ``sdot`` makes joint j's velocity jump by ``sdot * du_j``, and that
jump has to fit in one control period, giving ``sdot <= a_j * dt / |du_j|``.
Collinear waypoints have ``du = 0`` and are therefore unconstrained.
"""

import numpy as np

# a segment shorter than this is a duplicate waypoint, not a motion
_EPS_LEN = 1e-9


def retime(q_seq, v_max, a_max, dt=0.01, corner_dt=None):
    """
    Time-parameterize a joint waypoint path with forward/backward sweeps.

    Parameters
    ----------
    q_seq : array-like, shape (N, D)
        Joint waypoints. Duplicate (zero-length) waypoints are ignored.
    v_max : array-like, shape (D,)
        Joint velocity limits (absolute).
    a_max : array-like, shape (D,)
        Joint acceleration limits (absolute).
    dt : float
        Output sampling period in seconds. ``t_seq`` is a uniform grid of
        exactly this step.
    corner_dt : float, optional
        Control period the corner bound is allowed to spread a direction
        change over. Defaults to ``dt``; pass the robot's servo period when
        sampling finer than the robot is driven.

    Returns
    -------
    t_seq : np.ndarray, shape (M,), dtype float64
    q_out : np.ndarray, shape (M, D), dtype float32
    qd_out : np.ndarray, shape (M, D), dtype float32
    qdd_out : np.ndarray, shape (M, D), dtype float32
    """
    q_seq = np.asarray(q_seq, dtype=np.float64)
    v_max = np.asarray(v_max, dtype=np.float64)
    a_max = np.asarray(a_max, dtype=np.float64)
    if q_seq.ndim != 2:
        raise ValueError(f'q_seq must be 2D, got shape {q_seq.shape}')
    if q_seq.shape[0] < 1:
        raise ValueError('q_seq must contain at least one waypoint')
    n_jnts = q_seq.shape[1]
    if v_max.shape != (n_jnts,):
        raise ValueError(f'v_max shape must be ({n_jnts},), got {v_max.shape}')
    if a_max.shape != (n_jnts,):
        raise ValueError(f'a_max shape must be ({n_jnts},), got {a_max.shape}')
    if np.any(v_max <= 0):
        raise ValueError('all v_max must be > 0')
    if np.any(a_max <= 0):
        raise ValueError('all a_max must be > 0')
    if dt <= 0:
        raise ValueError(f'dt must be > 0, got {dt}')
    corner_dt = float(dt) if corner_dt is None else float(corner_dt)
    if corner_dt <= 0:
        raise ValueError(f'corner_dt must be > 0, got {corner_dt}')

    starts, seg_len, u = _geometry(q_seq)
    if seg_len.size == 0:                       # no motion at all
        return _at_rest(q_seq[0])
    seg_ceil, seg_acc, node_ceil = _limits(seg_len, u, v_max, a_max, corner_dt)
    w = _sweep(seg_len, seg_acc, node_ceil)
    profiles = [_seg_profile(w[i], w[i + 1], seg_len[i], seg_acc[i], seg_ceil[i])
                for i in range(seg_len.size)]
    return _sample(starts, u, profiles, dt)


def _geometry(q_seq):
    """Split the polyline into segment start points, arc lengths and unit
    tangents, dropping duplicate waypoints."""
    dq = np.diff(q_seq, axis=0)
    seg_len = np.linalg.norm(dq, axis=1)
    keep = seg_len > _EPS_LEN
    dq, seg_len = dq[keep], seg_len[keep]
    if seg_len.size == 0:
        return q_seq[:1], seg_len, np.zeros((0, q_seq.shape[1]))
    return q_seq[:-1][keep], seg_len, dq / seg_len[:, None]


def _min_ratio(limits, u):
    """min_j limits_j / |u_ij| per segment; a joint that does not move on a
    segment (u_ij == 0) constrains nothing."""
    au = np.abs(u)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(au > 0.0, limits / np.where(au > 0.0, au, 1.0), np.inf)
    return np.min(ratio, axis=1)


def _limits(seg_len, u, v_max, a_max, corner_dt):
    """Speed ceiling and acceleration bound per segment, plus the speed
    ceiling at every waypoint (both adjacent segments, and the corner)."""
    seg_ceil = _min_ratio(v_max, u)
    seg_acc = _min_ratio(a_max, u)
    n_segs = seg_len.size
    node_ceil = np.full(n_segs + 1, np.inf)
    node_ceil[:-1] = np.minimum(node_ceil[:-1], seg_ceil)   # segment starts here
    node_ceil[1:] = np.minimum(node_ceil[1:], seg_ceil)     # segment ends here
    if n_segs > 1:
        # a turn of du costs joint j a velocity jump of sdot*du_j, which has to
        # fit in one control period: sdot <= a_j*corner_dt/|du_j|
        node_ceil[1:-1] = np.minimum(
            node_ceil[1:-1], _min_ratio(a_max * corner_dt, np.diff(u, axis=0)))
    node_ceil[0] = 0.0                                      # start at rest
    node_ceil[-1] = 0.0                                     # end at rest
    return seg_ceil, seg_acc, node_ceil


def _sweep(seg_len, seg_acc, node_ceil):
    """Path speed at every waypoint: forward (how fast can I get here) and
    backward (how fast may I be and still stop), then the pointwise min."""
    w = np.zeros(seg_len.size + 1)
    for i in range(1, w.size):
        w[i] = min(node_ceil[i],
                   np.sqrt(w[i - 1] ** 2 + 2.0 * seg_acc[i - 1] * seg_len[i - 1]))
    w[-1] = node_ceil[-1]
    for i in range(w.size - 2, -1, -1):
        w[i] = min(w[i], np.sqrt(w[i + 1] ** 2 + 2.0 * seg_acc[i] * seg_len[i]))
    return w


def _seg_profile(w0, w1, length, acc, ceil):
    """Fastest accel/cruise/decel profile entering at w0 and leaving at w1.

    Returns (t_acc, t_cruise, t_total, w0, w_peak, w1, acc, length). Both
    sweeps ran first, so w0 and w1 are always mutually reachable and the peak
    is at or above them.
    """
    peak = np.sqrt((w0 * w0 + w1 * w1) / 2.0 + acc * length)
    if peak <= ceil:                                    # never reaches the ceiling
        t_acc = (peak - w0) / acc
        return (t_acc, 0.0, t_acc + (peak - w1) / acc, w0, peak, w1, acc, length)
    t_acc = (ceil - w0) / acc
    t_dec = (ceil - w1) / acc
    d_ramp = ((ceil * ceil - w0 * w0) + (ceil * ceil - w1 * w1)) / (2.0 * acc)
    t_cruise = (length - d_ramp) / ceil
    return (t_acc, t_cruise, t_acc + t_cruise + t_dec,
            w0, ceil, w1, acc, length)


def _eval_profile(prof, tau):
    """Arc length, path speed and path acceleration at local times tau."""
    t_acc, t_cruise, t_total, w0, peak, w1, acc, length = prof
    t2 = t_acc + t_cruise
    dist = np.empty_like(tau)
    spd = np.empty_like(tau)
    accel = np.empty_like(tau)
    m1 = tau <= t_acc
    m2 = (tau > t_acc) & (tau <= t2)
    m3 = tau > t2
    ta = tau[m1]
    dist[m1] = w0 * ta + 0.5 * acc * ta * ta
    spd[m1] = w0 + acc * ta
    accel[m1] = acc
    d_acc = (peak * peak - w0 * w0) / (2.0 * acc)
    dist[m2] = d_acc + peak * (tau[m2] - t_acc)
    spd[m2] = peak
    accel[m2] = 0.0
    sigma = t_total - tau[m3]                           # time left to the exit
    dist[m3] = length - (w1 * sigma + 0.5 * acc * sigma * sigma)
    spd[m3] = w1 + acc * sigma
    accel[m3] = -acc
    return dist, spd, accel


def _sample(starts, u, profiles, dt):
    """Evaluate the profiles on a global grid of exactly dt."""
    seg_time = np.array([p[2] for p in profiles])
    t_end = np.cumsum(seg_time)
    t_start = np.concatenate([[0.0], t_end[:-1]])
    total = float(t_end[-1])
    n_samples = int(np.ceil(total / dt)) + 1
    t_seq = np.arange(n_samples, dtype=np.float64) * dt
    # the grid may overshoot the last waypoint by up to dt; hold there
    t_clip = np.minimum(t_seq, total)
    idx = np.clip(np.searchsorted(t_end, t_clip, side='left'), 0, len(profiles) - 1)
    q_out = np.empty((n_samples, starts.shape[1]))
    qd_out = np.empty_like(q_out)
    qdd_out = np.empty_like(q_out)
    for i, prof in enumerate(profiles):
        sel = idx == i
        if not np.any(sel):
            continue
        tau = np.clip(t_clip[sel] - t_start[i], 0.0, prof[2])
        dist, spd, accel = _eval_profile(prof, tau)
        q_out[sel] = starts[i] + dist[:, None] * u[i]
        qd_out[sel] = spd[:, None] * u[i]
        qdd_out[sel] = accel[:, None] * u[i]
    # land exactly on the goal, at rest
    q_out[-1] = starts[-1] + profiles[-1][7] * u[-1]
    qd_out[-1] = 0.0
    qdd_out[-1] = 0.0
    return (t_seq,
            q_out.astype(np.float32),
            qd_out.astype(np.float32),
            qdd_out.astype(np.float32))


def _at_rest(q):
    q = np.asarray(q, dtype=np.float32)[None, :]
    z = np.zeros_like(q)
    return np.zeros(1, dtype=np.float64), q.copy(), z, z.copy()
