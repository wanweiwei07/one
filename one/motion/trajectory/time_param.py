import numpy as np


def retime_trapezoidal(q_seq, v_max, a_max, dt=0.01):
    """
    Time-parameterize a joint waypoint path using per-segment trapezoidal scaling.

    Parameters
    ----------
    q_seq : array-like, shape (N, D)
        Joint waypoints.
    v_max : array-like, shape (D,)
        Joint velocity limits (absolute).
    a_max : array-like, shape (D,)
        Joint acceleration limits (absolute).
    dt : float
        Sampling period in seconds.

    Returns
    -------
    t_seq : np.ndarray, shape (M,), dtype float32
    q_out : np.ndarray, shape (M, D), dtype float32
    qd_out : np.ndarray, shape (M, D), dtype float32
    qdd_out : np.ndarray, shape (M, D), dtype float32
    """
    q_seq = np.asarray(q_seq, dtype=np.float32)
    v_max = np.asarray(v_max, dtype=np.float32)
    a_max = np.asarray(a_max, dtype=np.float32)
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
    if q_seq.shape[0] == 1:
        return (
            np.array([0.0], dtype=np.float32),
            q_seq.copy(),
            np.zeros_like(q_seq, dtype=np.float32),
            np.zeros_like(q_seq, dtype=np.float32),
        )

    t_all = []
    q_all = []
    qd_all = []
    qdd_all = []
    t_offset = 0.0
    eps = 1e-9

    for i in range(q_seq.shape[0] - 1):
        q0 = q_seq[i]
        q1 = q_seq[i + 1]
        dq = q1 - q0
        mask = np.abs(dq) > eps
        if not np.any(mask):
            continue
        sdot_max = float(np.min(v_max[mask] / np.abs(dq[mask])))
        sddot_max = float(np.min(a_max[mask] / np.abs(dq[mask])))
        t_acc, t_cruise, t_total = _unit_profile_timing(sdot_max, sddot_max)
        n_steps = max(int(np.ceil(t_total / dt)), 1) + 1
        t_local = np.linspace(0.0, t_total, n_steps, dtype=np.float32)
        s, sdot, sddot = _unit_profile_eval(t_local, t_acc, t_cruise, sdot_max, sddot_max)
        q_local = q0[None, :] + s[:, None] * dq[None, :]
        qd_local = sdot[:, None] * dq[None, :]
        qdd_local = sddot[:, None] * dq[None, :]
        t_local = t_local + np.float32(t_offset)

        # stitch segments without duplicate endpoint
        if len(t_all) > 0:
            t_local = t_local[1:]
            q_local = q_local[1:]
            qd_local = qd_local[1:]
            qdd_local = qdd_local[1:]
        t_all.append(t_local)
        q_all.append(q_local.astype(np.float32))
        qd_all.append(qd_local.astype(np.float32))
        qdd_all.append(qdd_local.astype(np.float32))
        t_offset += float(t_total)

    if len(t_all) == 0:
        q0 = q_seq[0:1].astype(np.float32)
        z = np.zeros_like(q0, dtype=np.float32)
        return np.array([0.0], dtype=np.float32), q0, z, z

    t_seq = np.concatenate(t_all, axis=0).astype(np.float32)
    q_out = np.concatenate(q_all, axis=0).astype(np.float32)
    qd_out = np.concatenate(qd_all, axis=0).astype(np.float32)
    qdd_out = np.concatenate(qdd_all, axis=0).astype(np.float32)
    return t_seq, q_out, qd_out, qdd_out


def _unit_profile_timing(sdot_max, sddot_max):
    # rest-to-rest profile on unit distance in scalar s
    if sdot_max * sdot_max / sddot_max >= 1.0:
        # triangular
        t_acc = np.sqrt(1.0 / sddot_max)
        t_cruise = 0.0
        t_total = 2.0 * t_acc
    else:
        # trapezoidal
        t_acc = sdot_max / sddot_max
        d_acc = 0.5 * sddot_max * t_acc * t_acc
        d_cruise = 1.0 - 2.0 * d_acc
        t_cruise = d_cruise / sdot_max
        t_total = 2.0 * t_acc + t_cruise
    return float(t_acc), float(t_cruise), float(t_total)


def _unit_profile_eval(t, t_acc, t_cruise, sdot_max, sddot_max):
    t = np.asarray(t, dtype=np.float32)
    t1 = t_acc
    t2 = t_acc + t_cruise
    t3 = 2.0 * t_acc + t_cruise
    s = np.zeros_like(t, dtype=np.float32)
    sdot = np.zeros_like(t, dtype=np.float32)
    sddot = np.zeros_like(t, dtype=np.float32)

    m1 = t <= t1
    m2 = (t > t1) & (t <= t2)
    m3 = t > t2

    # accel
    s[m1] = 0.5 * sddot_max * t[m1] * t[m1]
    sdot[m1] = sddot_max * t[m1]
    sddot[m1] = sddot_max

    # cruise
    s_at_t1 = 0.5 * sddot_max * t1 * t1
    s[m2] = s_at_t1 + sdot_max * (t[m2] - t1)
    sdot[m2] = sdot_max
    sddot[m2] = 0.0

    # decel
    tau = t3 - t[m3]
    s[m3] = 1.0 - 0.5 * sddot_max * tau * tau
    sdot[m3] = sddot_max * tau
    sddot[m3] = -sddot_max

    # guard final value
    if s.shape[0] > 0:
        s[-1] = 1.0
        sdot[-1] = 0.0
        sddot[-1] = 0.0
    return s.astype(np.float32), sdot.astype(np.float32), sddot.astype(np.float32)
