"""Checks for one.motion.trajectory.totg -- the properties that make the timing
depend on the path's geometry rather than on its sampling, plus the degenerate
paths."""

import numpy as np

import one.motion.trajectory.totg as omttg

DT = 0.008
V = np.full(6, np.pi / 2)
A = np.full(6, np.pi)

_failures = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def densify(path, k):
    out = [p0 + (p1 - p0) * i / k for p0, p1 in zip(path[:-1], path[1:])
           for i in range(k)]
    out.append(path[-1])
    return np.asarray(out)


rng = np.random.default_rng(0)
corners = rng.uniform(-1, 1, size=(6, 6))
line = np.stack([np.zeros(6) + np.ones(6) * s for s in np.linspace(0, 1, 5)])

print('1. density invariance -- geometry decides the timing, not the sampling')
t_sparse = omttg.retime(corners, V, A, dt=DT)[0][-1]
for k in (2, 4, 12, 30):
    t_dense = omttg.retime(densify(corners, k), V, A, dt=DT)[0][-1]
    check(f'{len(densify(corners, k)):3d} waypoints == 6 waypoints',
          abs(t_dense - t_sparse) <= DT,
          f'{t_dense:.4f}s vs {t_sparse:.4f}s')
t_one = omttg.retime(np.stack([np.zeros(6), np.ones(6)]), V, A, dt=DT)[0][-1]
t_cut = omttg.retime(line, V, A, dt=DT)[0][-1]
check('collinear cut into 4 == single segment', abs(t_cut - t_one) <= DT,
      f'{t_cut:.4f}s vs {t_one:.4f}s')

print('2. the time grid is exactly dt')
for tag, path in (('corners', corners), ('dense', densify(corners, 12)),
                  ('line', line)):
    t = omttg.retime(path, V, A, dt=DT)[0]
    d = np.diff(t)
    check(f'{tag}: diff(t) == dt', bool(np.allclose(d, DT, atol=1e-12)),
          f'min={d.min():.12f} max={d.max():.12f}')

print('3. limits are respected')
for tag, path in (('corners', corners), ('dense', densify(corners, 12))):
    _, _, qd, qdd = omttg.retime(path, V, A, dt=DT)
    check(f'{tag}: |qd| <= v_max', bool(np.all(np.abs(qd) <= V + 1e-4)),
          f'max={np.abs(qd).max():.6f}')
    check(f'{tag}: |qdd| <= a_max', bool(np.all(np.abs(qdd) <= A + 1e-4)),
          f'max={np.abs(qdd).max():.6f}')

print('4. qd really is the derivative of q')
# The midpoint rule is exact only where qd is continuous. An interval that
# straddles a corner sees qd turn, so it may differ -- but never by more than
# one control period of full acceleration, which is what the corner bound buys.
for tag, path in (('corners', corners), ('dense', densify(corners, 12)),
                  ('line', line)):
    t, q, qd, _ = omttg.retime(path, V, A, dt=DT)
    fd = np.diff(q.astype(np.float64), axis=0) / np.diff(t)[:, None]
    err = np.abs(fd - 0.5 * (qd[1:] + qd[:-1])).max(axis=1)
    n_corners = max(len(np.unique(np.round(path, 9), axis=0)) - 2, 0)
    check(f'{tag}: away from corners', int(np.sum(err >= 5e-3)) <= n_corners,
          f'{int(np.sum(err >= 5e-3))} of {len(err)} intervals, '
          f'<= {n_corners} corners')
    check(f'{tag}: at corners, within a_max*dt',
          bool(np.all(err <= A.max() * DT + 1e-4)),
          f'max={err.max():.2e} <= {A.max() * DT:.2e}')

print('5. the path is actually followed (endpoints and no shortcutting)')
t, q, qd, _ = omttg.retime(corners, V, A, dt=DT)
check('starts at the first waypoint', bool(np.allclose(q[0], corners[0], atol=1e-5)))
check('ends at the last waypoint', bool(np.allclose(q[-1], corners[-1], atol=1e-5)))
check('starts and ends at rest',
      bool(np.allclose(qd[0], 0, atol=1e-5) and np.allclose(qd[-1], 0, atol=1e-5)))
hit = [np.abs(q - wp).sum(axis=1).min() for wp in corners]
check('every waypoint is passed through', max(hit) < 5e-2, f'max miss={max(hit):.2e}')

print('6. degenerate paths (these break the WRS totg)')
cases = {
    'single waypoint': np.zeros((1, 6)),
    'two waypoints': np.stack([np.zeros(6), np.ones(6)]),
    'duplicate waypoints': np.stack([np.zeros(6), np.ones(6), np.ones(6),
                                     np.ones(6) * 2]),
    'all waypoints identical': np.zeros((4, 6)),
    'exact 180 deg reversal': np.stack([np.zeros(6), np.ones(6), np.zeros(6)]),
    'one joint holds then reverses': np.array([
        [0, 0, 0, 0, 0, 0], [.5, .1, 0, 0, 0, 0], [.5, .2, 0, 0, 0, 0],
        [.5, .3, 0, 0, 0, 0], [.2, .4, 0, 0, 0, 0], [-.3, .5, 0, 0, 0, 0]],
        dtype=float),
}
for tag, path in cases.items():
    try:
        t, q, qd, qdd = omttg.retime(path, V, A, dt=DT)
        finite = bool(np.all(np.isfinite(t)) and np.all(np.isfinite(q))
                      and np.all(np.isfinite(qd)) and np.all(np.isfinite(qdd)))
        moves = path.shape[0] > 1 and np.linalg.norm(path[-1] - path[0]) > 1e-9
        check(tag, finite and np.all(np.abs(qd) <= V + 1e-4)
              and (t[-1] > 0 if moves else True),
              f'{t[-1]:.4f}s, {len(t)} samples')
    except Exception as exc:                                  # noqa: BLE001
        check(tag, False, f'{type(exc).__name__}: {exc}')

print('7. duplicate waypoints change nothing')
plain = np.stack([np.zeros(6), np.ones(6)])
dup = np.stack([np.zeros(6), np.zeros(6), np.ones(6), np.ones(6)])
check('with vs without duplicates',
      abs(omttg.retime(plain, V, A, dt=DT)[0][-1]
          - omttg.retime(dup, V, A, dt=DT)[0][-1]) < 1e-9)

print()
if _failures:
    print(f'{len(_failures)} FAILED: {_failures}')
    raise SystemExit(1)
print('all checks passed')
