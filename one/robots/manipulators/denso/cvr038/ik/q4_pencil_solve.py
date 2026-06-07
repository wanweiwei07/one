import numpy as np
from scipy.linalg import eig

"""
CVR038 q4 pencil solver (runtime).

Given a target the q4 candidates are the real roots of a resultant pencil in
t = tan(q4 / 2).  The deflated res_c3 / res_len_unit coefficients come from the
generated closed forms in q4_pencil_coeffs.py (regenerated offline by
gen_q4_pencil_coeffs.py), so there is no per-call symbolic work and no sympy
dependency.  Eliminating s3 is done as a generalized-eigenvalue problem of the
Sylvester pencil rather than by expanding the badly conditioned high-degree
univariate polynomial.

Polynomials are dicts mapping exponent tuples to float coefficients.
"""


def _trim(poly, tol=1e-12):
    return {k: float(v) for k, v in poly.items() if abs(v) > tol}


def _degree(poly, var):
    if not poly:
        return -1
    return max(k[var] for k in poly)


def _coeffs_in_var(poly, var):
    deg = _degree(poly, var)
    out = []
    for d in range(deg, -1, -1):
        coeff = {}
        for k, v in poly.items():
            if k[var] != d:
                continue
            kk = list(k)
            kk[var] = 0
            coeff[tuple(kk)] = coeff.get(tuple(kk), 0.0) + v
        out.append(_trim(coeff))
    return out


def _sylvester(f_coeffs, g_coeffs):
    m = len(f_coeffs) - 1
    n = len(g_coeffs) - 1
    zero = {}
    rows = []
    for i in range(n):
        rows.append([zero] * i + f_coeffs + [zero] * (n - i - 1))
    for i in range(m):
        rows.append([zero] * i + g_coeffs + [zero] * (m - i - 1))
    return rows


def _sylvester_for_var(poly_a, poly_b, var):
    return _sylvester(_coeffs_in_var(poly_a, var), _coeffs_in_var(poly_b, var))


def _poly_to_t_coeffs(poly):
    if not poly:
        return np.zeros(1, dtype=np.float64)
    deg = max(k[0] for k in poly)
    coeffs = np.zeros(deg + 1, dtype=np.float64)
    for k, v in poly.items():
        if any(x != 0 for x in k[1:]):
            raise ValueError('Expected t-only polynomial entry.')
        coeffs[k[0]] += v
    return coeffs


def _poly_matrix_to_coeff_mats(mat):
    n = len(mat)
    max_deg = 0
    entry_coeffs = []
    for row in mat:
        coeff_row = []
        for entry in row:
            coeff = _poly_to_t_coeffs(entry)
            max_deg = max(max_deg, coeff.size - 1)
            coeff_row.append(coeff)
        entry_coeffs.append(coeff_row)
    mats = [np.zeros((n, n), dtype=np.float64) for _ in range(max_deg + 1)]
    for i in range(n):
        for j in range(n):
            coeff = entry_coeffs[i][j]
            for k, v in enumerate(coeff):
                mats[k][i, j] = v
    return mats


def _polymat_roots(mats, tol=1e-8):
    while len(mats) > 1 and np.linalg.norm(mats[-1]) < 1e-14:
        mats.pop()
    degree = len(mats) - 1
    if degree <= 0:
        return []
    n = mats[0].shape[0]
    size = degree * n
    cmat = np.zeros((size, size), dtype=np.float64)
    dmat = np.zeros((size, size), dtype=np.float64)
    for i in range(degree - 1):
        row = slice(i * n, (i + 1) * n)
        nxt = slice((i + 1) * n, (i + 2) * n)
        cur = slice(i * n, (i + 1) * n)
        cmat[row, nxt] = np.eye(n)
        dmat[row, cur] = np.eye(n)
    row = slice((degree - 1) * n, degree * n)
    cmat[row, :] = -np.hstack(mats[:-1])
    dmat[row, row] = mats[-1]
    vals = eig(
        cmat,
        dmat,
        right=False,
        overwrite_a=True,
        overwrite_b=True,
        check_finite=False,
    )
    roots = []
    for val in vals:
        if not np.isfinite(val):
            continue
        if abs(val.imag) > tol:
            continue
        roots.append(float(val.real))
    return roots


def _coeff_dict(keys, vals):
    scale = max((abs(v) for v in vals), default=1.0) or 1.0
    return {k: v / scale for k, v in zip(keys, vals)}


def q4_roots_pencil_fast(px, py, pz, vx, vy, vz, tol=1e-8):
    """Closed-form-coefficient pencil for the fixed CVR038 geometry.

    q4 is invariant under base-Z rotation of the target, so the target is first
    rotated to py = 0.  The deflated res_c3 / res_len_unit coefficients then come
    from the generated closed forms (no sympy, no dict resultant build), and the
    s3 pencil is ~deg 12 instead of ~deg 32 -- both fast and well conditioned.
    """
    from one.robots.manipulators.denso.cvr038.ik import q4_pencil_coeffs as coeffs

    a = np.arctan2(py, px)
    ca, sa = np.cos(a), np.sin(a)
    px_c = px * ca + py * sa            # = hypot(px, py), py rotates to 0
    vx_c = vx * ca + vy * sa
    vy_c = -vx * sa + vy * ca
    c3_vals, lu_vals = coeffs.eval_coeffs(px_c, pz, vx_c, vy_c, vz)
    rc3 = _coeff_dict(coeffs.C3_KEYS, c3_vals)
    rlu = _coeff_dict(coeffs.LU_KEYS, lu_vals)
    syl = _sylvester_for_var(rc3, rlu, 1)
    mats = _poly_matrix_to_coeff_mats(syl)
    roots = []
    for t in _polymat_roots(mats, tol=tol):
        q4 = 2.0 * np.arctan(t)
        q4 = (q4 + np.pi) % (2.0 * np.pi) - np.pi
        if all(abs(((q4 - old + np.pi) % (2.0 * np.pi)) - np.pi) > 1e-7
               for old in roots):
            roots.append(float(q4))
    roots.sort()
    return roots


if __name__ == '__main__':
    target = (0.13353576, -0.01027953, 0.30813415,
              0.99296349, -0.11385252, 0.03257641)
    print('pencil_fast:', q4_roots_pencil_fast(*target))
