import itertools
import numpy as np
from scipy.linalg import eig

"""
Experimental numeric resultant prototype.

This module backs CVR038PencilIK.  The pencil path avoids expanding the final
determinant into a badly conditioned high-degree polynomial.
"""


def _trim(poly, tol=1e-12):
    out = {k: float(v) for k, v in poly.items() if abs(v) > tol}
    return out


def _const(c, nvars):
    return {} if abs(c) == 0.0 else {(0,) * nvars: float(c)}


def _var(i, nvars):
    key = [0] * nvars
    key[i] = 1
    return {tuple(key): 1.0}


def _add(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return _trim(out)


def _neg(a):
    return {k: -v for k, v in a.items()}


def _sub(a, b):
    return _add(a, _neg(b))


def _mul(a, b):
    if not a or not b:
        return {}
    out = {}
    nvars = len(next(iter(a)))
    if nvars == 4:
        for ka, va in a.items():
            a0, a1, a2, a3 = ka
            for kb, vb in b.items():
                k = (a0 + kb[0], a1 + kb[1], a2 + kb[2], a3 + kb[3])
                out[k] = out.get(k, 0.0) + va * vb
    elif nvars == 3:
        for ka, va in a.items():
            a0, a1, a2 = ka
            for kb, vb in b.items():
                k = (a0 + kb[0], a1 + kb[1], a2 + kb[2])
                out[k] = out.get(k, 0.0) + va * vb
    elif nvars == 2:
        for ka, va in a.items():
            a0, a1 = ka
            for kb, vb in b.items():
                k = (a0 + kb[0], a1 + kb[1])
                out[k] = out.get(k, 0.0) + va * vb
    elif nvars == 1:
        for ka, va in a.items():
            a0 = ka[0]
            for kb, vb in b.items():
                k = (a0 + kb[0],)
                out[k] = out.get(k, 0.0) + va * vb
    else:
        for ka, va in a.items():
            for kb, vb in b.items():
                k = tuple(x + y for x, y in zip(ka, kb))
                out[k] = out.get(k, 0.0) + va * vb
    return _trim(out)


def _pow(a, n):
    if n == 0:
        nvars = len(next(iter(a))) if a else 1
        return _const(1.0, nvars)
    out = a
    for _ in range(n - 1):
        out = _mul(out, a)
    return out


def _scale(a, s):
    if abs(s) == 0.0:
        return {}
    return _trim({k: v * s for k, v in a.items()})


def _degree(poly, var):
    if not poly:
        return -1
    return max(k[var] for k in poly)


def _coeffs_in_var(poly, var):
    deg = _degree(poly, var)
    nvars = len(next(iter(poly))) if poly else 1
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


def _drop_var(poly, var):
    out = {}
    for k, v in poly.items():
        if k[var] != 0:
            raise ValueError('Cannot drop variable with non-zero degree.')
        kk = tuple(x for i, x in enumerate(k) if i != var)
        out[kk] = out.get(kk, 0.0) + v
    return _trim(out)


def _raise_to_nvars(poly, nvars):
    if not poly:
        return {}
    old_nvars = len(next(iter(poly)))
    if old_nvars == nvars:
        return poly
    out = {}
    for k, v in poly.items():
        out[k + (0,) * (nvars - old_nvars)] = v
    return out


def _sylvester(f_coeffs, g_coeffs):
    m = len(f_coeffs) - 1
    n = len(g_coeffs) - 1
    size = m + n
    zero = {}
    rows = []
    for i in range(n):
        rows.append([zero] * i + f_coeffs + [zero] * (n - i - 1))
    for i in range(m):
        rows.append([zero] * i + g_coeffs + [zero] * (m - i - 1))
    return rows


def _sylvester_for_var(poly_a, poly_b, var):
    return _sylvester(_coeffs_in_var(poly_a, var), _coeffs_in_var(poly_b, var))


def _det_poly(mat):
    n = len(mat)
    if n == 0:
        return {}
    if n == 1:
        return mat[0][0]
    if n == 2:
        return _sub(_mul(mat[0][0], mat[1][1]),
                    _mul(mat[0][1], mat[1][0]))
    if n == 3:
        a = _add(_add(_mul(_mul(mat[0][0], mat[1][1]), mat[2][2]),
                      _mul(_mul(mat[0][1], mat[1][2]), mat[2][0])),
                 _mul(_mul(mat[0][2], mat[1][0]), mat[2][1]))
        b = _add(_add(_mul(_mul(mat[0][2], mat[1][1]), mat[2][0]),
                      _mul(_mul(mat[0][0], mat[1][2]), mat[2][1])),
                 _mul(_mul(mat[0][1], mat[1][0]), mat[2][2]))
        return _sub(a, b)
    if n == 4:
        out = {}
        for j in range(4):
            minor = [[mat[r][c] for c in range(4) if c != j]
                     for r in range(1, 4)]
            term = _mul(mat[0][j], _det_poly(minor))
            out = _add(out, _neg(term) if j % 2 else term)
        return out
    out = {}
    for perm in itertools.permutations(range(n)):
        term = None
        inv = 0
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inv += 1
            elem = mat[i][perm[i]]
            if term is None:
                term = elem
            else:
                term = _mul(term, elem)
            if not term:
                break
        if term:
            out = _add(out, _neg(term) if inv % 2 else term)
    return _trim(out)


def resultant(poly_a, poly_b, var):
    coeffs_a = _coeffs_in_var(poly_a, var)
    coeffs_b = _coeffs_in_var(poly_b, var)
    mat = _sylvester(coeffs_a, coeffs_b)
    res = _det_poly(mat)
    return _drop_var(res, var)


def _poly_to_univar_coeffs(poly):
    if not poly:
        return np.array([], dtype=np.float64)
    min_deg = min(k[0] for k in poly)
    deg = max(k[0] for k in poly) - min_deg
    coeffs = np.zeros(deg + 1, dtype=np.float64)
    for k, v in poly.items():
        if len(k) != 1:
            raise ValueError('Expected univariate polynomial.')
        coeffs[deg - (k[0] - min_deg)] += v
    return coeffs


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


def _roots_from_poly(poly, tol=1e-8):
    coeffs = _poly_to_univar_coeffs(poly)
    if coeffs.size == 0:
        return []
    scale = np.max(np.abs(coeffs))
    if scale == 0.0:
        return []
    coeffs = coeffs / scale
    roots = []
    for root in np.roots(coeffs):
        if abs(root.imag) > tol:
            continue
        q4 = 2.0 * np.arctan(root.real)
        q4 = (q4 + np.pi) % (2.0 * np.pi) - np.pi
        if all(abs(((q4 - old + np.pi) % (2.0 * np.pi)) - np.pi) > 1e-7
               for old in roots):
            roots.append(float(q4))
    roots.sort()
    return roots


def _build_intermediate_resultants(px, py, pz, vx, vy, vz):
    # Variables are ordered as (t, c3, s3, m).
    n = 4
    t = _var(0, n)
    c3 = _var(1, n)
    s3 = _var(2, n)
    m = _var(3, n)
    one = _const(1.0, n)

    den = _add(one, _pow(t, 2))
    c4_num = _sub(one, _pow(t, 2))
    s4_num = _scale(t, 2.0)

    # Multiply every equation by enough powers of den to keep polynomials.
    ax_num = _sub(_scale(s4_num, 0.0645), _scale(den, 0.012))
    ay_num = _sub(_scale(den, 0.02), _scale(c4_num, 0.0645))
    az_num = _scale(den, 0.1775)
    l_num = _scale(den, 0.165)

    rho2 = px * px + py * py
    pos2 = rho2 + pz * pz

    # e_len * den^2:
    e_len = {}
    for p in [ax_num, ay_num, az_num, l_num]:
        e_len = _add(e_len, _mul(p, p))
    e_len = _sub(e_len, _scale(_pow(den, 2), pos2))
    e_len = _add(e_len, _scale(_mul(l_num, _sub(_mul(c3, az_num), _mul(s3, ax_num))), 2.0))

    e_unit3 = _sub(_add(_pow(c3, 2), _pow(s3, 2)), one)
    e_m = _sub(_mul(_pow(m, 2), _pow(den, 2)),
               _sub(_scale(_pow(den, 2), rho2), _mul(ay_num, ay_num)))

    u = px * vx + py * vy
    vv = py * vx - px * vy
    wx_num = _add(_scale(_mul(m, den), u), _scale(ay_num, vv))
    wy_num = _add(_scale(_mul(m, den), -vv), _scale(ay_num, u))
    d_num = _sub(_scale(_pow(den, 2), pos2), _mul(ay_num, ay_num))

    lc3_plus_c_num = _add(_mul(_const(0.165, n), _mul(c3, den)), az_num)
    minus_a_plus_ls3_num = _sub(_mul(_const(0.165, n), _mul(s3, den)), ax_num)
    a_minus_ls3_num = _sub(ax_num, _mul(_const(0.165, n), _mul(s3, den)))
    c23_num = _add(_mul(m, a_minus_ls3_num),
                   _scale(lc3_plus_c_num, pz))
    s23_num = _add(_mul(m, lc3_plus_c_num),
                   _scale(minus_a_plus_ls3_num, pz))
    inner_num = _sub(_mul(c23_num, wx_num),
                     _scale(_mul(s23_num, den), vz * rho2))
    left = _mul(_mul(s4_num, den), inner_num)
    right = _mul(c4_num, _mul(wy_num, d_num))
    e_wrist = _sub(right, left)

    res_m = resultant(e_wrist, e_m, 3)
    e_len3 = _drop_var(e_len, 3)
    e_unit3_3 = _drop_var(e_unit3, 3)
    res_c3 = resultant(res_m, e_len3, 1)
    res_len_unit = resultant(e_len3, e_unit3_3, 1)
    return res_c3, res_len_unit


def q4_resultant_numeric_poly(px, py, pz, vx, vy, vz):
    res_c3, res_len_unit = _build_intermediate_resultants(px, py, pz, vx, vy, vz)
    res_t = resultant(res_c3, res_len_unit, 1)

    return res_t


def q4_roots_numeric(px, py, pz, vx, vy, vz, tol=1e-8):
    # Remove common denominator/artifact powers implicitly by finding roots of
    # the full numeric resultant and letting FK filtering reject extras.
    return _roots_from_poly(q4_resultant_numeric_poly(px, py, pz, vx, vy, vz), tol=tol)


def q4_roots_pencil(px, py, pz, vx, vy, vz, tol=1e-8):
    res_c3, res_len_unit = _build_intermediate_resultants(px, py, pz, vx, vy, vz)
    syl = _sylvester_for_var(res_c3, res_len_unit, 1)
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
    roots = q4_roots_pencil(
        0.13353576, -0.01027953, 0.30813415,
        0.99296349, -0.11385252, 0.03257641,
    )
    print(roots)
