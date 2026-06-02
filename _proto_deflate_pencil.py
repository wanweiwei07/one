"""Deflate common (1+t^2)^k from res_c3 & res_lu (bivariate in t,s3),
then run the SAME stable pencil. Check identical roots + speedup.
"""
import time
import numpy as np
import one.robots.manipulators.denso.cvr038.ik.q4_resultant_np as rn


def s3_coeff_arr(poly, j):
    deg = max((k[0] for k in poly if k[1] == j), default=-1)
    arr = np.zeros(deg + 1) if deg >= 0 else np.zeros(1)
    for (i, jj), v in poly.items():
        if jj == j:
            arr[deg - i] += v
    return arr


def max_den_power(poly):
    """Largest k with (1+t^2)^k dividing ALL s3-coefficients of poly."""
    den = np.array([1.0, 0.0, 1.0])
    cols = [np.trim_zeros(s3_coeff_arr(poly, j), 'f') for j in range(3)]
    cols = [c if c.size else np.zeros(1) for c in cols]
    scale = max(np.max(np.abs(c)) for c in cols if c.size)
    k = 0
    while True:
        nxt = []
        ok = True
        for c in cols:
            if c.size < 3:
                ok = False
                break
            q, r = np.polydiv(c, den)
            if np.max(np.abs(r)) > 1e-7 * scale:
                ok = False
                break
            nxt.append(q)
        if not ok:
            break
        cols = nxt
        k += 1
    return k


def deflate_poly(poly, k):
    den = np.array([1.0, 0.0, 1.0])
    out = {}
    for j in range(3):
        c = s3_coeff_arr(poly, j)
        if np.trim_zeros(c, 'f').size == 0:
            continue
        for _ in range(k):
            c, r = np.polydiv(c, den)
        c = c
        deg = c.size - 1
        for idx, v in enumerate(c):
            if abs(v) > 1e-14:
                out[(deg - idx, j)] = out.get((deg - idx, j), 0.0) + float(v)
    return out


def pencil_roots_from(rc3, rlu, tol=1e-6):
    syl = rn._sylvester_for_var(rc3, rlu, 1)
    mats = rn._poly_matrix_to_coeff_mats(syl)
    roots = []
    for t in rn._polymat_roots(mats, tol=tol):
        q4 = 2.0 * np.arctan(t)
        q4 = (q4 + np.pi) % (2.0 * np.pi) - np.pi
        if all(abs(((q4 - o + np.pi) % (2.0 * np.pi)) - np.pi) > 1e-7 for o in roots):
            roots.append(float(q4))
    roots.sort()
    return roots


tests = [
    (0.13353576, -0.01027953, 0.30813415, 0.99296349, -0.11385252, 0.03257641),
    (0.25, 0.15, 0.25, 0.0, 0.0, -1.0),
    (0.18, 0.22, 0.31, -0.4, 0.5, 0.766),
    (0.30, -0.05, 0.20, 0.2, 0.3, -0.93),
]
for args in tests:
    rc3, rlu = rn._build_intermediate_resultants(*args)
    k3 = max_den_power(rc3)
    klu = max_den_power(rlu)
    k = min(k3, klu)
    rc3d = deflate_poly(rc3, k3)
    rlud = deflate_poly(rlu, klu)
    ref = rn.q4_roots_pencil(*args)
    got = pencil_roots_from(rc3d, rlud)
    print("args", [round(x, 3) for x in args], " den_pow rc3:", k3, "rlu:", klu,
          " new deg_t rc3:", rn._degree(rc3d, 0))
    print("   ref:", [round(r, 5) for r in ref])
    print("   new:", [round(r, 5) for r in got])

args = tests[0]
rc3, rlu = rn._build_intermediate_resultants(*args)
k3 = max_den_power(rc3); klu = max_den_power(rlu)
N = 50
t0 = time.perf_counter()
for _ in range(N):
    rc3, rlu = rn._build_intermediate_resultants(*args)
    rc3d = deflate_poly(rc3, k3); rlud = deflate_poly(rlu, klu)
    pencil_roots_from(rc3d, rlud)
print("deflate+pencil per call (ms):", round((time.perf_counter() - t0) / N * 1000, 3))
t0 = time.perf_counter()
for _ in range(N):
    rn.q4_roots_pencil(*args)
print("orig pencil per call (ms):", round((time.perf_counter() - t0) / N * 1000, 3))
