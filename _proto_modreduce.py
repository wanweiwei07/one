"""Find target-independent artifact factors (1+t^2)^k and t^m in res_t,
then test whether np.roots on the DEFLATED genuine polynomial is stable.
"""
import numpy as np
import one.robots.manipulators.denso.cvr038.ik.q4_resultant_np as rn


def s3_coeffs(poly):
    out = []
    for j in range(3):
        deg = max((k[0] for k in poly if k[1] == j), default=-1)
        arr = np.zeros(deg + 1) if deg >= 0 else np.zeros(1)
        for (i, jj), v in poly.items():
            if jj == j:
                arr[deg - i] += v
        out.append(arr)
    return out


def psub(a, b):
    n = max(a.size, b.size)
    return np.pad(a, (n - a.size, 0)) - np.pad(b, (n - b.size, 0))


def res_t_quad(rc3, rlu):
    A0, A1, A2 = s3_coeffs(rc3)
    B0, B1, B2 = s3_coeffs(rlu)
    t1 = psub(np.polymul(A2, B0), np.polymul(A0, B2))
    t1 = np.polymul(t1, t1)
    t2 = psub(np.polymul(A2, B1), np.polymul(A1, B2))
    t3 = psub(np.polymul(A1, B0), np.polymul(A0, B1))
    return np.trim_zeros(psub(t1, np.polymul(t2, t3)), 'f')


def deflate(coeffs):
    """Divide out t^m (trailing zeros) and (1+t^2)^k. Return (defl, m, k)."""
    c = coeffs.copy()
    # strip trailing zeros (factors of t): roots at 0
    m = 0
    while c.size > 1 and abs(c[-1]) < 1e-9 * np.max(np.abs(c)):
        c = c[:-1]
        m += 1
    # divide by (1+t^2) while it divides cleanly
    den = np.array([1.0, 0.0, 1.0])
    k = 0
    while c.size > 2:
        q, r = np.polydiv(c, den)
        if np.max(np.abs(r)) < 1e-6 * np.max(np.abs(c)):
            c = q
            k += 1
        else:
            break
    return c, m, k


def roots_q4(coeffs, tol=1e-6):
    coeffs = coeffs / np.max(np.abs(coeffs))
    out = []
    for r in np.roots(coeffs):
        if abs(r.imag) > tol:
            continue
        q4 = 2.0 * np.arctan(r.real)
        q4 = (q4 + np.pi) % (2.0 * np.pi) - np.pi
        if all(abs(((q4 - o + np.pi) % (2.0 * np.pi)) - np.pi) > 1e-6 for o in out):
            out.append(float(q4))
    out.sort()
    return out


tests = [
    (0.13353576, -0.01027953, 0.30813415, 0.99296349, -0.11385252, 0.03257641),
    (0.25, 0.15, 0.25, 0.0, 0.0, -1.0),
    (0.18, 0.22, 0.31, -0.4, 0.5, 0.766),
    (0.30, -0.05, 0.20, 0.2, 0.3, -0.93),
]
for args in tests:
    rc3, rlu = rn._build_intermediate_resultants(*args)
    rt = res_t_quad(rc3, rlu)
    defl, m, k = deflate(rt)
    print("args", [round(x, 3) for x in args])
    print("   res_t deg:", rt.size - 1, " t^m m=", m, " (1+t^2)^k k=", k,
          " genuine deg:", defl.size - 1)
    print("   deflated roots:", [round(r, 5) for r in roots_q4(defl)])
    print("   pencil   roots:", [round(r, 5) for r in rn.q4_roots_pencil(*args)])
