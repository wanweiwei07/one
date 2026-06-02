"""Offline: derive res_c3(t,s3), res_len_unit(t,s3) symbolically in py=0
invariants, exactly deflate (1+t^2)^k and t^m, persist to disk.

Then validate (numeric target) that the deflated pencil gives the SAME roots
as the current solver, faster.
"""
import pickle
import time
import numpy as np
import sympy as sp

t, c3, s3, m = sp.symbols('t c3 s3 m')
px, pz, vx, vy, vz = sp.symbols('px pz vx vy vz', real=True)
py = sp.Integer(0)

c4 = (1 - t * t) / (1 + t * t)
s4 = 2 * t / (1 + t * t)
a_x = sp.Rational(129, 2000) * s4 - sp.Rational(3, 250)
a_y = sp.Rational(1, 50) - sp.Rational(129, 2000) * c4
a_z = sp.Rational(71, 400)
p23_z = sp.Rational(33, 200)
rho2 = px * px + py * py
pos_norm2 = rho2 + pz * pz
m2 = rho2 - a_y * a_y
d = pos_norm2 - a_y * a_y

e_len = sp.together(a_x * a_x + a_z * a_z + p23_z * p23_z + a_y * a_y - pos_norm2
                    + 2 * p23_z * (-s3 * a_x + c3 * a_z)).as_numer_denom()[0]
e_unit3 = c3 * c3 + s3 * s3 - 1
e_m = sp.together(m * m - m2).as_numer_denom()[0]
u = px * vx + py * vy
vv = py * vx - px * vy
wx_num = m * u + a_y * vv
wy_num = -m * vv + a_y * u
c23_num = m * (a_x - p23_z * s3) + pz * (p23_z * c3 + a_z)
s23_num = m * (p23_z * c3 + a_z) + pz * (-a_x + p23_z * s3)
e_wrist = sp.together(-s4 * (c23_num * wx_num - s23_num * vz * rho2)
                      + c4 * wy_num * d).as_numer_denom()[0]

print("deriving res_m...", flush=True); t0 = time.perf_counter()
res_m = sp.resultant(e_wrist, e_m, m)
print(f"  {time.perf_counter()-t0:.1f}s", flush=True)
print("deriving res_c3...", flush=True); t0 = time.perf_counter()
res_c3 = sp.Poly(sp.resultant(res_m, e_len, c3), t, s3)
print(f"  {time.perf_counter()-t0:.1f}s", flush=True)
print("deriving res_lu...", flush=True); t0 = time.perf_counter()
res_lu = sp.Poly(sp.resultant(e_len, e_unit3, c3), t, s3)
print(f"  {time.perf_counter()-t0:.1f}s", flush=True)


def deflate(poly):
    """Divide out the largest (1+t^2)^k and t^m common to all s3-coeffs."""
    den = sp.Poly(1 + t * t, t, s3, px, pz, vx, vy, vz)
    tt = sp.Poly(t, t, s3, px, pz, vx, vy, vz)
    P = sp.Poly(poly, t, s3, px, pz, vx, vy, vz)
    k = 0
    while True:
        q, r = sp.div(P, den)
        if r.is_zero:
            P = q; k += 1
        else:
            break
    mt = 0
    while True:
        q, r = sp.div(P, tt)
        if r.is_zero:
            P = q; mt += 1
        else:
            break
    return P, k, mt


res_c3_d, k3, m3 = deflate(res_c3)
res_lu_d, klu, mlu = deflate(res_lu)
print(f"res_c3: deg_t {res_c3.degree(t)} -> {res_c3_d.degree(t)}  (1+t^2)^{k3} t^{m3}")
print(f"res_lu: deg_t {res_lu.degree(t)} -> {res_lu_d.degree(t)}  (1+t^2)^{klu} t^{mlu}")

with open('cvr038_q4_sym.pkl', 'wb') as f:
    pickle.dump({
        'res_c3_d': sp.srepr(res_c3_d.as_expr()),
        'res_lu_d': sp.srepr(res_lu_d.as_expr()),
        'syms': 't s3 px pz vx vy vz',
    }, f)
print("persisted to cvr038_q4_sym.pkl")
