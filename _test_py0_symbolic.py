"""Gate test: is the symbolic resultant tractable in the py=0 canonical frame?

Also try the m-free formulation to keep degree minimal.
"""
import time
import sympy as sp

t, c3, s3, m = sp.symbols('t c3 s3 m')
# py = 0 canonical frame: target invariants are (px, pz, vx, vy, vz)
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

e_len = sp.together(
    a_x * a_x + a_z * a_z + p23_z * p23_z + a_y * a_y - pos_norm2
    + 2 * p23_z * (-s3 * a_x + c3 * a_z)
).as_numer_denom()[0]
e_unit3 = c3 * c3 + s3 * s3 - 1
e_m = sp.together(m * m - m2).as_numer_denom()[0]

u = px * vx + py * vy
v = py * vx - px * vy
wx_num = m * u + a_y * v
wy_num = -m * v + a_y * u
c23_num = m * (a_x - p23_z * s3) + pz * (p23_z * c3 + a_z)
s23_num = m * (p23_z * c3 + a_z) + pz * (-a_x + p23_z * s3)
e_wrist = sp.together(
    -s4 * (c23_num * wx_num - s23_num * vz * rho2)
    + c4 * wy_num * d
).as_numer_denom()[0]

t0 = time.perf_counter()
res_m = sp.resultant(e_wrist, e_m, m)
print(f"res_m (py=0): {time.perf_counter()-t0:.2f}s  | deg_t={sp.Poly(res_m,t).degree()} "
      f"nterms={len(sp.Poly(res_m,t,c3,s3,vx,vy,vz).terms())}", flush=True)

t0 = time.perf_counter()
res_c3 = sp.resultant(res_m, e_len, c3)
print(f"res_c3 (py=0): {time.perf_counter()-t0:.2f}s", flush=True)

t0 = time.perf_counter()
res_lu = sp.resultant(e_len, e_unit3, c3)
print(f"res_len_unit (py=0): {time.perf_counter()-t0:.2f}s", flush=True)

t0 = time.perf_counter()
res_t = sp.resultant(res_c3, res_lu, s3)
print(f"res_t (py=0): {time.perf_counter()-t0:.2f}s", flush=True)

P = sp.Poly(res_t, t)
print("res_t degree in t:", P.degree(), " #terms:", len(P.terms()))
