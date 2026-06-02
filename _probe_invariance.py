"""Verify q4 polynomial depends on target only through base-Z-rotation invariants.

If true, rotating both p and v about Z by the same angle must leave the
numeric res_t (up to scale) unchanged.
"""
import numpy as np
import one.robots.manipulators.denso.cvr038.ik.q4_resultant_np as rn


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def res_t_coeffs(px, py, pz, vx, vy, vz):
    poly = rn.q4_resultant_numeric_poly(px, py, pz, vx, vy, vz)
    c = rn._poly_to_univar_coeffs(poly)
    # normalize: strip leading/trailing zeros and scale by max abs
    c = c / np.max(np.abs(c))
    return c


p = np.array([0.13353576, -0.01027953, 0.30813415])
v = np.array([0.99296349, -0.11385252, 0.03257641])
v = v / np.linalg.norm(v)

base = res_t_coeffs(*p, *v)
print("base degree:", base.size - 1)
for ang in [0.3, 1.1, -2.0, 2.7]:
    R = rotz(ang)
    p2 = R @ p
    v2 = R @ v
    c2 = res_t_coeffs(*p2, *v2)
    if c2.size != base.size:
        print(f"ang={ang}: SIZE MISMATCH {c2.size} vs {base.size}")
        continue
    # align sign
    err = min(np.max(np.abs(c2 - base)), np.max(np.abs(c2 + base)))
    print(f"ang={ang}: max coeff diff after Z-rot = {err:.2e}")
