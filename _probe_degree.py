"""Probe the TRUE minimal q4 polynomial degree across random targets."""
import numpy as np
import sympy as sp
import one.robots.manipulators.denso.cvr038.ik.derive_q4_poly as dq

rng = np.random.default_rng(0)
seen_degrees = {}
samples = [
    (0.25, 0.15, 0.25, 0.0, 0.0, -1.0),
    (0.13353576, -0.01027953, 0.30813415, 0.99296349, -0.11385252, 0.03257641),
    (0.3, -0.1, 0.2, 0.2, 0.3, -0.9),
    (0.18, 0.22, 0.31, -0.4, 0.5, 0.766),
]
for s in samples:
    poly = dq.q4_resultant_poly(*s)
    facs = dq.effective_q4_factors(poly)
    facs_degs = []
    t = poly.gens[0]
    for factor, mult in facs:
        d = sp.Poly(factor, t).degree()
        facs_degs.append((d, mult))
    total_real_roots = len(dq.q4_real_roots(*s))
    print("target", [round(x, 3) for x in s])
    print("   full deg:", poly.degree(), " effective factors (deg,mult):", facs_degs,
          " #real q4:", total_real_roots)
