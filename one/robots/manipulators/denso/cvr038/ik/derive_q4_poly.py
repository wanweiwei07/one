"""
Derive the CVR038 q4 half-angle polynomial.

This is an offline derivation helper, not the runtime IK solver.

Variables:
    t = tan(q4 / 2)
    c4 = (1 - t^2) / (1 + t^2)
    s4 = 2t / (1 + t^2)

Inputs are expressed in the equivalent IK-Geo frame used by CVR038PencilIK:
    p = p06 - R06 @ p6t - p01 = (px, py, pz)
    v = R06 @ ez = (vx, vy, vz)

The script eliminates q1, q2, q3 and the shoulder branch variable m from:
    p = Rz(q1) Ry(q2) (p23 + Ry(q3) (p34 + Rz(q4) p45))
    e_y.T Rz(-q4) Ry(-(q2 + q3)) Rz(-q1) v = 0

The output polynomial contains extraneous factors from squaring/elimination.
Runtime code should use the real roots as q4 candidates and keep the existing
FK residual filter.
"""

import argparse
import sympy as sp


def _rat(value):
    return sp.Rational(str(value))


def q4_resultant_poly(px, py, pz, vx, vy, vz):
    t, c3, s3, m = sp.symbols('t c3 s3 m')
    px = _rat(px)
    py = _rat(py)
    pz = _rat(pz)
    vx = _rat(vx)
    vy = _rat(vy)
    vz = _rat(vz)

    c4 = (1 - t * t) / (1 + t * t)
    s4 = 2 * t / (1 + t * t)

    # CVR038 equivalent geometry:
    # p23 = [0, 0.020, 0.165]
    # p34 = [-0.012, 0, 0]
    # p45 = [0, -0.0645, 0.1775]
    a_x = sp.Rational(129, 2000) * s4 - sp.Rational(3, 250)
    a_y = sp.Rational(1, 50) - sp.Rational(129, 2000) * c4
    a_z = sp.Rational(71, 400)
    p23_z = sp.Rational(33, 200)

    rho2 = px * px + py * py
    pos_norm2 = rho2 + pz * pz
    m2 = rho2 - a_y * a_y
    d = pos_norm2 - a_y * a_y

    # q3 length equation after using c3^2 + s3^2 = 1.
    e_len = sp.together(
        a_x * a_x + a_z * a_z + p23_z * p23_z + a_y * a_y - pos_norm2
        + 2 * p23_z * (-s3 * a_x + c3 * a_z)
    ).as_numer_denom()[0]
    e_unit3 = c3 * c3 + s3 * s3 - 1
    e_m = sp.together(m * m - m2).as_numer_denom()[0]

    # q1 branch variable:
    #   u = px*vx + py*vy
    #   v = py*vx - px*vy
    u = px * vx + py * vy
    v = py * vx - px * vy
    wx_num = m * u + a_y * v
    wy_num = -m * v + a_y * u

    # q2+q3 in terms of q2 branch m and q3.
    c23_num = m * (a_x - p23_z * s3) + pz * (p23_z * c3 + a_z)
    s23_num = m * (p23_z * c3 + a_z) + pz * (-a_x + p23_z * s3)

    # Wrist feasibility: R46[1, 2] = 0.  Multipled by denominators rho2*d.
    e_wrist = sp.together(
        -s4 * (c23_num * wx_num - s23_num * vz * rho2)
        + c4 * wy_num * d
    ).as_numer_denom()[0]

    res_m = sp.resultant(e_wrist, e_m, m)
    res_c3 = sp.resultant(res_m, e_len, c3)
    res_len_unit = sp.resultant(e_len, e_unit3, c3)
    res_t = sp.resultant(res_c3, res_len_unit, s3)
    return sp.Poly(sp.factor(res_t), t)


def q4_real_roots(px, py, pz, vx, vy, vz, tol=1e-8):
    poly = q4_resultant_poly(px, py, pz, vx, vy, vz)
    t = poly.gens[0]
    roots = []
    for factor, _mult in effective_q4_factors(poly):
        factor_poly = sp.Poly(factor, t)
        for root in factor_poly.nroots(n=30, maxsteps=200):
            root = complex(root)
            if abs(root.imag) > tol:
                continue
            q4 = float(2.0 * sp.atan(root.real))
            q4 = (q4 + float(sp.pi)) % (2.0 * float(sp.pi)) - float(sp.pi)
            if all(abs(((q4 - old + float(sp.pi)) % (2.0 * float(sp.pi))) - float(sp.pi)) > 1e-7
                   for old in roots):
                roots.append(q4)
    roots.sort()
    return roots


def effective_q4_factors(poly):
    """Return non-artifact factors that may produce real finite q4 roots."""
    t = poly.gens[0]
    out = []
    for factor, mult in sp.factor_list(poly.as_expr())[1]:
        factor_poly = sp.Poly(factor, t)
        degree = factor_poly.degree()
        if degree <= 0:
            continue
        if factor == t or factor == t ** 2 + 1:
            continue
        # Drop factors whose roots are all non-real for this target.  These are
        # usually squared artifacts from eliminating q3 branches.
        roots = factor_poly.nroots(n=30, maxsteps=200)
        if not any(abs(complex(root).imag) < 1e-8 for root in roots):
            continue
        out.append((factor, mult))
    return out


def main():
    parser = argparse.ArgumentParser(description='Derive CVR038 q4 polynomial.')
    parser.add_argument('--p', nargs=3, type=float, default=(0.25, 0.15, 0.25),
                        metavar=('PX', 'PY', 'PZ'))
    parser.add_argument('--v', nargs=3, type=float, default=(0.0, 0.0, -1.0),
                        metavar=('VX', 'VY', 'VZ'))
    parser.add_argument('--no-factor', action='store_true')
    parser.add_argument('--roots', action='store_true')
    parser.add_argument('--effective', action='store_true')
    args = parser.parse_args()

    poly = q4_resultant_poly(*args.p, *args.v)
    expr = poly.as_expr() if args.no_factor else sp.factor(poly.as_expr())
    print('degree:', poly.degree())
    print('terms:', len(poly.terms()))
    print(expr)
    if args.effective:
        print('effective factors:')
        for factor, mult in effective_q4_factors(poly):
            print('mult:', mult, 'degree:', sp.Poly(factor, poly.gens[0]).degree())
            print(factor)
    if args.roots:
        print('q4 roots:', q4_real_roots(*args.p, *args.v))


if __name__ == '__main__':
    main()
