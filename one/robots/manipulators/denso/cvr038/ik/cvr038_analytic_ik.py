import math

import numpy as np

import one.robots.base.kine.anaik as orbka


class CVR038AnalyticIK(orbka.AnaIKBase):
    """CVR038 analytic IK (returns all solutions).

    Geometry (base frame, home):
        h1 = Z, h2 = Y, h3 = Y, h4 = Z, h5 = Y, h6 = Z
        h1 x h2, h2 // h3, h4 x h5, h5 x h6; h3 and h4 miss by a small offset.

    Solve outline.  Given q4, the wrist center fixes (q1, q2, q3) in closed form
    and the only condition left is h5 . h6 = 0, a function g(q4) with two square
    roots in it (the q3 and q1 branches).  The product over the four branches
    drops the roots, which leaves a trigonometric polynomial in q4 solved
    directly (see _q4_roots).  Each root gives q1, q2, q3, then (q5, q6) follow
    from the remaining rotation.  All candidates are FK-residual filtered.
    """
    ROOT_TOL = 1e-3     # |abs(z) - 1| for a polynomial root to count as real
    REACH_TOL = 1e-9    # slack on a*cos + b*sin = c, for a root at a tangency
    POLISH_ITERS = 3    # Newton steps on each polynomial root
    POLISH_MAX_STEP = 1e-3  # rad, a larger step is off its root (near-double)
    F_DEGREE = 8        # degree of the trigonometric polynomial F(q4)
    # the four branches, as signs of the q3 and q1 square roots
    _BRANCH_Q3 = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    _BRANCH_M = np.array([[1.0], [-1.0], [1.0], [-1.0]])

    def __init__(self, chain, joint_limits=None):
        super().__init__(chain, joint_limits=joint_limits)
        self._setup_geometry()
        n_samples = 2 * self.F_DEGREE + 1
        q4s = 2.0 * np.pi * np.arange(n_samples) / n_samples
        self._c4, self._s4 = np.cos(q4s), np.sin(q4s)
        self._orders = np.arange(self.F_DEGREE, -self.F_DEGREE - 1, -1)

    # ---- geometry from the chain (home FK) ------------------------------
    def _setup_geometry(self):
        origins = [np.asarray(o, dtype=np.float64) for o in self.chain.origins]
        axes = [np.asarray(h, dtype=np.float64) for h in self.chain.axes]
        axes = [h / np.linalg.norm(h) for h in axes]
        x12_a, x12_b, d12 = self._closest_points_on_lines(
            origins[0], axes[0], origins[1], axes[1])
        x34_on_h3, x34_on_h4, d34 = self._closest_points_on_lines(
            origins[2], axes[2], origins[3], axes[3])
        x56_a, x56_b, d56 = self._closest_points_on_lines(
            origins[4], axes[4], origins[5], axes[5])
        self.intersection_errors = (d12, d34, d56)
        x12 = (x12_a + x12_b) * 0.5
        x56 = (x56_a + x56_b) * 0.5
        self.p01 = x12
        self.p23 = x34_on_h3 - x12
        self.p34 = x34_on_h4 - x34_on_h3
        self.p45 = x56 - x34_on_h4
        self.p6t = origins[5] - x56
        self.r6t = np.asarray(self.chain.tfs[-1], dtype=np.float64)[:3, :3]
        # fast-FK tables (CVR038 axes are cardinal Z/Y, so motion is Rz/Ry)
        self._zero_tfs = [np.asarray(jnt.zero_tf, dtype=np.float64)
                          for jnt in self.chain.jnts]
        self._axis_kind = []
        for h in axes:
            i = int(np.argmax(np.abs(h)))
            self._axis_kind.append(('xyz'[i], float(np.sign(h[i]))))

    @staticmethod
    def _closest_points_on_lines(p1, h1, p2, h2):
        a = np.array([[h1 @ h1, -(h1 @ h2)], [h1 @ h2, -(h2 @ h2)]])
        b = np.array([h1 @ (p2 - p1), h2 @ (p2 - p1)])
        if abs(np.linalg.det(a)) < 1e-8:
            s, t = float(h1 @ (p2 - p1)), 0.0
        else:
            s, t = np.linalg.solve(a, b)
        c1 = p1 + h1 * s
        c2 = p2 + h2 * t
        return c1, c2, float(np.linalg.norm(c1 - c2))

    # ---- rotation helpers ----------------------------------------------
    @staticmethod
    def _rotz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _roty(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    @staticmethod
    def _rotx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    # ---- public entry ---------------------------------------------------
    def ik_all(self, tgt_tf_in_root, residual_tol=1e-4, rot_weight=0.2, **kwargs):
        tgt_tf = np.asarray(tgt_tf_in_root, dtype=np.float64)
        r06 = tgt_tf[:3, :3] @ self.r6t.T
        p16 = tgt_tf[:3, 3] - r06 @ self.p6t - self.p01
        all_qs = []
        for q4 in self._q4_roots(p16, r06[:, 2]):
            all_qs.extend(self._complete(p16, r06, float(q4)))
        all_qs = self._normalize_angles(all_qs)
        all_qs = self._filter_limits(all_qs)
        all_qs = self._filter_fk_residual(all_qs, tgt_tf, residual_tol, rot_weight)
        return self._unique([q.astype(np.float32) for q in all_qs])

    # ---- driving equation h5 . h6 = 0, as a polynomial in q4 -------------
    def _q4_roots(self, p, v):
        """Every real q4 with h5 . v = 0, for wrist center p and h6 = v.

        Writing p26 = p23 + Ry(q3) p36(q4) and y = p26 . Y:
            A cos q3 + B sin q3 = C             gives q3, root S = sqrt(r2 - C^2)
            m = sqrt(rho^2 - y^2)               gives q1 and q2
        so g = h5 . v is rational in (S, m) with denominators in r2 = A^2 + B^2
        and D = rho^2 - y^2 + pz^2.  Over the four sign choices of (S, m),
            F = g1 g2 g3 g4 D^2 r2^2
        has no roots or poles left, and is a trigonometric polynomial of degree
        8 in q4.  That degree and the D^2 r2^2 factor were measured from the
        spectrum of F on this geometry, not derived.  17 samples determine it,
        complex m and S included, and with z = exp(i q4) its roots are polynomial
        roots of degree 16.
        """
        rho2 = p[0] * p[0] + p[1] * p[1]
        if rho2 < 1e-14:    # wrist center on h1: m and q1 are not defined
            return []
        c4, s4 = self._c4, self._s4
        a23, b23, c23 = self.p23
        ax = self.p34[0] + self.p45[0] * c4 - self.p45[1] * s4
        ay = self.p34[1] + self.p45[0] * s4 + self.p45[1] * c4
        az = (self.p34[2] + self.p45[2]) * np.ones_like(c4)
        y = b23 + ay
        a3 = a23 * ax + c23 * az
        b3 = a23 * az - c23 * ax
        c3r = 0.5 * (p @ p - self.p23 @ self.p23 - ax * ax - ay * ay - az * az) - b23 * ay
        r2 = a3 * a3 + b3 * b3
        sq3 = self._BRANCH_Q3 * np.sqrt((r2 - c3r * c3r) + 0j)
        mm = self._BRANCH_M * np.sqrt((rho2 - y * y) + 0j)
        d = rho2 - y * y + p[2] * p[2]
        c3 = (a3 * c3r - b3 * sq3) / r2
        s3 = (b3 * c3r + a3 * sq3) / r2
        x = a23 + ax * c3 + az * s3
        z = c23 - ax * s3 + az * c3
        c1 = (p[0] * mm + p[1] * y) / rho2
        s1 = (p[1] * mm - p[0] * y) / rho2
        c2 = (mm * x + p[2] * z) / d
        s2 = (mm * z - p[2] * x) / d
        c23_ = c2 * c3 - s2 * s3
        s23_ = s2 * c3 + c2 * s3
        ux = c1 * v[0] + s1 * v[1]
        uy = -s1 * v[0] + c1 * v[1]
        g = -s4 * (c23_ * ux - s23_ * v[2]) + c4 * uy
        f = (np.prod(g, axis=0) * d * d * r2 * r2).real
        if not np.all(np.isfinite(f)):
            return []
        coeffs = (np.fft.fft(f) / f.size)[self._orders]  # z^16 ... z^0
        z_roots = np.roots(coeffs)
        roots = np.angle(z_roots[np.abs(np.abs(z_roots) - 1.0) < self.ROOT_TOL])
        for _ in range(self.POLISH_ITERS):
            terms = coeffs * np.exp(1j * np.outer(roots, self._orders))
            step = (np.sum(terms, axis=1).real
                    / np.sum(1j * self._orders * terms, axis=1).real)
            roots = roots - np.where(np.abs(step) < self.POLISH_MAX_STEP, step, 0.0)
        return roots

    # ---- closed-form completion for one q4 ------------------------------
    def _cos_sin_solve(self, a, b, c):
        # all theta with a*cos(theta) + b*sin(theta) = c
        r = math.hypot(a, b)
        if r < 1e-12:
            return []
        ratio = c / r
        if abs(ratio) > 1.0 + self.REACH_TOL:
            return []
        ratio = max(-1.0, min(1.0, ratio))
        phi0 = math.atan2(b, a)
        delta = math.acos(ratio)
        if delta < 1e-9:
            return [phi0]
        return [phi0 + delta, phi0 - delta]

    def _complete(self, p16, r06, q4):
        c4, s4 = math.cos(q4), math.sin(q4)
        px, py, pz = float(p16[0]), float(p16[1]), float(p16[2])
        a23, b23, c23 = (float(x) for x in self.p23)
        p36x = float(self.p34[0] + self.p45[0] * c4 - self.p45[1] * s4)
        p36y = float(self.p34[1] + self.p45[0] * s4 + self.p45[1] * c4)
        p36z = float(self.p34[2] + self.p45[2])
        # ||p23 + Ry(q3) p36|| = ||p16||  ->  A cos q3 + B sin q3 = C
        a3 = a23 * p36x + c23 * p36z
        b3 = a23 * p36z - c23 * p36x
        c3r = (0.5 * (px * px + py * py + pz * pz - a23 * a23 - b23 * b23 - c23 * c23
                      - p36x * p36x - p36y * p36y - p36z * p36z) - b23 * p36y)
        out = []
        for q3 in self._cos_sin_solve(a3, b3, c3r):
            c3, s3 = math.cos(q3), math.sin(q3)
            p26x = a23 + p36x * c3 + p36z * s3
            p26y = b23 + p36y
            p26z = c23 - p36x * s3 + p36z * c3
            # p16 = Rz(q1) Ry(q2) p26, so p26z cos q2 - p26x sin q2 = p16z
            for q2 in self._cos_sin_solve(p26z, -p26x, pz):
                vx = p26x * math.cos(q2) + p26z * math.sin(q2)
                q1 = math.atan2(py, px) - math.atan2(p26y, vx)
                # what is left is R46 = Ry(q5) Rz(q6)
                r03 = self._rotz(q1) @ self._roty(q2 + q3)
                r46 = self._rotz(-q4) @ r03.T @ r06
                q5 = math.atan2(r46[0, 2], r46[2, 2])
                r56 = self._roty(-q5) @ r46
                q6 = math.atan2(r56[1, 0], r56[0, 0])
                out.append(np.array([q1, q2, q3, q4, q5, q6]))
        return out

    # ---- FK residual + housekeeping ------------------------------------
    def _fk_tf(self, qs):
        rmap = {'x': self._rotx, 'y': self._roty, 'z': self._rotz}
        tf = np.eye(4)
        m = np.eye(4)
        for ztf, (char, sign), q in zip(self._zero_tfs, self._axis_kind, qs):
            m[:3, :3] = rmap[char](sign * float(q))
            tf = tf @ ztf @ m
        return tf

    def _fk_error(self, qs, tgt_tf, rot_weight):
        cur_tf = self._fk_tf(qs)
        pos_err = tgt_tf[:3, 3] - cur_tf[:3, 3]
        # small-angle rotation error from the antisymmetric part (~sin theta ~
        # theta); stable near 0, unlike arccos, and scipy-free
        rel = cur_tf[:3, :3].T @ tgt_tf[:3, :3]
        vx = rel[2, 1] - rel[1, 2]
        vy = rel[0, 2] - rel[2, 0]
        vz = rel[1, 0] - rel[0, 1]
        theta = 0.5 * math.sqrt(vx * vx + vy * vy + vz * vz)
        return math.sqrt(float(pos_err @ pos_err) + (rot_weight * theta) ** 2)

    def _filter_fk_residual(self, qs_list, tgt_tf, residual_tol, rot_weight):
        return [q for q in qs_list
                if self._fk_error(q, tgt_tf, rot_weight) <= residual_tol]

    @staticmethod
    def _normalize_angles(qs_list):
        return [(q + np.pi) % (2.0 * np.pi) - np.pi for q in qs_list]
