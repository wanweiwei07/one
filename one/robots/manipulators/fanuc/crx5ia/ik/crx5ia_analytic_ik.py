import math

import numpy as np

import one.robots.base.kine.anaik as orbka


class CRX5iaAnalyticIK(orbka.AnaIKBase):
    """CRX-5iA analytic IK (returns all solutions).

    Geometry (base frame, home):
        h1 = Z, h2 = Y, h3 = -Y, h4 = -X, h5 = -Y, h6 = -X
        h2 // h3 (shoulder/elbow), h4 // h6.

    The FANUC CRX offsets are each parallel to their own joint axis:
        j4 -> j5 = [L4, 0, 0]  is parallel to h4  ==> q4 does not move j5
        j5 -> j6 = [0, -L5, 0] is parallel to h5  ==> q5 does not move j6
    Consequences that make this much simpler than an offset wrist in general:
        * p_j5 (axis4 ^ axis5 intersection) depends only on (q1, q2, q3).
        * p_j6 is read straight off the target: p_j6 = p_tcp - R06 @ d6t.

    Solve outline.  The only coupling is the L5 wrist offset along h5, whose
    direction h5 is unknown.  h5 is perpendicular to the known h6, so it lives
    on a circle parameterised by one angle phi:
        h5(phi) = cos(phi) e1 + sin(phi) e2,   e1, e2 _|_ h6
        p_j5    = p_j6 - L5 * h5(phi)
    Given p_j5 the arm (h1, h2//h3) is closed form, and h5 _|_ h4 must hold, i.e.
    "wrist-offset segment _|_ forearm".  Eliminating the elbow's square root
    turns that into a trigonometric polynomial in phi (see _phi_roots), solved
    directly.  Each root gives q1, q2, q3, and h5 gives q4, then (q5, q6) follow
    from the remaining Y-X rotation.  This has no gimbal case: q4 is never split
    from q6 through the wrist rotation, so q5 = 0 needs no special handling.
    All candidates are FK-residual filtered.
    """
    ROOT_TOL = 1e-3     # |abs(z) - 1| for a polynomial root to count as real
    REACH_TOL = 1e-7    # slack on the elbow cosine, for a root at full stretch
    G_TOL = 1e-3        # |h5 . h4| selects the arm branches of a root
    POLISH_ITERS = 3    # Newton steps on each polynomial root
    POLISH_MAX_STEP = 1e-3  # rad, a larger step is off its root (near-double)

    def __init__(self, chain, joint_limits=None):
        super().__init__(chain, joint_limits=joint_limits)
        self._setup_geometry()

    # ---- geometry from the chain (home FK), frame agnostic --------------
    def _setup_geometry(self):
        tf = np.eye(4, dtype=np.float64)
        frames = []
        for jnt in self.chain.jnts:
            tf = tf @ np.asarray(jnt.zero_tf, dtype=np.float64)
            frames.append(tf.copy())
        p_j2 = frames[1][:3, 3]
        p_j3 = frames[2][:3, 3]
        p_j5 = frames[4][:3, 3]
        p_j6 = frames[5][:3, 3]
        self.z0 = float(p_j2[2])                     # base height to j2
        self.l2 = float(np.linalg.norm(p_j3 - p_j2))  # upper arm
        self.l4 = float(np.linalg.norm(p_j5 - p_j3))  # forearm
        self.l5 = float(np.linalg.norm(p_j6 - p_j5))  # wrist offset
        # local joint axes
        self.h_local = [np.asarray(a, dtype=np.float64) for a in self.chain.axes]
        # constant frame6 -> tcp transform (identity for the bare flange)
        t6_home = frames[5]
        t_tcp_home = np.asarray(self.chain.tfs[-1], dtype=np.float64)
        self.t_6tcp = np.linalg.inv(t6_home) @ t_tcp_home
        # fast-FK tables: zero_tf and (cardinal axis, sign) per joint
        self._zero_tfs = [np.asarray(jnt.zero_tf, dtype=np.float64)
                          for jnt in self.chain.jnts]
        self._axis_kind = []
        for h in self.h_local:
            i = int(np.argmax(np.abs(h)))
            self._axis_kind.append(('xyz'[i], float(np.sign(h[i]))))

    # ---- rotation helpers ----------------------------------------------
    @staticmethod
    def _rotz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _roty(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    @staticmethod
    def _rotx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    # ---- public entry ---------------------------------------------------
    def ik_all(self, tgt_tf_in_root, residual_tol=1e-4, **kwargs):
        tgt_tf = np.asarray(tgt_tf_in_root, dtype=np.float64)
        t06 = tgt_tf @ np.linalg.inv(self.t_6tcp)
        r06 = t06[:3, :3]
        p_j6 = t06[:3, 3]

        h6 = r06 @ self.h_local[5]
        ref = np.array([1.0, 0.0, 0.0]) if abs(h6[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(h6, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(h6, e1)

        all_qs = []
        for phi in self._phi_roots(p_j6, e1, e2):
            h5 = math.cos(phi) * e1 + math.sin(phi) * e2
            p_j5 = p_j6 - self.l5 * h5
            for label in ((0, 1), (0, -1), (1, 1), (1, -1)):
                q123 = self._arm_branch(p_j5, label)
                if q123 is None:
                    continue
                q1, q2, q3 = q123
                # h4 = Rz(q1) Ry(q2 - q3) (-X), closed form
                cpsi, spsi = math.cos(q2 - q3), math.sin(q2 - q3)
                h4 = np.array([-cpsi * math.cos(q1), -cpsi * math.sin(q1), spsi])
                if abs(h5 @ h4) <= self.G_TOL:
                    all_qs.append(self._complete(q123, h5, r06))

        all_qs = self._wrap_to_limits(all_qs)
        all_qs = self._filter_fk_residual(all_qs, tgt_tf, residual_tol)
        return self._unique(all_qs)

    # ---- driving equation h5 . h4 = 0, as a polynomial in phi ------------
    def _phi_roots(self, p_j6, e1, e2):
        """Every real phi with h5 . h4 = 0.

        With v = p_j5 - p_j2, d2 = v . v, the forearm is f = a v + b w, where
        w is the unit vector _|_ v in the arm's vertical plane.  The triangle
        (l2, l4, |v|) gives
            a d2 = K / 2,   b^2 = l4^2 - K^2 / (4 d2),   K = d2 + l4^2 - l2^2
        and h5 . w = (B d2 - A vz) / (rho |v|), with A = h5 . v, B = h5 . Z,
        rho^2 = d2 - vz^2.  Squaring h5 . f = 0 away from the sign of b (the
        elbow) and clearing denominators:
            F = K^2 A^2 (d2 - vz^2) - (4 d2 l4^2 - K^2) (B d2 - A vz)^2 = 0
        d2, K, A, vz, B are all degree 1 in (cos phi, sin phi), so F is a
        trigonometric polynomial of degree 6.  13 samples determine it exactly,
        and with z = exp(i phi) its roots are polynomial roots of degree 12.
        """
        phis = 2.0 * np.pi * np.arange(13) / 13
        h5 = np.outer(np.cos(phis), e1) + np.outer(np.sin(phis), e2)
        v = p_j6 - np.array([0.0, 0.0, self.z0]) - self.l5 * h5
        d2 = np.sum(v * v, axis=1)
        k = d2 + self.l4 ** 2 - self.l2 ** 2
        a = np.sum(h5 * v, axis=1)
        vz = v[:, 2]
        b = h5[:, 2]
        f = (k * k * a * a * (d2 - vz * vz)
             - (4.0 * d2 * self.l4 ** 2 - k * k) * (b * d2 - a * vz) ** 2)
        c = np.fft.fft(f) / 13
        coeffs = c[np.arange(6, -7, -1)]    # z^12 ... z^0, negative k wraps
        if np.abs(coeffs).max() < 1e-14:    # F = 0 for all phi: a self-motion
            return []
        z = np.roots(coeffs)
        roots = np.angle(z[np.abs(np.abs(z) - 1.0) < self.ROOT_TOL])
        # eigenvalues are only good to ~1e-6 here, and near full stretch q moves
        # as sqrt(phi - limit): polish on F itself
        orders = np.arange(6, -7, -1)
        for _ in range(self.POLISH_ITERS):
            terms = coeffs * np.exp(1j * np.outer(roots, orders))
            step = np.sum(terms, axis=1).real / np.sum(1j * orders * terms, axis=1).real
            roots = roots - np.where(np.abs(step) < self.POLISH_MAX_STEP, step, 0.0)
        return roots

    # ---- closed-form arm: p_j5 -> (q1, q2, q3) for one branch label -----
    def _arm_branch(self, p_j5, label):
        q1_branch, elbow = label
        px, py, pz = float(p_j5[0]), float(p_j5[1]), float(p_j5[2])
        q1 = math.atan2(py, px) + (math.pi if q1_branch else 0.0)
        r = px * math.cos(q1) + py * math.sin(q1)
        h = pz - self.z0
        l2, l4 = self.l2, self.l4
        cos_el = (r * r + h * h - l2 * l2 - l4 * l4) / (2.0 * l2 * l4)
        if abs(cos_el) > 1.0 + self.REACH_TOL:
            return None
        cos_el = -1.0 if cos_el < -1.0 else (1.0 if cos_el > 1.0 else cos_el)
        el = elbow * math.acos(cos_el)
        beta = math.atan2(l4 * math.sin(el), l2 + l4 * math.cos(el))
        q2 = math.atan2(r, h) - beta
        ux, uz = l2 * math.sin(q2), l2 * math.cos(q2)
        fx, fz = r - ux, h - uz
        q3 = q2 - math.atan2(-fz, fx)
        return q1, q2, q3

    # ---- wrist: R36 = Rx(-q4) Ry(-q5) Rx(-q6) ---------------------------
    def _complete(self, q123, h5, r06):
        q1, q2, q3 = q123
        r03 = self._rotz(q1) @ self._roty(q2) @ self._roty(-q3)
        # h5 in frame 3 is Rx(-q4) (-Y) = (0, -cos q4, sin q4)
        h5_3 = r03.T @ h5
        q4 = math.atan2(h5_3[2], -h5_3[1])
        # what is left is Ry(-q5) Rx(-q6)
        r = self._rotx(q4) @ r03.T @ r06
        q5 = -math.atan2(-r[2, 0], r[0, 0])
        q6 = -math.atan2(-r[1, 2], r[1, 1])
        return np.array([q1, q2, q3, q4, q5, q6], dtype=np.float32)

    # ---- FK residual + housekeeping ------------------------------------
    def _fk_tf(self, qs):
        rmap = {'x': self._rotx, 'y': self._roty, 'z': self._rotz}
        tf = np.eye(4, dtype=np.float64)
        m = np.eye(4, dtype=np.float64)
        for ztf, (char, sign), q in zip(self._zero_tfs, self._axis_kind, qs):
            m[:3, :3] = rmap[char](sign * float(q))
            tf = tf @ ztf @ m
        return tf

    def _fk_error(self, qs, tgt_tf):
        cur = self._fk_tf(qs)
        pos_err = np.linalg.norm(cur[:3, 3] - tgt_tf[:3, 3])
        rot_err = np.linalg.norm(cur[:3, :3] - tgt_tf[:3, :3])
        return pos_err + rot_err

    def _filter_fk_residual(self, qs_list, tgt_tf, residual_tol):
        return [q for q in qs_list if self._fk_error(q, tgt_tf) <= residual_tol]

    def _wrap_to_limits(self, qs_list):
        """Per-joint 2*pi fold into [lo, hi]; drop any joint that cannot fit.

        CRX joints have wide, asymmetric ranges (e.g. q3 in [-68, 248] deg),
        so a plain wrap to [-pi, pi] would push valid solutions out of range.
        """
        if self.joint_limits is None:
            return [q.astype(np.float32) for q in qs_list]
        lo, hi = self.joint_limits
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        two_pi = 2.0 * np.pi
        out = []
        for q in qs_list:
            wq = np.asarray(q, dtype=np.float64).copy()
            ok = True
            for i in range(wq.size):
                a = wq[i] - two_pi * np.floor((wq[i] - lo[i]) / two_pi)
                if a > hi[i] + 1e-9:
                    ok = False
                    break
                wq[i] = a
            if ok:
                out.append(wq.astype(np.float32))
        return out
