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
    Given p_j5 the arm (h1, h2//h3) is closed form (q1 = atan2, 2-link elbow),
    which yields R03 and h4 = R03 @ (-X).  Because h5 _|_ h4 always holds, phi is
    pinned by the single scalar equation
        g(phi) = h5(phi) . h4(phi) = 0
    i.e. geometrically "wrist-offset segment _|_ forearm".  g has only ~2 roots
    per arm branch (cf. the degree-16 q4 polynomial of CVR038), so a robust 1-D
    bracket/bisection over phi finds them all.  q4 then comes from h5, and
    (q5, q6) from an X-Y-X Euler split of R36 = R03^T R06.  All candidates are
    FK-residual filtered, so any numerically marginal phi root is rejected.
    """

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
    def ik_all(self, tgt_tf_in_root, residual_tol=1e-4, n_phi=720, **kwargs):
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
        for label, phi in self._phi_roots(p_j6, e1, e2, n_phi):
            cand = self._complete(label, phi, p_j6, e1, e2, r06)
            all_qs.extend(cand)

        all_qs = self._wrap_to_limits(all_qs)
        all_qs = self._filter_fk_residual(all_qs, tgt_tf, residual_tol)
        return self._unique(all_qs)

    # ---- driving equation g(phi) = h5 . h4 ------------------------------
    def _g(self, label, phi, p_j6, e1, e2):
        cphi, sphi = math.cos(phi), math.sin(phi)
        h5x = cphi * e1[0] + sphi * e2[0]
        h5y = cphi * e1[1] + sphi * e2[1]
        h5z = cphi * e1[2] + sphi * e2[2]
        p5 = (p_j6[0] - self.l5 * h5x,
              p_j6[1] - self.l5 * h5y,
              p_j6[2] - self.l5 * h5z)
        q123 = self._arm_branch(p5, label)
        if q123 is None:
            return None, None
        q1, q2, q3 = q123
        psi = q2 - q3                       # h4 = Rz(q1) Ry(psi) (-X), closed form
        cpsi, spsi = math.cos(psi), math.sin(psi)
        # g = h5 . h4,  h4 = [-cpsi cos q1, -cpsi sin q1, spsi]
        g = -cpsi * math.cos(q1) * h5x - cpsi * math.sin(q1) * h5y + spsi * h5z
        return g, q123

    def _g_all_labels(self, phis, p_j6, e1, e2):
        """Vectorised g(phi) for all 4 arm branches at once.

        The wrist point, elbow magnitude and arccos are shared across branches;
        only q1 (+pi for the back branch) and the elbow sign differ.  Returns a
        dict {label: g_array} with NaN where the arm cannot reach.
        """
        h5 = np.outer(np.cos(phis), e1) + np.outer(np.sin(phis), e2)
        p5 = p_j6[None, :] - self.l5 * h5
        px, py, pz = p5[:, 0], p5[:, 1], p5[:, 2]
        h = pz - self.z0
        a = np.arctan2(py, px)
        rho = np.hypot(px, py)
        reach_far = (self.l2 + self.l4) ** 2
        reach_near = (self.l2 - self.l4) ** 2

        out = {}
        for q1_branch in (0, 1):
            q1 = a + (np.pi if q1_branch else 0.0)
            r = rho if q1_branch == 0 else -rho     # px cos q1 + py sin q1
            d2 = r * r + h * h
            reach = (d2 <= reach_far) & (d2 >= reach_near)
            d2c = np.clip(d2, reach_near, reach_far)
            ac = np.arccos(np.clip((d2c - self.l2 ** 2 - self.l4 ** 2)
                                   / (2.0 * self.l2 * self.l4), -1.0, 1.0))
            atan_rh = np.arctan2(r, h)
            cq1, sq1 = np.cos(q1), np.sin(q1)
            for elbow in (+1, -1):
                el = elbow * ac
                beta = np.arctan2(self.l4 * np.sin(el),
                                  self.l2 + self.l4 * np.cos(el))
                q2 = atan_rh - beta
                ux, uz = self.l2 * np.sin(q2), self.l2 * np.cos(q2)
                fx, fz = r - ux, h - uz
                psi = q2 - (q2 - np.arctan2(-fz, fx))   # = atan2(-fz, fx)
                cpsi, spsi = np.cos(psi), np.sin(psi)
                # g = h5 . h4,  h4 = [-cpsi cq1, -cpsi sq1, spsi]
                g = (h5[:, 0] * (-cpsi * cq1) + h5[:, 1] * (-cpsi * sq1)
                     + h5[:, 2] * spsi)
                g = np.where(reach, g, np.nan)
                out[(q1_branch, elbow)] = g
        return out

    def _phi_roots(self, p_j6, e1, e2, n_phi):
        """Bracket (vectorised scan) + bisect every g(phi)=0 per arm branch."""
        phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        step = 2.0 * np.pi / n_phi
        g_by_label = self._g_all_labels(phis, p_j6, e1, e2)
        roots = []
        for label, g in g_by_label.items():
            sign = g < 0.0
            cross = np.where((~np.isnan(g)) & (~np.isnan(np.roll(g, -1)))
                             & (sign != np.roll(sign, -1)))[0]
            for k in cross:
                phi = self._bisect(label, phis[k], phis[k] + step,
                                   p_j6, e1, e2)
                if phi is not None:
                    roots.append((label, phi))
        return roots

    def _bisect(self, label, lo, hi, p_j6, e1, e2, n_iter=14):
        glo = self._g(label, lo, p_j6, e1, e2)[0]
        if glo is None:
            return None
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            gm = self._g(label, mid, p_j6, e1, e2)[0]
            if gm is None:
                return None
            if (glo < 0.0) != (gm < 0.0):
                hi = mid
            else:
                lo = mid
                glo = gm
        return 0.5 * (lo + hi)

    # ---- closed-form arm: p_j5 -> (q1, q2, q3) for one branch label -----
    def _arm_branch(self, p_j5, label):
        q1_branch, elbow = label
        px, py, pz = float(p_j5[0]), float(p_j5[1]), float(p_j5[2])
        q1 = math.atan2(py, px) + (math.pi if q1_branch else 0.0)
        r = px * math.cos(q1) + py * math.sin(q1)
        h = pz - self.z0
        d2 = r * r + h * h
        l2, l4 = self.l2, self.l4
        if d2 > (l2 + l4) ** 2 or d2 < (l2 - l4) ** 2:
            return None
        cos_el = (d2 - l2 * l2 - l4 * l4) / (2.0 * l2 * l4)
        cos_el = -1.0 if cos_el < -1.0 else (1.0 if cos_el > 1.0 else cos_el)
        el = elbow * math.acos(cos_el)
        beta = math.atan2(l4 * math.sin(el), l2 + l4 * math.cos(el))
        q2 = math.atan2(r, h) - beta
        ux, uz = l2 * math.sin(q2), l2 * math.cos(q2)
        fx, fz = r - ux, h - uz
        q3 = q2 - math.atan2(-fz, fx)
        return q1, q2, q3

    # ---- wrist: R36 = Rx(-q4) Ry(-q5) Rx(-q6) (X-Y-X Euler) -------------
    def _wrist_xyx(self, r36):
        out = []
        cb = float(np.clip(r36[0, 0], -1.0, 1.0))
        for beta in (np.arccos(cb), -np.arccos(cb)):
            sb = np.sin(beta)
            if abs(sb) < 1e-9:
                # gimbal: fold q4+q6, pick q4 such that q6 = 0
                a_plus_c = np.arctan2(r36[2, 1], r36[1, 1])
                out.append((-a_plus_c, -beta, 0.0))
                continue
            a = np.arctan2(r36[1, 0] / sb, -r36[2, 0] / sb)
            c = np.arctan2(r36[0, 1] / sb, r36[0, 2] / sb)
            out.append((-a, -beta, -c))
        return out

    def _complete(self, label, phi, p_j6, e1, e2, r06):
        _, q123 = self._g(label, phi, p_j6, e1, e2)
        if q123 is None:
            return []
        q1, q2, q3 = q123
        r03 = self._rotz(q1) @ self._roty(q2) @ self._roty(-q3)
        r36 = r03.T @ r06
        out = []
        for q4, q5, q6 in self._wrist_xyx(r36):
            out.append(np.array([q1, q2, q3, q4, q5, q6], dtype=np.float32))
        return out

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
