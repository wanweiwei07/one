import math

import numpy as np

import one.utils.math as oum
import one.robots.base.kine.anaik as orbka
import one.robots.manipulators.denso.cvr038.ik.q4_resultant_np as ormdcrn


class CVR038PencilIK(orbka.AnaIKBase):
    """
    CVR038-specific semi-analytic IK.

    Geometry:
        h1 = Z, h2 = Y, h3 = Y, h4 = Z, h5 = Y, h6 = Z
        h1 x h2, h2 // h3, h5 x h6
        h3 and h4 have a small common-normal offset.

    q4 is solved by a resultant pencil that uses generated closed-form
    coefficients (q4_pencil_coeffs) in the base-Z canonical frame, so the pencil
    is built without any per-call symbolic/dict resultant work and is ~deg 12
    instead of ~deg 32 -- both faster and better conditioned.  The deflation
    drops the spurious q4 ~= +-pi (t -> inf) candidate, which the FK residual
    filter would reject anyway.  The runtime carries no sympy dependency; the
    coefficients are regenerated offline by gen_q4_pencil_coeffs.py.
    """

    def __init__(self, chain, joint_limits=None):
        super().__init__(chain, joint_limits=joint_limits)
        self.h1 = np.asarray(self.chain.axes[0], dtype=np.float32)
        self.h2 = np.asarray(self.chain.axes[1], dtype=np.float32)
        self.h3 = np.asarray(self.chain.axes[2], dtype=np.float32)
        self.h4 = np.asarray(self.chain.axes[3], dtype=np.float32)
        self.h5 = np.asarray(self.chain.axes[4], dtype=np.float32)
        self.h6 = np.asarray(self.chain.axes[5], dtype=np.float32)
        # fast-FK tables (CVR038 axes are cardinal Z/Y, so motion is Rz/Ry)
        self._zero_tfs = [np.asarray(jnt.zero_tf, dtype=np.float32)
                          for jnt in self.chain.jnts]
        self._axis_kind = []
        for h in self.chain.axes:
            h = np.asarray(h, dtype=np.float32)
            i = int(np.argmax(np.abs(h)))
            self._axis_kind.append(('xyz'[i], float(np.sign(h[i]))))
        self._setup_equivalent_geometry()

    def ik_all(
        self,
        tgt_tf_in_root,
        residual_tol=1e-4,
        rot_weight=0.2,
        **kwargs,
    ):
        tgt_tf = np.asarray(tgt_tf_in_root, dtype=np.float32).copy()
        q4s = self._pencil_q4_roots(tgt_tf)
        return self._ik_all_from_q4_roots(tgt_tf, q4s, residual_tol, rot_weight)

    def _ik_all_from_q4_roots(self, tgt_tf, q4s, residual_tol, rot_weight):
        all_qs = []
        for q4 in q4s:
            all_qs.extend(self._fixed_q4_candidates(tgt_tf, q4))
        all_qs = self._normalize_angles(all_qs)
        all_qs = self._filter_limits(all_qs)
        all_qs = self._filter_fk_residual(all_qs, tgt_tf, residual_tol, rot_weight)
        return self._unique(all_qs)

    def _target_p16_v(self, tgt_tf):
        R06 = tgt_tf[:3, :3] @ self.chain.tfs[-1][:3, :3].T
        p16 = tgt_tf[:3, 3] - R06 @ self.p6t - self.p01
        v = R06 @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return p16, v

    def _pencil_q4_roots(self, tgt_tf):
        p16, v = self._target_p16_v(tgt_tf)
        return ormdcrn.q4_roots_pencil_fast(
            float(p16[0]), float(p16[1]), float(p16[2]),
            float(v[0]), float(v[1]), float(v[2]),
            tol=1e-6,
        )

    def _setup_equivalent_geometry(self):
        origins = [np.asarray(o, dtype=np.float32) for o in self.chain.origins]
        axes = [oum.unit_vec(np.asarray(h, dtype=np.float32), return_length=False)
                for h in self.chain.axes]
        x12_a, x12_b, d12 = self._closest_points_on_lines(
            origins[0], axes[0], origins[1], axes[1])
        x34_on_h3, x34_on_h4, d34 = self._closest_points_on_lines(
            origins[2], axes[2], origins[3], axes[3])
        x56_a, x56_b, d56 = self._closest_points_on_lines(
            origins[4], axes[4], origins[5], axes[5])
        self.intersection_errors = (d12, d34, d56)

        x12 = ((x12_a + x12_b) * 0.5).astype(np.float32)
        x56 = ((x56_a + x56_b) * 0.5).astype(np.float32)
        self.p01 = x12
        self.p12 = np.zeros(3, dtype=np.float32)
        self.p23 = (x34_on_h3 - x12).astype(np.float32)
        self.p34 = (x34_on_h4 - x34_on_h3).astype(np.float32)
        self.p45 = (x56 - x34_on_h4).astype(np.float32)
        self.p56 = np.zeros(3, dtype=np.float32)
        self.p6t = (origins[5] - x56).astype(np.float32)

    def _closest_points_on_lines(self, p1, h1, p2, h2):
        h1 = oum.unit_vec(h1, return_length=False)
        h2 = oum.unit_vec(h2, return_length=False)
        a = np.array([[np.dot(h1, h1), -np.dot(h1, h2)],
                      [np.dot(h1, h2), -np.dot(h2, h2)]], dtype=np.float32)
        b = np.array([np.dot(h1, p2 - p1), np.dot(h2, p2 - p1)], dtype=np.float32)
        if abs(float(np.linalg.det(a))) < 1e-8:
            s = float(np.dot(h1, p2 - p1))
            t = 0.0
        else:
            s, t = np.linalg.solve(a, b)
        c1 = p1 + h1 * float(s)
        c2 = p2 + h2 * float(t)
        return c1.astype(np.float32), c2.astype(np.float32), float(np.linalg.norm(c1 - c2))

    def _fixed_q4_candidates(self, tgt_tf, q4):
        all_qs = []
        for q1, q2, q3, R06 in self._fixed_q4_q123_candidates(tgt_tf, q4):
            R01 = self._rotz(q1)
            R12 = self._roty(q2)
            R23 = self._roty(q3)
            R03 = R01 @ R12 @ R23
            R36 = R03.T @ R06
            R43 = self._rotz(-q4)
            R46 = R43 @ R36
            q5 = float(np.arctan2(R46[0, 2], R46[2, 2]))
            R54 = self._roty(-q5)
            R56 = R54 @ R46
            q6 = float(np.arctan2(R56[1, 0], R56[0, 0]))
            all_qs.append(np.array([q1, q2, q3, q4, q5, q6], dtype=np.float32))
        return all_qs

    @staticmethod
    def _cos_sin_solve(a, b, c):
        # all theta with a*cos(theta) + b*sin(theta) = c
        r = math.hypot(a, b)
        if r < 1e-12:
            return []
        ratio = c / r
        if ratio > 1.0:
            if ratio < 1.0 + 1e-9:
                ratio = 1.0
            else:
                return []
        elif ratio < -1.0:
            if ratio > -1.0 - 1e-9:
                ratio = -1.0
            else:
                return []
        phi0 = math.atan2(b, a)
        delta = math.acos(ratio)
        if delta < 1e-9:
            return [phi0]
        return [phi0 + delta, phi0 - delta]

    def _fixed_q4_q123_candidates(self, tgt_tf, q4):
        p06 = tgt_tf[:3, 3]
        R06 = tgt_tf[:3, :3] @ self.chain.tfs[-1][:3, :3].T
        p16 = p06 - R06 @ self.p6t - self.p01
        R34 = self._rotz(q4)
        p36 = self.p34 + R34 @ self.p45

        p16x, p16y, p16z = float(p16[0]), float(p16[1]), float(p16[2])
        p36x, p36y, p36z = float(p36[0]), float(p36[1]), float(p36[2])
        a23, b23, c23 = float(self.p23[0]), float(self.p23[1]), float(self.p23[2])
        p16n2 = p16x * p16x + p16y * p16y + p16z * p16z
        p36n2 = p36x * p36x + p36y * p36y + p36z * p36z
        p23n2 = a23 * a23 + b23 * b23 + c23 * c23

        # sp3: ||p23 + Ry(q3) p36|| = ||p16||  ->  A cos q3 + B sin q3 = C
        A3 = a23 * p36x + c23 * p36z
        B3 = a23 * p36z - c23 * p36x
        C3 = 0.5 * (p16n2 - p23n2 - p36n2) - b23 * p36y

        out = []
        for q3 in self._cos_sin_solve(A3, B3, C3):
            c3, s3 = math.cos(q3), math.sin(q3)
            # p26 = p23 + Ry(q3) p36
            p26x = a23 + p36x * c3 + p36z * s3
            p26y = b23 + p36y
            p26z = c23 - p36x * s3 + p36z * c3
            # sp2: p16 = Rz(q1) Ry(q2) p26
            #   q2 from p26z cos q2 - p26x sin q2 = p16z
            for q2 in self._cos_sin_solve(p26z, -p26x, p16z):
                c2, s2 = math.cos(q2), math.sin(q2)
                vx = p26x * c2 + p26z * s2
                vy = p26y
                q1 = math.atan2(p16y, p16x) - math.atan2(vy, vx)
                out.append((q1, q2, float(q3), R06))
        return out

    def _rotz(self, angle):
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.array([[c, -s, 0.0],
                         [s, c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    def _roty(self, angle):
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.array([[c, 0.0, s],
                         [0.0, 1.0, 0.0],
                         [-s, 0.0, c]], dtype=np.float32)

    def _fk_error(self, qs, tgt_tf, rot_weight):
        cur_tf = self._fk_tf(qs)
        pos_err = tgt_tf[:3, 3] - cur_tf[:3, 3]
        # small-angle rotation error from the antisymmetric part (~sin theta ~
        # theta); stable near 0, unlike arccos, and scipy-free
        rel = cur_tf[:3, :3].T @ tgt_tf[:3, :3]
        vx = rel[2, 1] - rel[1, 2]
        vy = rel[0, 2] - rel[2, 0]
        vz = rel[1, 0] - rel[0, 1]
        theta = 0.5 * float(np.sqrt(vx * vx + vy * vy + vz * vz))
        return float(np.sqrt(float(pos_err @ pos_err) + (rot_weight * theta) ** 2))

    def _rotx(self, angle):
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, c, -s],
                         [0.0, s, c]], dtype=np.float32)

    def _fk_tf(self, qs):
        rmap = {'x': self._rotx, 'y': self._roty, 'z': self._rotz}
        tf = np.eye(4, dtype=np.float32)
        m = np.eye(4, dtype=np.float32)
        for ztf, (char, sign), q in zip(self._zero_tfs, self._axis_kind, qs):
            m[:3, :3] = rmap[char](sign * float(q))
            tf = tf @ ztf @ m
        return tf

    def _filter_fk_residual(self, qs_list, tgt_tf, residual_tol, rot_weight):
        out = []
        for qs in qs_list:
            err = self._fk_error(qs, tgt_tf, rot_weight)
            if err <= residual_tol:
                out.append(qs)
        return out

    def _normalize_angles(self, qs_list):
        out = []
        for qs in qs_list:
            out.append(((qs + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32))
        return out

    def _angles(self, value):
        return np.asarray(value, dtype=np.float32).reshape(-1)


# Backward-compatible alias.
CVR038GeoIK = CVR038PencilIK
