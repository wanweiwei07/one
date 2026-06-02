import numpy as np

import one.utils.math as oum
import one.robots.base.kine.anaik as orbka
import one.robots.base.kine.ikgeo.sp2_lib as orbkisp2
import one.robots.base.kine.ikgeo.sp3_lib as orbkisp3
import one.robots.manipulators.denso.cvr038.ik.derive_q4_poly as ormdcip
import one.robots.manipulators.denso.cvr038.ik.q4_resultant_np as ormdcrn


class CVR038PencilIK(orbka.AnaIKBase):
    """
    CVR038-specific semi-analytic IK.

    Geometry:
        h1 = Z, h2 = Y, h3 = Y, h4 = Z, h5 = Y, h6 = Z
        h1 x h2, h2 // h3, h5 x h6
        h3 and h4 have a small common-normal offset.

    q4 is solved by a numeric resultant pencil ('pencil_roots', default) or by
    the offline sympy resultant ('sympy_roots').
    """

    def __init__(self, chain, joint_limits=None):
        super().__init__(chain, joint_limits=joint_limits)
        self.h1 = np.asarray(self.chain.axes[0], dtype=np.float32)
        self.h2 = np.asarray(self.chain.axes[1], dtype=np.float32)
        self.h3 = np.asarray(self.chain.axes[2], dtype=np.float32)
        self.h4 = np.asarray(self.chain.axes[3], dtype=np.float32)
        self.h5 = np.asarray(self.chain.axes[4], dtype=np.float32)
        self.h6 = np.asarray(self.chain.axes[5], dtype=np.float32)
        self._setup_equivalent_geometry()

    def ik_all(
        self,
        tgt_tf_in_root,
        q4_method='pencil_roots',
        residual_tol=1e-4,
        rot_weight=0.2,
        **kwargs,
    ):
        tgt_tf = np.asarray(tgt_tf_in_root, dtype=np.float32).copy()
        if q4_method == 'sympy_roots':
            return self._ik_all_from_sympy_q4_roots(tgt_tf, residual_tol, rot_weight)
        return self._ik_all_from_pencil_q4_roots(tgt_tf, residual_tol, rot_weight)

    def _ik_all_from_sympy_q4_roots(self, tgt_tf, residual_tol, rot_weight):
        q4s = self._sympy_q4_roots(tgt_tf)
        return self._ik_all_from_q4_roots(tgt_tf, q4s, residual_tol, rot_weight)

    def _ik_all_from_pencil_q4_roots(self, tgt_tf, residual_tol, rot_weight):
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

    def _sympy_q4_roots(self, tgt_tf):
        p16, v = self._target_p16_v(tgt_tf)
        return ormdcip.q4_real_roots(
            float(p16[0]), float(p16[1]), float(p16[2]),
            float(v[0]), float(v[1]), float(v[2]),
        )

    def _pencil_q4_roots(self, tgt_tf):
        p16, v = self._target_p16_v(tgt_tf)
        return ormdcrn.q4_roots_pencil(
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

    def _fixed_q4_q123_candidates(self, tgt_tf, q4):
        p06 = tgt_tf[:3, 3]
        R06 = tgt_tf[:3, :3] @ self.chain.tfs[-1][:3, :3].T
        p16 = p06 - R06 @ self.p6t - self.p01
        R34 = self._rotz(q4)
        p36 = self.p34 + R34 @ self.p45

        q3s, _ = orbkisp3.sp3_run(p36, -self.p23, self.h3, float(np.linalg.norm(p16)))
        q3s = self._angles(q3s)
        out = []
        for q3 in q3s:
            R23 = self._roty(q3)
            p26 = self.p23 + R23 @ p36
            q1s, q2s, _ = orbkisp2.sp2_run(p16, p26, -self.h1, self.h2)
            for q1, q2 in zip(self._angles(q1s), self._angles(q2s)):
                out.append((float(q1), float(q2), float(q3), R06))
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
        rot_err = oum.delta_rotvec_between_rotmats(cur_tf[:3, :3], tgt_tf[:3, :3])
        return float(np.linalg.norm(np.concatenate([pos_err, rot_weight * rot_err])))

    def _fk_tf(self, qs):
        tf = np.eye(4, dtype=np.float32)
        for jnt, q in zip(self.chain.jnts, qs):
            tf = tf @ jnt.zero_tf @ jnt.motion_tf(float(q))
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
