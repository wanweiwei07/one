import numpy as np

import one.utils.math as oum


class AnaIKBase:
    """
    Analytic IK base class (returns all solutions).
    ik(...) returns the closest solution to ref_qs (if provided).
    """

    def __init__(self, chain, joint_limits=None):
        self.chain = chain
        if joint_limits is None:
            self.joint_limits = (chain.lmt_lo, chain.lmt_up)
        else:
            self.joint_limits = joint_limits

    def ik(self, root_rotmat, root_pos, tgt_rotmat, tgt_pos, ref_qs=None, **kwargs):
        root_tf = oum.tf_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_tf_in_root = np.linalg.inv(root_tf) @ tgt_tf
        sols = self.ik_all(tgt_tf_in_root, **kwargs)
        if not sols:
            return []
        if ref_qs is None:
            return [sols[0]]
        d2 = [np.linalg.norm(q - ref_qs) for q in sols]
        best = sols[int(np.argmin(d2))]
        return [best]

    def ik_all(self, tgt_tf_in_root, **kwargs):
        raise NotImplementedError

    def get_rotmat_from_fk(self, qs, k):
        tf = np.eye(4, dtype=np.float32)
        for i in range(k):
            tf = tf @ self.chain.jnts[i].zero_tf @ self.chain.jnts[i].motion_tf(qs[i])
        return tf[:3, :3]

    def _filter_limits(self, qs_list):
        if self.joint_limits is None:
            return qs_list
        low, high = self.joint_limits
        out = []
        for q in qs_list:
            if np.all(q >= low) and np.all(q <= high):
                out.append(q)
        return out

    def _unique(self, qs_list, tol=1e-4):
        uniq = []
        for q in qs_list:
            if all(np.linalg.norm(q - u) > tol for u in uniq):
                uniq.append(q)
        return uniq


class S456X12(AnaIKBase):
    def __init__(self, chain, joint_limits=None):
        super().__init__(chain, joint_limits=joint_limits)
        self.o1 = np.asarray(self.chain.origins[0], dtype=np.float32)
        self.o2 = np.asarray(self.chain.origins[1], dtype=np.float32)
        self.o3 = np.asarray(self.chain.origins[2], dtype=np.float32)
        o4 = np.asarray(self.chain.origins[3], dtype=np.float32)
        o5 = np.asarray(self.chain.origins[4], dtype=np.float32)
        o6 = np.asarray(self.chain.origins[5], dtype=np.float32)
        self.a1 = oum.unit_vec(self.chain.axes[0], return_length=False)
        self.a2 = oum.unit_vec(self.chain.axes[1], return_length=False)
        self.a3 = oum.unit_vec(self.chain.axes[2], return_length=False)
        self.a4 = oum.unit_vec(self.chain.axes[3], return_length=False)
        self.a5 = oum.unit_vec(self.chain.axes[4], return_length=False)
        self.a6 = oum.unit_vec(self.chain.axes[5], return_length=False)
        # ow = origin of wrist center (intersection of 456),
        # compute it by least squares
        A = np.zeros((3, 3), dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        for a, o in [(self.a4, o4), (self.a5, o5), (self.a6, o6)]:
            a = oum.unit_vec(a, return_length=False)
            i_minus_aat = np.eye(3, dtype=np.float32) - np.outer(a, a)
            A += i_minus_aat
            b += i_minus_aat @ o
        self.ow = np.linalg.solve(A, b)
        # paper-style p vectors (from o/a cache)
        # p_ij denotes displacement from origin i to origin j in zero config.
        self.p12 = (self.o2 - self.o1).astype(np.float32)
        self.p23 = (self.o3 - self.o2).astype(np.float32)
        self.p34 = (o4 - self.o3).astype(np.float32)
        self.p45 = (o5 - o4).astype(np.float32)
        self.p56 = (o6 - o5).astype(np.float32)
        # offsets related to spherical wrist center
        self.p3w = (self.ow - self.o3).astype(np.float32)
        self.pw6 = (o6 - self.ow).astype(np.float32)
        # subtract wrist offset
        self.ow_6 = o6 - self.ow
        # p01, p12, p23 lengths
        self.l2 = float(np.linalg.norm(self.o3 - self.o2))
        self.l3 = float(np.linalg.norm(self.ow - self.o3))
        rotmat0_3_zero = self.get_rotmat_from_fk([0, 0, 0], k=3)
        rotmat0_6_zero = self.get_rotmat_from_fk([0, 0, 0, 0, 0, 0], k=6)
        self.wrist_offset = rotmat0_3_zero.T @ rotmat0_6_zero

    def ik_all(self, tgt_tf_in_root, **kwargs):
        tf0_6 = np.asarray(tgt_tf_in_root, dtype=np.float32)
        rotmat0_6 = tf0_6[:3, :3]
        pos0_6 = tf0_6[:3, 3]
        pw = pos0_6 - rotmat0_6 @ self.ow_6
        q123_list = self._solve_first3(pw)
        sols = []
        for q1, q2, q3 in q123_list:
            rotmat0_3 = self.get_rotmat_from_fk([q1, q2, q3], k=3)
            rotmat3_6 = rotmat0_3.T @ rotmat0_6
            for q4, q5, q6 in self._solve_wrist_ZXZ(rotmat3_6):
                sols.append(np.array([q1, q2, q3, q4, q5, q6], dtype=np.float32))
        sols = self._filter_limits(sols)
        sols = self._unique(sols)
        return sols

    def _solve_first3(self, pw):
        a1 = oum.unit_vec(self.a1, return_length=False)
        l2, l3 = self.l2, self.l3
        v = pw - self.o2
        d_total = np.linalg.norm(v)
        if d_total > l2 + l3 + 1e-6 or d_total < abs(l2 - l3) - 1e-6:
            return []
        v_projected = v - np.dot(v, a1) * a1
        r_xy = np.linalg.norm(v_projected)
        q1_solutions = []
        if r_xy < 1e-9:
            q1_solutions = [0.0]
        else:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(a1[2]) > 0.9:
                x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                x_ref = oum.unit_vec(np.cross(a1, z_axis), return_length=False)
            y_ref = np.cross(a1, x_ref)
            azimuth = np.arctan2(np.dot(v_projected, y_ref), np.dot(v_projected, x_ref))
            q1_solutions = [
                float(oum.wrap_to_pi(azimuth - np.pi / 2)),
                float(oum.wrap_to_pi(azimuth + np.pi / 2)),
            ]

        sols = []
        for q1 in q1_solutions:
            R_j1 = oum.rotmat_from_axangle(a1, q1)
            v_j1 = R_j1 @ v
            a2_j1 = R_j1 @ self.a2
            a2_unit = oum.unit_vec(a2_j1, return_length=False)
            v_parallel = np.dot(v_j1, a2_unit)
            v_perp_vec = v_j1 - v_parallel * a2_unit
            v_perp = np.linalg.norm(v_perp_vec)
            d_2d = v_perp
            c3 = oum.clamp((d_2d**2 - l2**2 - l3**2) / (2 * l2 * l3), lo=-1.0, hi=1.0)
            s3_abs = np.sqrt(max(0.0, 1.0 - c3**2))
            q3_sols_for_this_q1 = []
            for s3_sign in (+1, -1):
                s3 = s3_sign * s3_abs
                if abs(s3) < 1e-9 and q3_sols_for_this_q1:
                    continue
                q3 = float(np.arctan2(s3, c3))
                alpha = np.arctan2(l3 * s3, l2 + l3 * c3)
                z_world = np.array([0, 0, 1], dtype=np.float32)
                arm_zero_in_j1 = R_j1 @ z_world
                arm_zero_perp = (
                    arm_zero_in_j1 - np.dot(arm_zero_in_j1, a2_unit) * a2_unit
                )
                arm_zero_perp_unit = oum.unit_vec(arm_zero_perp, return_length=False)
                if v_perp > 1e-9:
                    v_perp_unit = oum.unit_vec(v_perp_vec, return_length=False)
                else:
                    v_perp_unit = arm_zero_perp_unit
                cos_beta = np.clip(np.dot(arm_zero_perp_unit, v_perp_unit), -1, 1)
                sin_beta_cross = np.cross(arm_zero_perp_unit, v_perp_unit)
                cross_dot_a2 = np.dot(sin_beta_cross, a2_unit)
                if abs(cross_dot_a2) > 1e-9:
                    sin_beta = np.linalg.norm(sin_beta_cross) * np.sign(cross_dot_a2)
                else:
                    sin_beta = 0.0
                beta = np.arctan2(sin_beta, cos_beta)
                q2 = float(beta - alpha)
                q3_sols_for_this_q1.append(q3)
                sols.append((q1, oum.wrap_to_pi(q2), oum.wrap_to_pi(q3)))
        return sols

    def _solve_wrist_ZXZ(self, rotmat3_6):
        R_pure = self.wrist_offset.T @ rotmat3_6
        c5 = oum.clamp(R_pure[2, 2], lo=-1.0, hi=1.0)
        q5a = float(np.arccos(c5))
        q5b = float(-q5a)
        sols = []
        for q5 in (q5a, q5b):
            s5 = np.sin(q5)
            if abs(s5) < 1e-8:
                q4 = 0.0
                if c5 > 0:
                    q6 = float(np.arctan2(-R_pure[0, 1], R_pure[0, 0]))
                else:
                    q6 = float(np.arctan2(R_pure[0, 1], -R_pure[0, 0]))
            else:
                q4 = float(np.arctan2(R_pure[0, 2], -R_pure[1, 2]))
                q6 = float(np.arctan2(R_pure[2, 0], R_pure[2, 1]))
            sols.append((oum.wrap_to_pi(q4), oum.wrap_to_pi(q5), oum.wrap_to_pi(q6)))
        return sols
