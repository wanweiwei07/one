import numpy as np

import one.utils.math as oum
import one.robots.base.kine.ikgeo.sp1_lib as orbkisp1
import one.robots.base.kine.ikgeo.sp2_lib as orbkisp2
import one.robots.base.kine.ikgeo.sp3_lib as orbkisp3
import one.robots.base.kine.ikgeo.sp4_lib as orbkisp4


class AnaIKBase:
    """
    Analytic IK base class (returns all solutions).
    ik(...) returns solutions in root frame.
    """

    def __init__(self, chain, joint_limits=None):
        self.chain = chain
        if joint_limits is None:
            self.joint_limits = (chain.lmt_lo, chain.lmt_up)
        else:
            self.joint_limits = joint_limits

    def ik(
        self,
        root_rotmat,
        root_pos,
        tgt_rotmat,
        tgt_pos,
        max_solutions=8,
        ref_qs=None,
        **kwargs,
    ):
        root_tf = oum.tf_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_tf_in_root = np.linalg.inv(root_tf) @ tgt_tf
        sols = self.ik_all(tgt_tf_in_root, **kwargs)
        if not sols:
            return []
        if ref_qs is not None:
            ref_qs = np.asarray(ref_qs, dtype=np.float32)
            order = np.argsort([np.linalg.norm(q - ref_qs) for q in sols])
            sols = [sols[i] for i in order]
        if max_solutions is not None:
            sols = sols[:max_solutions]
        return sols

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
        o1 = np.asarray(self.chain.origins[0], dtype=np.float32)
        o2 = np.asarray(self.chain.origins[1], dtype=np.float32)
        o3 = np.asarray(self.chain.origins[2], dtype=np.float32)
        o4 = np.asarray(self.chain.origins[3], dtype=np.float32)
        o5 = np.asarray(self.chain.origins[4], dtype=np.float32)
        o6 = np.asarray(self.chain.origins[5], dtype=np.float32)
        a4 = oum.unit_vec(self.chain.axes[3], return_length=False)
        a5 = oum.unit_vec(self.chain.axes[4], return_length=False)
        a6 = oum.unit_vec(self.chain.axes[5], return_length=False)
        # ow = origin of wrist center (intersection of 456),
        # compute it by least squares
        A = np.zeros((3, 3), dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        for a, o in [(a4, o4), (a5, o5), (a6, o6)]:
            a = oum.unit_vec(a, return_length=False)
            i_minus_aat = np.eye(3, dtype=np.float32) - np.outer(a, a)
            A += i_minus_aat
            b += i_minus_aat @ o
        ow = np.linalg.solve(A, b)
        # paper-style vectors
        self.p01 = o1.astype(np.float32)
        self.p12 = (o2 - o1).astype(np.float32)  # 0,0,0
        self.p23 = (o3 - o2).astype(np.float32)
        self.p34 = (ow - o3).astype(np.float32)
        self.p45 = np.zeros(3, dtype=np.float32)
        self.p56 = np.zeros(3, dtype=np.float32)
        self.p6t = (o6 - ow).astype(np.float32)
        self.h1 = self.chain.axes[0]
        self.h2 = self.chain.axes[1]
        self.h3 = self.chain.axes[2]
        self.h4 = self.chain.axes[3]
        self.h5 = self.chain.axes[4]
        self.h6 = self.chain.axes[5]

    def ik_all(self, tgt_tf_in_root, **kwargs):
        # compute q3 by sp3
        # R23p34 + p23 = p16
        # p16 = p0t - R06 p6t - p01
        p06 = tgt_tf_in_root[:3, 3]
        R06 = tgt_tf_in_root[:3, :3] @ self.chain.tfs[-1][:3, :3].T
        p16 = p06 - R06 @ self.p6t - self.p01
        p1 = self.p34
        p2 = -self.p23
        k = self.h3
        d = float(np.linalg.norm(p16))
        q3s, is_ls = orbkisp3.sp3_run(p1, p2, k, d)
        if is_ls:
            return []
        if len(q3s) == 0:
            return []
        all_qs = []
        for q3 in q3s:
            R23 = oum.rotmat_from_axangle(self.h3, q3)
            p23plusR23p34 = self.p23 + R23 @ self.p34
            # compute q1, q2 by sp2
            # R10p16 = R12(p23+R23p34)
            p1 = p16
            p2 = p23plusR23p34
            k1 = -self.h1
            k2 = self.h2
            q1s, q2s, is_sl = orbkisp2.sp2_run(p1, p2, k1, k2)
            if is_sl:
                continue
            else:
                q1s = np.asarray(q1s).reshape(-1)
                q2s = np.asarray(q2s).reshape(-1)
                pairs = list(zip(q1s, q2s))
            for q1, q2 in pairs:
                R01 = oum.rotmat_from_axangle(self.h1, q1)
                R12 = oum.rotmat_from_axangle(self.h2, q2)
                R03 = R01 @ R12 @ R23
                R36 = R03.T @ R06
                # compute q4, q5 by sp2
                # R43R36h6 = R45h6
                p1 = R36 @ self.h6
                k1 = -self.h4
                p2 = self.h6
                k2 = self.h5
                q4s, q5s, is_sl = orbkisp2.sp2_run(p1, p2, k1, k2)
                if is_sl:
                    continue
                else:
                    q4s = np.asarray(q4s).reshape(-1)
                    q5s = np.asarray(q5s).reshape(-1)
                    pairs = list(zip(q4s, q5s))
                for q4, q5 in pairs:
                    R43 = oum.rotmat_from_axangle(-self.h4, q4)
                    R54 = oum.rotmat_from_axangle(-self.h5, q5)
                    p1 = oum.orth_vec(self.h6)
                    k = self.h6
                    p2 = R54 @ R43 @ R36 @ p1
                    # compute q6 by sp1
                    q6, is_ls = orbkisp1.sp1_run(p1, p2, k)
                    if is_ls:
                        continue
                    qs = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float32)
                    all_qs.append(qs)
        return all_qs


class P234X56(AnaIKBase):
    def __init__(self, chain, joint_limits=None):
        super().__init__(chain, joint_limits=joint_limits)
        o1 = np.asarray(self.chain.origins[0], dtype=np.float32)
        o2 = np.asarray(self.chain.origins[1], dtype=np.float32)
        o3 = np.asarray(self.chain.origins[2], dtype=np.float32)
        o4 = np.asarray(self.chain.origins[3], dtype=np.float32)
        o5 = np.asarray(self.chain.origins[4], dtype=np.float32)
        o6 = np.asarray(self.chain.origins[5], dtype=np.float32)
        a5 = oum.unit_vec(self.chain.axes[4], return_length=False)
        a6 = oum.unit_vec(self.chain.axes[5], return_length=False)
        # ow = origin of wrist center (intersection of 56)
        A = np.zeros((3, 3), dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        for a, o in [(a5, o5), (a6, o6)]:
            a = oum.unit_vec(a, return_length=False)
            i_minus_aat = np.eye(3, dtype=np.float32) - np.outer(a, a)
            A += i_minus_aat
            b += i_minus_aat @ o
        o56 = np.linalg.solve(A, b)
        # paper-style vectors
        self.p01 = o1.astype(np.float32)
        self.p12 = (o2 - o1).astype(np.float32)  # 0,0,0
        self.p23 = (o3 - o2).astype(np.float32)
        self.p34 = (o4 - o3).astype(np.float32)
        self.p45 = (o56 - o4).astype(np.float32)
        self.p56 = np.zeros(3, dtype=np.float32)
        self.p6t = (o6 - o56).astype(np.float32)
        self.h1 = self.chain.axes[0]
        self.h2 = self.chain.axes[1]
        self.h3 = self.chain.axes[2]
        self.h4 = self.chain.axes[3]
        self.h5 = self.chain.axes[4]
        self.h6 = self.chain.axes[5]

    def ik_all(self, tgt_tf_in_root, **kwargs):
        # compute q1 by sp4
        # h2.T R10 p16 = h2.T(p12+p23+p34+p45)
        # p16 = p0t - R06 p6t - p01
        p06 = tgt_tf_in_root[:3, 3]
        R06 = tgt_tf_in_root[:3, :3] @ self.chain.tfs[-1][:3, :3].T
        p16 = p06 - R06 @ self.p6t - self.p01
        h = self.h2
        k = -self.h1
        p = p16
        d = self.h2.T @ (self.p12 + self.p23 + self.p34 + self.p45)
        q1s, is_ls = orbkisp4.sp4_run(p, k, h, d)
        if is_ls:
            return []
        if len(q1s) == 0:
            return []
        all_qs = []
        for q1 in q1s:
            # compute q5 by sp4
            # h2.T R10 R06 h6 - h2.T R45 h6 = 0
            R10 = oum.rotmat_from_axangle(-self.h1, q1)
            h = self.h2
            k = self.h5
            p = self.h6
            d = self.h2.T @ (R10 @ R06 @ self.h6)
            q5s, is_ls = orbkisp4.sp4_run(p, k, h, d)
            if is_ls:
                continue
            if len(q5s) == 0:
                continue
            for q5 in q5s:
                # compute q2+q3+q4 by sp1
                # R14 R45 h6 = R10 R06 h6
                R45 = oum.rotmat_from_axangle(self.h5, q5)
                k = self.h2  # assume all parallel axes have same direction as h2
                p1 = R45 @ self.h6
                p2 = R10 @ R06 @ self.h6
                q234, is_ls = orbkisp1.sp1_run(p1, p2, k)
                if is_ls:
                    continue
                R14 = oum.rotmat_from_axangle(self.h2, q234)
                # compute q6 by sp1
                # R65 R54 h2 = R06.T R01 h2
                R01 = R10.T
                k = -self.h6
                p1 = R45.T @ self.h2
                p2 = R06.T @ R01 @ self.h2
                q6, is_ls = orbkisp1.sp1_run(p1, p2, k)
                if is_ls:
                    continue
                # compute q3 by sp3
                # R23 p34 + p23 = R10 p16 - p12 - R14 p45 - R15 p56
                rhs = R10 @ p16 - self.p12 - R14 @ self.p45
                d = np.linalg.norm(rhs)
                p1 = self.p34
                p2 = -self.p23
                k = self.h3
                q3s, is_ls = orbkisp3.sp3_run(p1, p2, k, d)
                if is_ls:
                    continue
                if len(q3s) == 0:
                    continue
                for q3 in q3s:
                    R23 = oum.rotmat_from_axangle(self.h3, q3)
                    # compute q2 by sp1
                    # R12 (p23 + R23 p34) = R10 p16 - p12 - R14 p45
                    p2 = rhs
                    p1 = self.p23 + R23 @ self.p34
                    k = self.h2
                    q2, is_ls = orbkisp1.sp1_run(p1, p2, k)
                    if is_ls:
                        continue
                    # compute q4 by subtraction
                    if self.h4.T @ self.h2 > 0:
                        q4 = q234 - q2 - q3
                    else:
                        q4 = q234 - q2 + q3
                    qs = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float32)
                    all_qs.append(qs)
        return all_qs
