import numpy as np
import one.utils.math as oum


class AnaIKBase:
    """
    Analytic IK base class (returns all solutions).
    ik(...) returns the closest solution to qs_active_init (if provided).
    """

    def __init__(self, chain, joint_limits=None):
        self.chain = chain
        self.jnts = chain.joints
        self.joint_limits = joint_limits  # optional (low, high)
        # cache zero-pose joint frames in root
        self.T0J0 = self._wd_jtf_0s()

    def _wd_jtf_0s(self):
        wd_tf_0s = []
        tmp_tf = np.eye(4, dtype=np.float32)
        for k in range(len(self.jnts)):
            tmp_tf = tmp_tf @ self.jnts[k].origin_tfmat
            wd_tf_0s.append(tmp_tf.copy())
        return wd_tf_0s

    def ik(self, root_rotmat, root_pos, tgt_rotmat, tgt_pos,
           qs_active_init=None, max_iter=50, **kwargs):
        """
        Return a single closest solution (list with 0 or 1 element).
        """
        root_tf = oum.tf_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        T0TCP = np.linalg.inv(root_tf) @ tgt_tf
        sols = self.ik_all(T0TCP, qs_active_init=qs_active_init, **kwargs)
        if not sols:
            return []
        if qs_active_init is None:
            return [sols[0]]
        d2 = [np.linalg.norm(q - qs_active_init) for q in sols]
        best = sols[int(np.argmin(d2))]
        return [best]

    def ik_all(self, T0TCP, qs_active_init=None, **kwargs):
        """
        Return all analytic solutions.
        Override in subclasses.
        """
        raise NotImplementedError

    def get_R0k_from_fk(self, qs, k):
        """
        Minimal FK for serial chain. Returns rotation R0k.
        """
        T = np.eye(4, dtype=np.float32)
        for i in range(k):
            T = T @ self.jnts[i].tf_0 @ self.jnts[i].motion_tf(qs[i])
        return T[:3, :3]

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


class AnaSphericalWrist6DOF(AnaIKBase):
    """
    Example: 6-DOF arm with spherical wrist.
    You must implement:
      - _solve_first3(pw)
      - _solve_wrist_ZYZ(R36) or your wrist convention
      - get_R0k_from_fk(qs, k)
    """

    def __init__(self, chain, joint_limits=None):
        super().__init__(chain, joint_limits=joint_limits)
        # cache zero-pose joint origins/axes
        # first 3 joints + wrist center
        self.o1 = self.T0J0[0][:3, 3]
        self.o2 = self.T0J0[1][:3, 3]
        self.o3 = self.T0J0[2][:3, 3]
        self.ow = self.T0J0[3][:3, 3]  # wrist center
        # last 3 joint axes
        self.a1 = self.T0J0[0][:3, :3] @ self.jnts[0].axis
        self.a2 = self.T0J0[1][:3, :3] @ self.jnts[1].axis
        self.a3 = self.T0J0[2][:3, :3] @ self.jnts[2].axis
        self.a4 = self.T0J0[3][:3, :3] @ self.jnts[3].axis
        self.a5 = self.T0J0[4][:3, :3] @ self.jnts[4].axis
        self.a6 = self.T0J0[5][:3, :3] @ self.jnts[5].axis
        # link lengths
        self.L2 = float(np.linalg.norm(self.o3 - self.o2))
        self.L3 = float(np.linalg.norm(self.ow - self.o3))

    def ik_all(self, T0TCP, qs_active_init=None, **kwargs):
        T0TCP = np.asarray(T0TCP, dtype=np.float32)
        T0_6 = T0TCP @ np.linalg.inv(self.TnTCP0)
        R06 = T0_6[:3, :3]
        p06 = T0_6[:3, 3]

        pw = p06  # wrist center assumption
        q123_list = self._solve_first3(pw)
        sols = []

        for (q1, q2, q3) in q123_list:
            R03 = self.get_R0k_from_fk([q1, q2, q3], k=3)
            R36 = R03.T @ R06
            for (q4, q5, q6) in self._solve_wrist_ZYZ(R36):
                sols.append(np.array([q1, q2, q3, q4, q5, q6], dtype=np.float32))

        sols = self._filter_limits(sols)
        sols = self._unique(sols)
        return sols

    def _solve_first3(self, pw):
        """
        Solve q1,q2,q3 from wrist center position pw (root frame).
        Uses:
          - q1 about axis1 to align wrist to shoulder plane
          - q2,q3 from planar 2R in plane perpendicular to axis2
        """
        a1 = oum.unit_vec(self.a1, return_length=False)
        v12 = self.o2 - self.o1
        x1 = v12 - np.dot(v12, a1) * a1
        x1 = oum.unit_vec(x1, return_length=False)
        y1 = np.cross(a1, x1)

        v1w = pw - self.o1
        px = np.dot(v1w, x1)
        py = np.dot(v1w, y1)
        q1 = float(oum.wrap_to_pi(np.arctan2(py, px)))

        # rotate by -q1 around a1
        R1 = oum.rotmat_from_axangle(a1, -q1)
        pw1 = self.o1 + R1 @ (pw - self.o1)
        o21 = self.o1 + R1 @ (self.o2 - self.o1)

        a2_0 = oum.unit_vec(self.a2, return_length=False)
        a2_1 = R1 @ a2_0

        v_sw = pw1 - o21
        u = v_sw - np.dot(v_sw, a2_1) * a2_1
        if np.linalg.norm(u) < 1e-9:
            return []
        u = oum.unit_vec(u, return_length=False)
        v = np.cross(a2_1, u)

        x = float(np.dot(v_sw, u))
        y = float(np.dot(v_sw, v))

        d = np.hypot(x, y)
        L2, L3 = self.L2, self.L3
        c3 = oum.clamp((d * d - L2 * L2 - L3 * L3) / (2 * L2 * L3), lo=-1.0, hi=1.0)
        s3_abs = np.sqrt(max(0.0, 1.0 - c3 * c3))

        sols = []
        for s3 in (+s3_abs, -s3_abs):
            q3 = float(np.arctan2(s3, c3))
            q2 = float(np.arctan2(y, x) - np.arctan2(L3 * s3, L2 + L3 * c3))
            sols.append((oum.wrap_to_pi(q1), oum.wrap_to_pi(q2), oum.wrap_to_pi(q3)))
        return sols

    def _solve_wrist_ZYZ(self, R36):
        """
        Solve q4,q5,q6 from R36 using ZYZ Euler (or your convention).
        """
        c5 = oum.clamp(R36[2, 2], lo=-1.0, hi=1.0)
        q5a = float(np.arccos(c5))
        q5b = float(-q5a)

        sols = []
        for q5 in (q5a, q5b):
            s5 = np.sin(q5)
            if abs(s5) < 1e-8:
                q4 = 0.0
                q6 = float(np.arctan2(R36[1, 0], R36[0, 0]))
            else:
                q4 = float(np.arctan2(R36[1, 2] / s5, R36[0, 2] / s5))
                q6 = float(np.arctan2(R36[2, 1] / s5, -R36[2, 0] / s5))
            sols.append((oum.wrap_to_pi(q4), oum.wrap_to_pi(q5), oum.wrap_to_pi(q6)))
        return sols