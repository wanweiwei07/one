import numpy as np
import one.utils.math as oum

class AnaSphWstSolver:

    def __init__(self, structure, chain, T6TCP0=None):
        self._chain = chain
        self._jnts = chain.joints
        T6TCP0 = oum.ensure_tfmat(T6TCP0)
        # cache zero-pose joint frames in root
        self.T0J0 = self._zero_pose_joint_frames_in_root()
        # extract key points in root frame at zero pose
        self.o1 = self.T0J0[0][:3, 3]
        self.o2 = self.T0J0[1][:3, 3]
        self.o3 = self.T0J0[2][:3, 3]
        self.ow = self.T0J0[3][:3, 3]  # wrist center (assume J4 origin is wrist center)
        # extract axes in root frame at zero pose
        self.a1 = self.T0J0[0][:3, :3] @ self._jnts[0].axis
        self.a2 = self.T0J0[1][:3, :3] @ self._jnts[1].axis
        self.a3 = self.T0J0[2][:3, :3] @ self._jnts[2].axis
        self.a4 = self.T0J0[3][:3, :3] @ self._jnts[3].axis
        self.a5 = self.T0J0[4][:3, :3] @ self._jnts[4].axis
        self.a6 = self.T0J0[5][:3, :3] @ self._jnts[5].axis
        # link lengths for planar 2R (in the elbow plane)
        # use distances between joint origins as a practical geometric length
        self.L2 = float(np.linalg.norm(self.o3 - self.o2))
        self.L3 = float(np.linalg.norm(self.ow - self.o3))

    def _zero_pose_joint_frames_in_root(self):
        T0J = []
        T = np.eye(4, dtype=np.float32)
        for k in range(6):
            Tj = T @ self._jnts[k].origin_tfmat
            T0J.append(Tj)
            T = Tj @ self._jnts[k].motion_tfmat(0.0)
        return T0J

    def _fk_R03(self, q1, q2, q3):
        # Build R03 by chaining joint rotations in their *current* joint frames.
        # Here we approximate using axes in root at zero + rotating them properly is complex.
        # The robust way: just run your existing forward FK to get R03.
        raise NotImplementedError("Use your KinematicSolver.forward to compute R03 for q1-3.")

    def _solve_first3(self, pw):
        """
        Solve q1,q2,q3 from wrist center position pw (root frame).
        Return list of (q1,q2,q3) candidates (up to 4).
        This implementation assumes:
          - q1 is rotation around axis1, and after q1 the arm (j2/j3) is planar.
          - We use the plane normal = axis2 (since joints2/3 rotate about that axis).
        """
        # --- q1: rotate around axis1 to align the wrist center with the shoulder direction ---
        # We build a local frame for joint1 at zero: x1,y1 span perpendicular to a1.
        a1 = _unit(self.a1)
        # choose reference x1 using vector from o1 to o2 projected to plane ⟂ a1
        v12 = self.o2 - self.o1
        x1 = v12 - np.dot(v12, a1) * a1
        x1 = _unit(x1)
        y1 = np.cross(a1, x1)

        # express pw in joint1-local coordinates (perp plane)
        v1w = pw - self.o1
        px = np.dot(v1w, x1)
        py = np.dot(v1w, y1)
        q1 = np.arctan2(py, px)  # candidate (the other branch is q1+pi, handled by elbow branches usually)
        q1 = float(_wrap_to_pi(q1))

        # rotate pw into the "after q1" frame: equivalent is rotate around a1 by -q1
        R1 = _rot_axis_angle(a1, -q1)
        pw1 = self.o1 + R1 @ (pw - self.o1)
        o21 = self.o1 + R1 @ (self.o2 - self.o1)

        # now solve planar 2R for q2,q3 in plane perpendicular to a2 (use a2 at zero rotated similarly)
        a2_0 = _unit(self.a2)
        a2_1 = R1 @ a2_0

        # plane basis (u,v) in plane ⟂ a2_1
        # choose u from shoulder->wrist projected onto plane
        v_sw = pw1 - o21
        u = v_sw - np.dot(v_sw, a2_1) * a2_1
        if np.linalg.norm(u) < 1e-9:
            return []  # degenerate
        u = _unit(u)
        v = np.cross(a2_1, u)

        # 2D coords of wrist center relative to shoulder origin o2
        x = float(np.dot(v_sw, u))
        y = float(np.dot(v_sw, v))

        d = np.hypot(x, y)
        L2, L3 = self.L2, self.L3

        # cosine law for elbow
        c3 = _clamp((d*d - L2*L2 - L3*L3) / (2*L2*L3))
        s3_abs = np.sqrt(max(0.0, 1.0 - c3*c3))

        sols = []
        for s3 in (+s3_abs, -s3_abs):  # elbow up/down
            q3 = float(np.arctan2(s3, c3))
            q2 = float(np.arctan2(y, x) - np.arctan2(L3*s3, L2 + L3*c3))
            sols.append((_wrap_to_pi(q1), _wrap_to_pi(q2), _wrap_to_pi(q3)))
        return sols

    def _solve_wrist_ZYZ(self, R36):
        """
        Solve q4,q5,q6 from R36 assuming ZYZ Euler:
          R36 = Rz(q4) * Ry(q5) * Rz(q6)
        Return 2 solutions (wrist flip).
        """
        # q5 from R36[2,2] if using ZYZ about z-y-z (in standard basis)
        # This assumes your wrist axes align with those bases in frame3/4/5.
        c5 = _clamp(R36[2, 2])
        q5a = float(np.arccos(c5))
        q5b = float(-q5a)

        sols = []
        for q5 in (q5a, q5b):
            s5 = np.sin(q5)
            if abs(s5) < 1e-8:
                # singular: q4+q6 coupled
                q4 = 0.0
                q6 = float(np.arctan2(R36[1,0], R36[0,0]))
            else:
                q4 = float(np.arctan2(R36[1,2]/s5, R36[0,2]/s5))
                q6 = float(np.arctan2(R36[2,1]/s5, -R36[2,0]/s5))
            sols.append((_wrap_to_pi(q4), _wrap_to_pi(q5), _wrap_to_pi(q6)))
        return sols

    def ik_all(self, T0TCP):
        """
        Return up to 8 solutions for q[0..5].
        """
        T0TCP = np.asarray(T0TCP, dtype=np.float32)
        # convert target to joint6 frame
        T0_6 = T0TCP @ np.linalg.inv(self.T6TCP0)
        R06 = T0_6[:3, :3]
        p06 = T0_6[:3, 3]

        pw = p06  # assume joint6 origin is wrist center

        q123_list = self._solve_first3(pw)
        sols = []

        for (q1, q2, q3) in q123_list:
            # IMPORTANT: R03 needs to be computed using your true FK (not the simplified axis chaining)
            # Here we expect you to provide R03 via your KinematicSolver forward to link3.
            # Suppose you have function get_R03(q1,q2,q3) -> 3x3:
            R03 = self.get_R03_from_fk(q1, q2, q3)  # <-- you implement by calling your solver

            R36 = R03.T @ R06
            for (q4, q5, q6) in self._solve_wrist_ZYZ(R36):
                sols.append(np.array([q1,q2,q3,q4,q5,q6], dtype=np.float32))

        # optional: joint limit filtering + unique
        return sols

    # ---- You must plug this with your real FK ----
    def get_R03_from_fk(self, q1, q2, q3):
        raise NotImplementedError