import numpy as np
import one.utils.math as oum


class Ana6RSWSolver:
    # 6RSW: 6 rotational joints, spherical wrist

    def __init__(self, structure, chain):
        self.structure = structure
        self.max_iter = 5
        self.damp = 1e-4
        self._jnts = chain.jnts
        if len(self._jnts) != 6:
            raise ValueError("Expects a 6R chain!")
        self._lmt_low = np.asarray(chain.lmt_low)
        self._lmt_up = np.asarray(chain.lmt_up)
        self._wrist_offset = self._infer_wrist_offset()

    def ik(self, root_rotmat, root_pos, tgt_rotmat, tgt_pos,
           qs_seed=None, tol_pos=1e-4, tol_rot=1e-3):
        root_tf = oum.tf_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        if qs_seed is None:
            qs_seed = 0.5 * (self._lmt_low + self._lmt_up)
        # wrist center target in world
        tgt_pos_wrist = tgt_pos + tgt_rotmat @ self._wrist_offset
        seeds = self._make_branch_seeds(qs_seed)
        sols = []
        for seed in seeds:
            q = seed.copy()
            # solve q1-3 for position
            is_solved, q = self._solve_arm_position(
                root_tf, tgt_pos_wrist, q, tol_pos)
            if not is_solved:
                continue
            # solve q4-6 for orientation
            is_solved, q = self._solve_wrist_orientation(
                root_tf, tgt_tf, q, tol_rot)
            if not is_solved:
                continue
            q = self._normalize_and_clip(q)
            if not self._is_duplicate(q, sols):
                sols.append(q)
        return sols

    def _solve_arm_position(self, root_tf, p_wc_tgt, q, tol_pos):
        """wc = wrist center"""
        q = q.copy()
        for _ in range(self.max_iter):
            wd_jnt_tf, wd_lnk_tf = self._fk_all(q, root_tf)
            tip_tf = wd_lnk_tf[-1]
            p_wc_cur = tip_tf[:3, 3] + tip_tf[:3, :3] @ self._wrist_offset
            err = p_wc_tgt - p_wc_cur
            if np.linalg.norm(err) < tol_pos:
                return True, q
            # Jacobian: dp/dq = w × (p - p_axis)
            jac = np.zeros((3, 3))
            for i in range(3):
                _tf = wd_jnt_tf[i]
                w = _tf[:3, :3] @ self._jnts[i].axis
                p0 = _tf[:3, 3]
                jac[:, i] = np.cross(w, p_wc_cur - p0)
            A = jac @ jac.T + self.damp ** 2 * np.eye(3)
            dq = jac.T @ np.linalg.solve(A, err)
            q[:3] += dq
        return False, q

    def _solve_wrist_orientation(self, root_tf, tgt_tf, q, tol_rot):
        q = q.copy()
        for _ in range(self.max_iter):
            wd_jnt_tf, wd_lnk_tf = self._fk_all(q, root_tf)
            tip = wd_lnk_tf[-1]
            err = oum.delta_rotvec_between_rotmats(
                tip[:3, :3], tgt_tf[:3, :3])
            if np.linalg.norm(err) < tol_rot:
                return True, q
            # angular Jacobian: joint axes
            jac = np.zeros((3, 3))
            for col, i in enumerate([3, 4, 5]):
                _tf = wd_jnt_tf[i]
                w = _tf[:3, :3] @ self._jnts[i].axis
                jac[:, col] = w
            A = jac @ jac.T + self.damp ** 2 * np.eye(3)
            dq = jac.T @ np.linalg.solve(A, err)
            q[3:6] += dq
        return False, q

    def _make_branch_seeds(self, seed):
        seed = np.asarray(seed, dtype=np.float32)
        seeds = []
        for s1 in [+1, -1]:  # shoulder
            for s3 in [+1, -1]:  # elbow
                for flip in [0, 1]:  # wrist flip
                    q = seed.copy()
                    q[0] += s1 * np.pi
                    q[2] += s3 * np.pi
                    if flip:
                        q[3] += np.pi
                        q[4] += np.pi
                        q[5] += np.pi
                    seeds.append(q)
        seeds.append(seed.copy())
        return seeds

    def _fk_all(self, qs, root_tfmat):
        qs = np.asarray(qs, dtype=np.float32).reshape(-1)
        wd_lnk_tfarr = np.empty((7, 4, 4), dtype=np.float32)
        wd_lnk_tfarr[0] = root_tfmat
        wd_jnt_tfarr = np.empty((6, 4, 4), dtype=np.float32)
        for k in range(6):
            j = self._jnts[k]
            wd_jnt_tfarr[k] = wd_lnk_tfarr[k] @ j.origin_tfmat
            wd_lnk_tfarr[k + 1] = (
                    wd_jnt_tfarr[k] @
                    j.motion_tfmat(float(qs[k])))
        return wd_jnt_tfarr, wd_lnk_tfarr

    def _fk_tip(self, q, root_tf):
        _, wd_lnkarr = self._fk_all(q, root_tf)
        return wd_lnkarr[-1].copy()

    def _infer_wrist_offset(self):
        wd_jnt_tfarr, wd_lnk_tfarr = self._fk_all(np.zeros(6), np.eye(4))
        _tf = wd_lnk_tfarr[-1]
        # wrist center assumed at joint-4 origin
        p_wc = wd_jnt_tfarr[3][:3, 3]
        offset = _tf[:3, :3].T @ (p_wc - _tf[:3, 3])
        return offset.astype(np.float32)
