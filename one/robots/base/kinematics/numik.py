import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc


class NumIKSolver:
    """Solver for one chain"""

    def __init__(self, chain):
        self._chain = chain
        self._jnts = chain.jnts
        # active subset mapping: chain-index -> active-index
        self._active_pos_in_chain = np.nonzero(
            chain.active_mask)[0].astype(np.int32)

    def fk(self, qs_active, root_tf):
        _, _, tip_tf = self._forward(qs_active, root_tf)
        return tip_tf

    def ik(self, root_rotmat, root_pos, tgt_rotmat, tgt_pos,
           qs_active_init=None, max_iter=50, **kwargs):
        qs, info = self._backward(
            root_rotmat, root_pos, tgt_rotmat,
            tgt_pos, qs_active_init, max_iter)
        if not info["converged"]:
            return []
        else:
            return [qs]

    def _backward(self, root_rotmat, root_pos, tgt_rotmat, tgt_pos,
                  qs_active_init=None, max_iter=50,
                  tol_pos=1e-4, tol_rot=1e-3, step_scale=1.0,
                  pos_err_max=0.1, rot_err_max=0.3):
        root_tf = oum.tf_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        if qs_active_init is None:
            qs = (self._chain.lmt_lo + self._chain.lmt_up) * 0.5
        else:
            qs = np.array(qs_active_init, dtype=np.float32)
            assert qs.shape[0] == self._chain.n_active_jnts
        prev_err_norm = np.inf
        for it in range(int(max_iter)):
            _, jacmat, cur_tfmat = self._forward(qs, root_tf)
            delta_p = tgt_tf[:3, 3] - cur_tfmat[:3, 3]
            delta_theta = oum.delta_rotvec_between_rotmats(
                cur_tfmat[:3, :3], tgt_tf[:3, :3])
            delta_x = np.concatenate(
                [delta_p, delta_theta]).astype(np.float32)
            pos_err = np.linalg.norm(delta_p)
            rot_err = np.linalg.norm(delta_theta)
            if pos_err <= tol_pos and rot_err <= tol_rot:
                # if qs out of limits, treat as not converged
                lmt_lo = self._chain.lmt_lo
                lmt_up = self._chain.lmt_up
                if (np.any(qs < lmt_lo - 1e-5)
                        or np.any(qs > lmt_up + 1e-5)):
                    return qs, {"converged": False,
                                "iters": it, "err": delta_x,
                                "reason": "joint_limits_exceeded"}
                return qs, {"converged": True,
                            "iters": it, "err": delta_x}
            # check error increase
            err_norm = np.linalg.norm(delta_x)
            if err_norm > prev_err_norm:
                return qs, {"converged": False,
                            "iters": it, "err": delta_x,
                            "reason": "error_increased"}
            # trust region scaling
            if pos_err > pos_err_max:
                delta_p = delta_p / pos_err * pos_err_max
            if rot_err > rot_err_max:
                delta_theta = delta_theta / rot_err * rot_err_max
            delta_x = np.concatenate(
                [delta_p, delta_theta]).astype(np.float32)
            delta_q = np.linalg.lstsq(jacmat, delta_x, rcond=1e-4)[0]
            # null space optimization
            if self._chain.n_active_jnts > 6:
                J_pinv = np.linalg.pinv(jacmat)
                N = np.eye(self._chain.n_active_jnts) - J_pinv @ jacmat
                # subjective: stay in the middle of joint limits
                q_mid = (self._chain.lmt_lo + self._chain.lmt_up) * 0.5
                k_null = 0.2  # null space gain
                delta_q_secondary = k_null * (q_mid - qs)
                delta_q_null = N @ delta_q_secondary
                delta_q = delta_q + delta_q_null
            qs = qs + step_scale * delta_q
            # # debug purposes
            # new_robot=robot.clone()
            # new_robot.fk(qs)
            # new_robot.attach_to(base.scene)
            # print(delta_x)
        return qs, {"converged": False, "iters": max_iter,
                    "err": delta_x, "reason": "max_iters_reached"}

    def _forward(self, qs_active, root_tf, local_point=None):
        if qs_active.shape[0] != self._chain.n_active_jnts:
            raise ValueError(
                f"Expected {self._chain.n_active_jnts} active joints, "
                f"got {len(qs_active)}")
        # embed active qs into full chain qs
        q_chain = np.zeros(self._chain.n_jnts, dtype=np.float32)
        q_chain[self._active_pos_in_chain] = qs_active
        n = self._chain.n_jnts
        # world link frames along chain (base + each joint)
        wd_lnk_tfarr = np.empty((n + 1, 4, 4), dtype=np.float32)
        wd_lnk_tfarr[0] = root_tf
        # world joint frames
        wd_jnt_tfarr = np.empty((n, 4, 4), dtype=np.float32)
        for k in range(n):
            wd_jnt_tfarr[k] = wd_lnk_tfarr[k] @ self._jnts[k].zero_tf
            wd_lnk_tfarr[k + 1] = (
                    wd_jnt_tfarr[k] @ self._jnts[k].motion_tf(q_chain[k]))
        # tip position
        if local_point is None:
            wd_p_tip = wd_lnk_tfarr[-1, :3, 3]
        else:
            wd_p_tip = (wd_lnk_tfarr[-1, :3, :3] @ local_point +
                        wd_lnk_tfarr[-1, :3, 3])
        # Jacobian (6 x n_active)
        jacmat = np.zeros((6, self._chain.n_active_jnts), dtype=np.float32)
        for col, k in enumerate(self._active_pos_in_chain):
            wd_ax_k = wd_jnt_tfarr[k, :3, :3] @ self._jnts[k].ax
            wd_p_k = wd_jnt_tfarr[k, :3, 3]
            if self._jnts[k].jtype == ouc.JntType.REVOLUTE:
                jacmat[3:6, col] = wd_ax_k
                jacmat[0:3, col] = np.cross(wd_ax_k, wd_p_tip - wd_p_k)
            elif self._jnts[k].jtype == ouc.JntType.PRISMATIC:
                jacmat[0:3, col] = wd_ax_k
        return wd_p_tip, jacmat, wd_lnk_tfarr[-1]
