import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc


class KinematicSolver:
    """Solver for one chain"""

    def __init__(self, structure, chain):
        self._chain = chain
        self._jnts = chain.jnts
        # active subset mapping: chain-index -> active-index
        self._active_pos_in_chain = np.nonzero(
            chain.active_mask)[0].astype(np.int32)

    def fk(self, qs_active, root_tfmat):
        _, _, tip_tfmat = self._forward(qs_active, root_tfmat)
        return tip_tfmat

    def ik(self, root_rotmat, root_pos, tgt_romat, tgt_pos,
           qs_active_init=None, max_iter=50,
           tol_pos=1e-4, tol_rot=1e-3, step_scale=1.0,
           pos_err_max=0.1, rot_err_max=0.3):
        root_tfmat = oum.tfmat_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tfmat = oum.tfmat_from_rotmat_pos(tgt_romat, tgt_pos)
        if qs_active_init is None:
            qs = (self._chain.lmt_low + self._chain.lmt_up) * 0.5
        else:
            qs = np.array(qs_active_init, dtype=np.float32)
            assert qs.shape[0] == self._chain.n_active_jnts
        for it in range(int(max_iter)):
            _, jacmat, cur_tfmat = self._forward(qs, root_tfmat)
            delta_p = tgt_tfmat[:3, 3] - cur_tfmat[:3, 3]
            delta_theta = oum.delta_rotvec_between_rotmats(cur_tfmat[:3, :3],
                                                           tgt_tfmat[:3, :3])
            delta_x = np.concatenate([delta_p, delta_theta]).astype(np.float32)
            pos_err = np.linalg.norm(delta_p)
            rot_err = np.linalg.norm(delta_theta)
            if pos_err <= tol_pos and rot_err <= tol_rot:
                return qs, {"converged": True, "iters": it, "err": delta_x}
            # trust region scaling
            if pos_err > pos_err_max:
                delta_p = delta_p / pos_err * pos_err_max
            if rot_err > rot_err_max:
                delta_theta = delta_theta / rot_err * rot_err_max
            delta_x = np.concatenate([delta_p, delta_theta]).astype(np.float32)
            delta_q = np.linalg.lstsq(jacmat, delta_x, rcond=1e-4)[0]
            qs = qs + step_scale * delta_q
            # # debug purposes
            # new_robot=robot.clone()
            # new_robot.fk(qs)
            # new_robot.attach_to(base.scene)
            # print(delta_x)
        return qs, {"converged": False, "iters": max_iter, "err": delta_x}

    def _forward(self, qs_active, root_tfmat, local_point=None):
        if qs_active.shape[0] != self._chain.n_active_jnts:
            raise ValueError(
                f"Expected {self._chain.n_active_jnts} active joints, "
                f"got {len(qs_active)}")
        # embed active qs into full chain qs
        q_chain = np.zeros(self._chain.n_jnts, dtype=np.float32)
        q_chain[self._active_pos_in_chain] = qs_active
        n = self._chain.n_jnts
        # world link frames along chain (base + each joint)
        wd_lnk_tfmat_arr = np.empty((n + 1, 4, 4), dtype=np.float32)
        wd_lnk_tfmat_arr[0] = root_tfmat
        # world joint frames
        wd_jnt_tfmat_arr = np.empty((n, 4, 4), dtype=np.float32)
        for k in range(n):
            wd_jnt_tfmat_arr[k] = wd_lnk_tfmat_arr[k] @ self._jnts[k].origin_tfmat
            wd_lnk_tfmat_arr[k + 1] = (
                    wd_jnt_tfmat_arr[k] @ self._jnts[k].motion_tfmat(q_chain[k]))
        # tip position
        if local_point is None:
            wd_p_tip = wd_lnk_tfmat_arr[-1, :3, 3]
        else:
            wd_p_tip = (wd_lnk_tfmat_arr[-1, :3, :3] @ local_point +
                        wd_lnk_tfmat_arr[-1, :3, 3])
        # Jacobian (6 x n_active)
        jacmat = np.zeros((6, self._chain.n_active_jnts), dtype=np.float32)
        for col, k in enumerate(self._active_pos_in_chain):
            wd_ax_k = wd_jnt_tfmat_arr[k, :3, :3] @ self._jnts[k].axis
            wd_p_k = wd_jnt_tfmat_arr[k, :3, 3]
            if self._jnts[k].jtype == ouc.JntType.REVOLUTE:
                jacmat[3:6, col] = wd_ax_k
                jacmat[0:3, col] = np.cross(wd_ax_k, wd_p_tip - wd_p_k)
            elif self._jnts[k].jtype == ouc.JntType.PRISMATIC:
                jacmat[0:3, col] = wd_ax_k
        return wd_p_tip, jacmat, wd_lnk_tfmat_arr[-1]
