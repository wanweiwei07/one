import numpy as np
import one.utils.math as rm
import one.utils.constant as const


class KinematicSolver:
    """Solver for one chain"""

    def __init__(self, structure, chain):
        self._compiled = structure._compiled
        self.chain = chain
        # cache per-joint constants for chain joints (ordered base->tip)
        self._jnt_ids = chain.jnt_ids
        self._loc_jnt_origin_tfmats = self._compiled.jotfmat_by_idx[self._jnt_ids]  # (m,4,4)
        self._loc_jnt_axes = self._compiled.jax_by_idx[self._jnt_ids]  # (m,3)
        self._jnt_types = self._compiled.jtypes_by_idx[self._jnt_ids]  # (m,)
        # active subset mapping: chain-index -> active-index
        self._active_mask = chain.active_mask
        self._active_pos_in_chain = np.nonzero(self._active_mask)[0].astype(np.int32)
        # joint limits
        self.lmt_low = chain.lmt_low
        self.lmt_up = chain.lmt_up

    def fk(self, qs_active, root_tfmat):
        _, _, tip_tfmat = self._forward_to_lnk(qs_active, root_tfmat)
        return tip_tfmat

    def ik(self,
           root_rotmat,
           root_pos,
           tgt_romat,
           tgt_pos,
           qs_active_init=None,
           max_iter=50,
           tol_pos=1e-4,
           tol_rot=1e-3,
           step_scale=1.0,
           pos_step_max=0.1,
           rot_step_max=0.3):
        """
        :param tgt_romat:
        :param tgt_pos:
        :param qs_active_init:
        :param max_iter:
        :param tol_pos:
        :param tol_rot:
        :param step_scale:
        :param pos_step_max: parameter for trust region
        :param rot_step_max: parameter for trust region
        :return:
        """
        root_tfmat = rm.tfmat_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tfmat = rm.tfmat_from_rotmat_pos(tgt_romat, tgt_pos)
        if qs_active_init is None:
            qs = (self.lmt_low + self.lmt_up) * 0.5
        else:
            qs = np.array(qs_active_init, dtype=np.float32)
            assert qs.shape[0] == self.n_active_jnts
        for it in range(int(max_iter)):
            _, jacmat, cur_tfmat = self._forward_to_lnk(qs, root_tfmat)
            delta_p = tgt_tfmat[:3, 3] - cur_tfmat[:3, 3]
            delta_theta = rm.delta_rotvec_between_rotmats(cur_tfmat[:3, :3],
                                                          tgt_tfmat[:3, :3])
            delta_x = np.concatenate([delta_p, delta_theta]).astype(np.float32)
            pos_err = np.linalg.norm(delta_p)
            rot_err = np.linalg.norm(delta_theta)
            if pos_err <= tol_pos and rot_err <= tol_rot:
                return qs, {"converged": True, "iters": it, "err": delta_x}
            # trust region scaling
            if pos_err > pos_step_max:
                delta_p = delta_p / pos_err * pos_step_max
            if rot_err > rot_step_max:
                delta_theta = delta_theta / rot_err * rot_step_max
            delta_x = np.concatenate([delta_p, delta_theta]).astype(np.float32)
            delta_q = np.linalg.lstsq(jacmat, delta_x, rcond=1e-4)[0]
            qs = qs + step_scale * delta_q
            # # debug purposes
            # new_robot=robot.clone()
            # new_robot.fk(qs)
            # new_robot.attach_to(base.scene)
            # print(delta_x)
        return qs, {"converged": False, "iters": max_iter, "err": delta_x}

    def _forward_to_lnk(self, qs_active, root_tfmat, up_to_link=None, local_point=None):
        assert qs_active.shape[0] == self.n_active_jnts, \
            f"Expected {self.n_active_jnts} active joints, got {len(qs_active)}"
        if up_to_link is None:
            tgt_lnk_idx = self.chain.tip_lnk_idx
        else:
            tgt_lnk_idx = self.structure.link_dfs_index(up_to_link)
        try:
            up_to_pos = self.chain.link_pos_in_chain[tgt_lnk_idx]
        except KeyError:
            raise ValueError("Specified link is not on this kinematic chain")
        q_chain = np.zeros(self.n_jnts)
        q_chain[self._active_pos_in_chain] = qs_active
        wd_lnk_tfmat_arr = np.empty((up_to_pos + 1, 4, 4))
        wd_lnk_tfmat_arr[0] = root_tfmat
        wd_jnt_tfmat_arr = np.empty((up_to_pos, 4, 4))
        for k in range(up_to_pos):
            wd_jnt_tfmat_arr[k] = wd_lnk_tfmat_arr[k] @ self._loc_jnt_origin_tfmats[k]
            wd_lnk_tfmat_arr[k + 1] = (wd_jnt_tfmat_arr[k] @
                                       self._jnt_motion_tfmat(self._jnt_types[k],
                                                              self._loc_jnt_axes[k],
                                                              q_chain[k]))
        if local_point is None:
            wd_p_tgt = wd_lnk_tfmat_arr[-1, :3, 3]
        else:
            wd_p_tgt = wd_lnk_tfmat_arr[-1, :3, :3] @ local_point + wd_lnk_tfmat_arr[-1, :3, 3]
        jacmat = np.zeros((6, self.n_active_jnts))
        for col, k in enumerate(self._active_pos_in_chain):
            if k >= up_to_pos:
                continue
            wd_ax_k = wd_jnt_tfmat_arr[k, :3, :3] @ self._loc_jnt_axes[k]
            wd_p_k = wd_jnt_tfmat_arr[k, :3, 3]
            if self._jnt_types[k] == 1:  # REVOLUTE
                jacmat[3:6, col] = wd_ax_k
                jacmat[0:3, col] = np.cross(wd_ax_k, wd_p_tgt - wd_p_k)
            elif self._jnt_types[k] == 2:  # PRISMATIC
                jacmat[0:3, col] = wd_ax_k
        return wd_p_tgt.astype(np.float32), jacmat.astype(np.float32), wd_lnk_tfmat_arr[-1].astype(np.float32)

    def _jnt_motion_tfmat(self, jnt_type, loc_jnt_ax, q):
        if jnt_type == const.JntType.FIXED:
            return np.eye(4, dtype=np.float32)
        if jnt_type == const.JntType.REVOLUTE:
            return rm.tfmat_from_rotmat_pos(rotmat=rm.rotmat_from_axangle(loc_jnt_ax, q))
        if jnt_type == const.JntType.PRISMATIC:
            return rm.tfmat_from_rotmat_pos(pos=loc_jnt_ax * q)
        raise TypeError(f"Unknown joint type: {jnt_type}")

    @property
    def n_jnts(self):
        return len(self._jnt_ids)

    @property
    def n_active_jnts(self):
        return len(self._active_pos_in_chain)
