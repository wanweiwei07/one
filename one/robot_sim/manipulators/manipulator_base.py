import numpy as np
import one.utils.math as rm
import one.robot_sim.base.robot_base as rbase


class ManipulatorBase(rbase.RobotBase):

    def __init__(self):
        super().__init__()
        self._tcp_tfmat = np.eye(4, dtype=np.float32)
        self._base_link = self.structure.root_link
        self._tip_link = self.structure.link_dfs_order[-1]
        self._chain = self.structure.get_chain(self._base_link,
                                               self._tip_link)
        self._solver = self.structure.get_solver(self._base_link,
                                                 self._tip_link)

    def set_tcp(self, rotmat=None, pos=None, tfmat=None):
        if tfmat is not None:
            self._tcp_tfmat = np.asarray(tfmat, dtype=np.float32)
        else:
            if rotmat is not None:
                self._tcp_tfmat[:3, :3] = rotmat
            if pos is not None:
                self._tcp_tfmat[:3, 3] = pos

    def reset_tcp(self):
        self._tcp_tfmat[:] = np.eye(4, dtype=np.float32)

    def ik_tcp(self,
               tgt_rotmat,
               tgt_pos,
               qs_active_init=None):
        tgt_tcp_tfmat = rm.tfmat_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_flange_tfmat = tgt_tcp_tfmat @ np.linalg.inv(self._tcp_tfmat)
        qs_active, info = self._solver.ik(
            tgt_romat=tgt_flange_tfmat[:3, :3],
            tgt_pos=tgt_flange_tfmat[:3, 3],
            qs_active_init=qs_active_init)
        if not info["converged"]:
            return None, info
        qs_full = self._chain.embed_active_qs(qs_active, self.kin_state.qs)
        return qs_full, info