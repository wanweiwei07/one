import numpy as np
import one.utils.math as oum
import one.robots.base.mech_base as orbmb


class ManipulatorBase(orbmb.MechBase):

    def __init__(self, base_rotmat=None, base_pos=None):
        compiled = self.structure._compiled
        if len(compiled.tip_lnks) != 1:
            raise ValueError("ManipulatorBase must have a single tip.")
        super().__init__(base_rotmat=base_rotmat, base_pos=base_pos)
        self._tcp_tfmat = np.eye(4, dtype=np.float32)
        self._chain = self.structure.get_chain(compiled.root_lnk,
                                               compiled.tip_lnks[0])
        self._solver = self.structure.get_solver(compiled.root_lnk,
                                                 compiled.tip_lnks[0])

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

    def ik_tcp(self, tgt_rotmat, tgt_pos,
               qs_active_init=None):
        tgt_tcp_tfmat = oum.tfmat_from_rotmat_pos(tgt_rotmat, tgt_pos)
        tgt_flange_tfmat = tgt_tcp_tfmat @ np.linalg.inv(self._tcp_tfmat)
        qs_active, info = self._solver.ik(
            root_rotmat=self.base_rotmat,
            root_pos=self.base_pos,
            tgt_romat=tgt_flange_tfmat[:3, :3],
            tgt_pos=tgt_flange_tfmat[:3, 3],
            qs_active_init=qs_active_init)
        if not info["converged"]:
            return None, info
        qs_full = self._chain.embed_active_qs(qs_active, self.qs)
        return qs_full, info

    def clone(self):
        new = super().clone()
        # rebuild manipulator-specific stuff
        new._tcp_tfmat = self._tcp_tfmat.copy()
        new._chain = new.structure.get_chain(
            self.structure.compiled.root_lnk,
            self.structure.compiled.tip_lnks[0])
        new._solver = new.structure.get_solver(
            self.structure.compiled.root_lnk,
            self.structure.compiled.tip_lnks[0])
        return new
