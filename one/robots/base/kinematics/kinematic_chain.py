import numpy as np


class KinematicChain:
    """get a base_link to tip_link chain"""

    def __init__(self, structure, base_lnk, tip_lnk):
        compiled = structure._compiled
        self.base_lnk = base_lnk
        self.tip_lnk = tip_lnk
        self.base_lidx = compiled.lidx_map[base_lnk]
        self.tip_lidx = compiled.lidx_map[tip_lnk]
        self.jnt_ids_in_structure = self._get_jnt_ids(structure)
        # mimic joints are not supported
        self._check_mmc(structure)
        # active joints on this chain (exclude fixed + mimic)
        self.active_mask = compiled.active_jnt_ids_mask[self.jnt_ids_in_structure]
        self.active_jnt_ids = self.jnt_ids_in_structure[self.active_mask]
        # active joint limits
        self.lmt_lo = compiled.jlmt_low_by_idx[self.active_jnt_ids]
        self.lmt_up = compiled.jlmt_high_by_idx[self.active_jnt_ids]
        # pointer cache
        self.jnts = [structure.jnts[j] for j in self.jnt_ids_in_structure]
        # # link idx cache
        # pos = 0
        # self.lnk_pos_in_chain = {self.base_lidx: pos}
        # for jidx in self.jnt_ids_in_compiled:
        #     pos += 1
        #     lnk_idx = compiled.clidx_of_jidx[jidx]
        #     self.lnk_pos_in_chain[lnk_idx] = pos

    def __len__(self):
        return int(self.jnt_ids_in_structure.shape[0])

    def embed_active_qs(self, qs_active, qs_full):
        qs = qs_full.copy()
        qs[self.active_jnt_ids] = qs_active
        return qs

    def extract_active_qs(self, qs_full):
        return qs_full[self.active_jnt_ids]

    @property
    def n_active_jnts(self):
        return len(self.active_jnt_ids)

    @property
    def n_jnts(self):
        return len(self.jnt_ids_in_structure)

    def _get_jnt_ids(self, structure):
        compiled = structure._compiled
        r2b_jids = compiled.ancestor_jnt_ids[self.base_lidx]
        r2t_jids = compiled.ancestor_jnt_ids[self.tip_lidx]
        n_common = min(r2b_jids.size, r2t_jids.size)
        if n_common == 0:
            return np.concatenate([r2b_jids[::-1], r2t_jids]).astype(np.int32)
        same_prefix = (r2b_jids[:n_common] == r2t_jids[:n_common])
        # index where paths diverge (i = length of common prefix)
        # lca = lowest common ancestor
        div_jids = np.where(~same_prefix)[0]
        split_idx = int(div_jids[0]) if div_jids.size > 0 else n_common
        b2lca_jids = r2b_jids[split_idx:][::-1]
        lca2t_jids = r2t_jids[split_idx:]
        return np.concatenate([b2lca_jids, lca2t_jids]).astype(np.int32)

    def _check_mmc(self, structure):
        compiled = structure._compiled
        mmc_flags = compiled.mmc_src_by_idx[self.jnt_ids_in_structure] >= 0
        if np.any(mmc_flags):
            bad_ids = self.jnt_ids_in_structure[mmc_flags]
            names = [structure.jnts[j].name for j in bad_ids]
            raise ValueError("KinematicChain does not support mimic joints. "
                             f"Mimic joints in chain: {names}")
