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
        self.axes, self.origins = self._compute_axes_and_origins_in_base()

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

    def _compute_axes_and_origins_in_base(self):
        """returns:
        hs: list of unit vectors h_i in base frame
        os: list of points o_i in base frame
        """
        R = np.eye(3)
        p = np.zeros(3)
        axes = []
        origins = []
        for jnt in self.jnts:
            # move to this joint frame (zero configuration)
            R = R @ jnt.rotmat
            p = p + R @ jnt.pos
            # joint axis in base frame
            ax_in_base = R @ jnt.ax
            ax_in_base = ax_in_base / np.linalg.norm(ax_in_base)
            axes.append(ax_in_base.copy())
            origins.append(p.copy())
        return axes, origins

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
