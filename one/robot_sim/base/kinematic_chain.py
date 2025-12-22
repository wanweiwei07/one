import numpy as np


class KinematicChain:
    """get a base_link to tip_link chain"""

    def __init__(self, structure, base_link, tip_link):
        self.structure = structure
        self.flat = structure.flat
        self.base_link = base_link
        self.tip_link = tip_link
        self.base_link_idx = structure.link_dfs_index(base_link)
        self.tip_link_idx = structure.link_dfs_index(tip_link)
        self.joint_indices = self._build_chain_joints()
        # mimic joints are not supported
        self._check_no_mimic()
        # active joints on this chain (exclude fixed + mimic)
        self.active_mask = self.flat.active_joint_mask[self.joint_indices]
        self.active_joint_indices = self.joint_indices[self.active_mask]
        # active joint limits
        self.limit_lower = structure.joint_limits_lower_reg[self.active_joint_indices]
        self.limit_upper = structure.joint_limits_upper_reg[self.active_joint_indices]
        # link idx cache
        pos = 0
        lnk_idx = self.base_link_idx
        self.link_pos_in_chain = {lnk_idx: pos}
        for jnt_idx in self.joint_indices:
            pos += 1
            lnk_idx = self.flat.joint_child_link[jnt_idx]
            self.link_pos_in_chain[lnk_idx] = pos

    def __len__(self):
        return int(self.joint_indices.shape[0])

    def embed_active_qs(self, qs_active, qs_full):
        qs = qs_full.copy()
        qs[self.active_joint_indices] = qs_active
        return qs

    def extract_active_qs(self, qs_full):
        return qs_full[self.active_joint_indices]

    def _build_chain_joints(self):
        r2b_jids = self.flat.ancestor_joint_indices[self.base_link_idx]
        r2t_jids = self.flat.ancestor_joint_indices[self.tip_link_idx]
        n_common = min(r2b_jids.size, r2t_jids.size)
        if n_common == 0:
            return np.concatenate([r2b_jids[::-1], r2t_jids]).astype(np.int32)
        same_prefix = (r2b_jids[:n_common] == r2t_jids[:n_common])
        # index where paths diverge (i = length of common prefix)
        div_jids = np.where(~same_prefix)[0]
        split_idx = int(div_jids[0]) if div_jids.size > 0 else n_common
        b2lca_jids = r2b_jids[split_idx:][::-1]
        lca2t_jids = r2t_jids[split_idx:]
        return np.concatenate([b2lca_jids, lca2t_jids]).astype(np.int32)

    def _check_no_mimic(self):
        mimic_flags = self.flat.mimic_src[self.joint_indices] >= 0
        if np.any(mimic_flags):
            bad = self.joint_indices[mimic_flags]
            names = [self.structure.joint_reg_order[j].name for j in bad]
            raise ValueError(
                "KinematicChain does not support mimic joints. "
                f"Mimic joints in chain: {names}"
            )
