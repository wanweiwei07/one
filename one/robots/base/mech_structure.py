import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.utils.decorator as oud
import one.scene.scene_object as osso
import one.robots.base.kinematic_chain as orc
import one.robots.base.kinematic_solver as ors


class Link(osso.SceneObject):

    @classmethod
    def auto_name(cls, flag_str=None):
        if flag_str is None:
            flag_str = "lnk"
        if flag_str not in cls._auto_counter:
            cls._auto_counter[flag_str] = 0
        name = f"{flag_str}_{cls._auto_counter[flag_str]}"
        cls._auto_counter[flag_str] += 1
        return name


class Joint:

    def __init__(self, name, jnt_type,
                 parent_lnk, child_lnk,
                 axis, rotmat=None, pos=None,
                 mmc=None, lmt_low=None, lmt_up=None):
        self.name = name
        self.jtype = jnt_type
        self.axis = oum.unit_vec(axis, return_length=False)
        self.rotmat = oum.ensure_rotmat(rotmat)
        self.pos = oum.ensure_pos(pos)
        self.plnk = parent_lnk
        self.clnk = child_lnk
        # mimic
        self.mmc = mmc
        # joint limits
        if jnt_type == ouc.JntType.REVOLUTE:
            self.lmt_low = -2.0 * oum.pi if lmt_low is None else lmt_low
            self.lmt_up = 2.0 * oum.pi if lmt_up is None else lmt_up
        elif jnt_type == ouc.JntType.PRISMATIC:
            self.lmt_low = -1.0 if lmt_low is None else lmt_low
            self.lmt_up = 1.0 if lmt_up is None else lmt_up
        elif jnt_type == ouc.JntType.FIXED:
            self.lmt_low = 0.0
            self.lmt_up = 0.0
        else:
            raise ValueError(f"Unknown joint type: {jnt_type}")

    @property
    @oud.readonly_view
    def origin_tfmat(self):
        return oum.tfmat_from_rotmat_pos(self.rotmat, self.pos)

    def motion_tfmat(self, q):
        if self.jtype == ouc.JntType.FIXED:
            return np.eye(4, dtype=np.float32)
        if self.jtype == ouc.JntType.REVOLUTE:
            return oum.tfmat_from_rotmat_pos(
                rotmat=oum.rotmat_from_axangle(self.axis, q))
        if self.jtype == ouc.JntType.PRISMATIC:
            return oum.tfmat_from_rotmat_pos(pos=self.axis * q)
        raise TypeError(f"Unknown joint type: {self.jtype}")


class MechStruct:

    def __init__(self):
        # raw data
        self.lnks = []
        self.jnts = []
        # compiled data
        self._compiled = None  # MechStructHelper
        # kinematic chain and solver
        self._chains = {}
        self._solvers = {}

    def __repr__(self):
        return f"<MechDefinition: {len(self.lnks)} links, {len(self.jnts)} joints>"

    def get_chain(self, root_lnk, tip_lnk):
        key = (root_lnk, tip_lnk)
        if key not in self._chains:
            self._chains[key] = orc.KinematicChain(self, root_lnk, tip_lnk)
        return self._chains[key]

    def get_solver(self, root_lnk, tip_lnk):
        chain = self.get_chain(root_lnk, tip_lnk)
        if chain not in self._solvers:
            self._solvers[chain] = ors.KinematicSolver(self, chain)
        return self._solvers[chain]

    def add_lnk(self, lnk):
        if lnk in self.lnks:
            return
        self.lnks.append(lnk)

    def add_jnt(self, jnt, auto_add_lnks=True):
        for link in (jnt.plnk, jnt.clnk):
            if link not in self.lnks:
                if auto_add_lnks:
                    self.add_lnk(link)
                else:
                    raise RuntimeError("Link not in RobotStructure")
        self.jnts.append(jnt)

    def compile(self):
        self._compiled = FlatMechStructure(self)

    @property
    def n_jnts(self):
        return len(self.jnts)

    @property
    def n_lnks(self):
        return len(self.lnks)

    @property
    def compiled(self):
        if self._compiled is None:
            raise RuntimeError("MechStruct not compiled yet. Call compile() first.")
        return self._compiled


class FlatMechStructure:
    """flat representation of RobotStructure for efficient computation"""

    def __init__(self, structure):
        # self._meta = structure
        self.n_lnks = structure.n_lnks
        self.n_jnts = structure.n_jnts
        # indexing
        self.lidx_map = {lnk: i for i, lnk in enumerate(structure.lnks)}
        self.jidx_map = {j: i for i, j in enumerate(structure.jnts)}
        # topology
        self.plidx_of_lidx = np.full(self.n_lnks, -1, dtype=np.int32)
        self.pjidx_of_lidx = np.full(self.n_lnks, -1, dtype=np.int32)
        self.plidx_of_jidx = np.full(self.n_jnts, -1, dtype=np.int32)
        self.clidx_of_jidx = np.full(self.n_jnts, -1, dtype=np.int32)
        # children
        self.clnk_ids_of_lidx = [[] for _ in range(self.n_lnks)]
        # joint info
        self.jtypes_by_idx = np.zeros(self.n_jnts, dtype=np.int32)
        self.jax_by_idx = np.zeros((self.n_jnts, 3), dtype=np.float32)
        self.jotfmat_by_idx = np.zeros((self.n_jnts, 4, 4), dtype=np.float32)
        # mimic
        self.mmc_src_by_idx = np.full(self.n_jnts, -1, dtype=np.int32)
        self.mmc_mult_by_idx = np.ones(self.n_jnts, dtype=np.float32)
        self.mmc_offset_by_idx = np.zeros(self.n_jnts, dtype=np.float32)
        # limits # TODO only active non-mimic joints?
        self.jlmt_low_by_idx = np.zeros(self.n_jnts, dtype=np.float32)
        self.jlmt_high_by_idx = np.zeros(self.n_jnts, dtype=np.float32)
        # build
        self._build_from_structure(structure)
        # ---- compute root + tips (index only) ----
        self.root_lnk_idx = self._find_root_idx()
        self.tip_lnk_ids = self._find_tip_inds()
        self.root_lnk = structure.lnks[self.root_lnk_idx]
        self.tip_lnks = [structure.lnks[i] for i in self.tip_lnk_ids]
        # ancestor joint cache
        self.ancestor_jnt_ids = self._build_ancestor_cache()
        # active joint mask
        self.active_jnt_ids_mask = np.ones(self.n_jnts, dtype=bool)
        self.active_jnt_ids_mask[self.jtypes_by_idx == ouc.JntType.FIXED] = False
        self.active_jnt_ids_mask[self.mmc_src_by_idx >= 0] = False
        # traversal order (O(n) FK)
        self.lnk_ids_traversal_order = self._build_traversal_order()

    def resolve_all_qs(self, qs):
        q_resolved = np.asarray(qs, dtype=np.float32).copy()
        mask = self.mmc_src_by_idx >= 0
        src = self.mmc_src_by_idx[mask]
        q_resolved[mask] = self.mmc_mult_by_idx[mask] * q_resolved[src] + self.mmc_offset_by_idx[mask]
        return q_resolved

    def is_active_jnt(self, jnt_idx):
        return self.active_jnt_ids_mask[jnt_idx]

    def _build_from_structure(self, structure):
        for jnt in structure.jnts:
            # topology
            jidx = self.jidx_map[jnt]
            plidx = self.lidx_map[jnt.plnk]
            clidx = self.lidx_map[jnt.clnk]
            self.plidx_of_jidx[jidx] = plidx
            self.clidx_of_jidx[jidx] = clidx
            self.plidx_of_lidx[clidx] = plidx
            self.pjidx_of_lidx[clidx] = jidx
            # link children
            self.clnk_ids_of_lidx[plidx].append(clidx)
            # jnt attributes
            self.jtypes_by_idx[jidx] = jnt.jtype
            self.jax_by_idx[jidx] = jnt.axis
            self.jotfmat_by_idx[jidx] = jnt.origin_tfmat
            # mimic
            if jnt.mmc is not None:
                src, mult, offset = jnt.mmc
                self.mmc_src_by_idx[jidx] = self.jidx_map[src]
                self.mmc_mult_by_idx[jidx] = float(mult)
                self.mmc_offset_by_idx[jidx] = float(offset)
            # limits
            self.jlmt_low_by_idx[jidx] = float(jnt.lmt_low)
            self.jlmt_high_by_idx[jidx] = float(jnt.lmt_up)

    def _find_root_idx(self):
        roots = np.where(self.plidx_of_lidx < 0)[0]
        if roots.size != 1:
            raise RuntimeError(f"Mechanism can only have one root, got {roots.size}")
        return int(roots[0])

    def _find_tip_inds(self):
        tips = [i for i in range(self.n_lnks) if not self.clnk_ids_of_lidx[i]]
        return np.asarray(tips, dtype=np.int32)

    def _build_ancestor_cache(self):
        """
        ancestor_joint_indices[lnk_idx] = array of joint indices
        from root -> this link (excluding root link)
        """
        ancestor = []
        for lnk_idx in range(self.n_lnks):
            chain = []
            cur_lnk_idx = lnk_idx
            while True:
                pjnt_idx = self.pjidx_of_lidx[cur_lnk_idx]
                if pjnt_idx < 0:
                    break
                chain.append(pjnt_idx)
                cur_lnk_idx = self.plidx_of_lidx[cur_lnk_idx]
            chain.reverse()  # root -> leaf
            ancestor.append(np.asarray(chain, dtype=np.int32))
        return ancestor

    def _build_traversal_order(self):
        children = [[] for _ in range(self.n_lnks)]
        order = []

        def dfs(lidx):
            order.append(lidx)
            for clidx in self.clnk_ids_of_lidx[lidx]:
                dfs(clidx)

        dfs(self.root_lnk_idx)
        return np.asarray(order, dtype=np.int32)
