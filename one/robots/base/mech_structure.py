import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.utils.decorator as oud
import one.scene.scene_object as osob
import one.robots.base.kinematic_chain as orc
import one.robots.base.kinematic_solver as ors


class Link(osob.SceneObject):
    """Links is essentially a SceneObject"""
    _auto_counter = 0  # maintain new counter for Link class

    def __init__(self, name=None, rotmat=None, pos=None, collision_type=None, is_fixed=False):
        super().__init__(name=name, rotmat=rotmat, pos=pos,
                         collision_type=collision_type, is_fixed=is_fixed)


class Joint:

    def __init__(self,
                 name,
                 jnt_type,
                 parent_lnk,
                 child_lnk,
                 axis,
                 rotmat=np.eye(3, dtype=np.float32),
                 pos=np.zeros(3, dtype=np.float32),
                 mmc=None,
                 lmt_low=None,
                 lmt_up=None):
        self.name = name
        self.jnt_type = jnt_type
        self.axis = oum.unit_vec(axis, return_length=False)
        self.rotmat = rotmat
        self.pos = pos
        self.parent_lnk = parent_lnk
        self.child_lnk = child_lnk
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
        for link in (jnt.parent_lnk, jnt.child_lnk):
            if link not in self.lnks:
                if auto_add_lnks:
                    self.add_lnk(link)
                else:
                    raise RuntimeError("Link not in RobotStructure")
        self.jnts.append(jnt)

    def compile(self):
        self._compiled = FlatMechStructure(self)

    @property
    def compiled(self):
        if self._compiled is None:
            raise RuntimeError("MechStruct not compiled yet. Call compile() first.")
        return self._compiled


class FlatMechStructure:
    """flat representation of RobotStructure for efficient computation"""

    def __init__(self, structure):
        self._meta = structure
        self.n_lnks = len(structure.lnks)
        self.n_jnts = len(structure.jnts)
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
        for joint in structure.jnts:
            # topology
            jnt_idx = self.jidx_map[joint]
            plnk_idx = self.lidx_map[joint.parent_lnk]
            clnk_idx = self.lidx_map[joint.child_lnk]
            self.plidx_of_jidx[jnt_idx] = plnk_idx
            self.clidx_of_jidx[jnt_idx] = clnk_idx
            self.plidx_of_lidx[clnk_idx] = plnk_idx
            self.pjidx_of_lidx[clnk_idx] = jnt_idx
            # link children
            self.clnk_ids_of_lidx[plnk_idx].append(clnk_idx)
            # joint attributes
            self.jtypes_by_idx[jnt_idx] = joint.jnt_type
            self.jax_by_idx[jnt_idx] = joint.axis
            self.jotfmat_by_idx[jnt_idx] = joint.origin_tfmat
            # mimic
            if joint.mmc is not None:
                src, mult, offset = joint.mmc
                self.mmc_src_by_idx[jnt_idx] = self.jidx_map[src]
                self.mmc_mult_by_idx[jnt_idx] = float(mult)
                self.mmc_offset_by_idx[jnt_idx] = float(offset)
            # limits
            self.jlmt_low_by_idx[jnt_idx] = float(joint.lmt_low)
            self.jlmt_high_by_idx[jnt_idx] = float(joint.lmt_up)

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
