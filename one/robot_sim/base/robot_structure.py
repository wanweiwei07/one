import numpy as np
import one.utils.math as rm
import one.utils.constant as const
import one.utils.decorator as deco
import one.scene.scene_object as sob
import one.robot_sim.base.kinematic_chain as rchain
import one.robot_sim.base.kinematic_solver as rsolver


class Link(sob.SceneObject):
    # TODO: separate link and joints, use external data structure to manage the topology
    _auto_counter = 0  # maintain new counter for Link class

    def __init__(self, name=None, rotmat=None, pos=None,
                 collision_type=None, parent_node=None):
        super().__init__(name=name, rotmat=rotmat, pos=pos,
                         parent_node=parent_node,
                         collision_type=collision_type)
        self.parent_joint = None
        self.children_joints = []


class Joint:

    def __init__(self,
                 name,
                 joint_type,
                 parent_link,
                 child_link,
                 axis,
                 rotmat=np.eye(3, dtype=np.float32),
                 pos=np.zeros(3, dtype=np.float32),
                 mimic=None,
                 limit_lower=None,
                 limit_upper=None):
        """
        :param name:
        :param joint_type: JointType
        :param parent_link:
        :param child_link:
        :param axis:
        :param rotmat:
        :param pos:
        :param mimic: None or (other_joint, multiplier, offset)
        :param limit_lower:
        :param limit_upper:
        """
        self.name = name
        self.joint_type = joint_type
        self.axis = rm.unit_vec(axis, return_length=False)
        self.rotmat = rotmat
        self.pos = pos
        # link and joints
        self.parent_link = parent_link
        self.parent_link.children_joints.append(self)
        self.child_link = child_link
        self.child_link.parent_joint = self
        # mimic
        self.mimic = mimic
        # joint limits
        if joint_type == const.JointType.REVOLUTE:
            self.limit_lower = -2.0 * rm.pi if limit_lower is None else limit_lower
            self.limit_upper = 2.0 * rm.pi if limit_upper is None else limit_upper
        elif joint_type == const.JointType.PRISMATIC:
            self.limit_lower = -1.0 if limit_lower is None else limit_lower
            self.limit_upper = 1.0 if limit_upper is None else limit_upper
        elif joint_type == const.JointType.FIXED:
            self.limit_lower = 0.0
            self.limit_upper = 0.0
        else:
            raise ValueError(f"Unknown joint type: {joint_type}")

    @property
    @deco.readonly_view
    def origin_tfmat(self):
        return rm.tfmat_from_rotmat_pos(self.rotmat, self.pos)


class RobotStructure:
    # TODO unify with flat representation

    def __init__(self):
        self.links = []
        self.joints = []
        self.root_link = None
        # dfs default
        self.link_dfs_order = []
        self.link_dfs_index_map = {}
        self.joint_reg_order = []
        self.joint_reg_index_map = {}
        self.flat = None  # efficient computation representation
        # cached joint limits
        self.joint_limits_lower_reg = None
        self.joint_limits_upper_reg = None
        # kinematic chain and solver
        self._chains = {}
        self._solvers = {}

    def __repr__(self):
        return f"<RobotStructure: {len(self.links)} links, {len(self.joints)} joints>"

    def get_chain(self, base_link, tip_link):
        key = (base_link, tip_link)
        if key not in self._chains:
            self._chains[key] = rchain.KinematicChain(self, base_link, tip_link)
        return self._chains[key]

    def get_solver(self, base_link, tip_link):
        chain = self.get_chain(base_link, tip_link)
        if chain not in self._solvers:
            self._solvers[chain] = rsolver.KinematicSolver(self, chain)
        return self._solvers[chain]

    def add_link(self, link):
        if link in self.links:
            return
        self.links.append(link)

    def add_joint(self, joint, auto_add_links=True):
        for link in (joint.parent_link, joint.child_link):
            if link not in self.links:
                if auto_add_links:
                    self.add_link(link)
                else:
                    raise RuntimeError("Link not in RobotStructure")
        self.joints.append(joint)

    def finalize(self):
        if self.flat is not None:
            raise RuntimeError("RobotStructure already finalized")
        self.root_link = self._find_root()
        self.link_dfs_order = []

        def dfs(link):
            self.link_dfs_order.append(link)
            for joint in link.children_joints:
                dfs(joint.child_link)

        dfs(self.root_link)
        # Build index maps
        self.link_dfs_index_map = {lnk: i for i, lnk in enumerate(self.link_dfs_order)}
        # joint registry order
        self.joint_reg_order = list(self.joints)
        self.joint_reg_index_map = {jnt: i for i, jnt in enumerate(self.joint_reg_order)}
        # Build flat representation
        self.flat = FlatRobotStructure(self)
        # cache joint limits in registry order
        self.joint_limits_lower_reg = np.zeros(len(self.joint_reg_order), dtype=np.float32)
        self.joint_limits_upper_reg = np.zeros(len(self.joint_reg_order), dtype=np.float32)
        for i, jnt in enumerate(self.joint_reg_order):
            self.joint_limits_lower_reg[i] = float(jnt.limit_lower)
            self.joint_limits_upper_reg[i] = float(jnt.limit_upper)

    def link_dfs_index(self, link):
        """accept int, name, or Link object."""
        if isinstance(link, int):
            return link
        if isinstance(link, str):
            # find by name
            for l, idx in self.link_dfs_index_map.items():
                if l.name == link:
                    return idx
            raise KeyError(f"No link with name '{link}'")
        return self.link_dfs_index_map[link]

    def joint_reg_index(self, joint):
        """accepts int, name, or Joint object."""
        if isinstance(joint, int):
            return joint
        if isinstance(joint, str):
            # find by name
            for j, idx in self.joint_reg_index_map.items():
                if j.name == joint:
                    return idx
            raise KeyError(f"No joint with name '{joint}'")
        return self.joint_reg_index_map[joint]

    def _find_root(self):
        # Root link = one with no parent_joint
        for link in self.links:
            if link.parent_joint is None:
                return link
        raise RuntimeError("No root link found!")

    @property
    def n_links(self):
        return len(self.links)

    @property
    def n_joints(self):
        return len(self.joints)


class FlatRobotStructure:
    """flat representation of RobotStructure for efficient computation"""

    def __init__(self, structure):
        self.n_links = len(structure.link_dfs_order)
        self.n_joints = len(structure.joint_reg_order)
        # topology
        self.link_parent_link = np.full(self.n_links, -1, dtype=np.int32)
        self.link_parent_joint = np.full(self.n_links, -1, dtype=np.int32)
        self.joint_parent_link = np.full(self.n_joints, -1, dtype=np.int32)
        self.joint_child_link = np.full(self.n_joints, -1, dtype=np.int32)
        # joint info
        self.joint_type = np.zeros(self.n_joints, dtype=np.int32)
        self.joint_axis = np.zeros((self.n_joints, 3), dtype=np.float32)
        self.joint_origin_tfmat = np.zeros((self.n_joints, 4, 4), dtype=np.float32)
        # mimic
        self.mimic_src = np.full(self.n_joints, -1, dtype=np.int32)
        self.mimic_mult = np.ones(self.n_joints, dtype=np.float32)
        self.mimic_off = np.zeros(self.n_joints, dtype=np.float32)
        # root link index
        self.root_link_idx = structure.link_dfs_index_map[structure.root_link]
        # build
        self._build_from_structure(structure)
        self.ancestor_joint_indices = self._build_ancestor_cache()
        # active joint mask
        self.active_joint_mask = np.ones(self.n_joints, dtype=bool)
        self.active_joint_mask[self.joint_type == const.JointType.FIXED] = False
        self.active_joint_mask[self.mimic_src >= 0] = False

    def resolve_all_qs(self, qs):
        q_resolved = np.asarray(qs, dtype=np.float32).copy()
        mask = self.mimic_src >= 0
        src = self.mimic_src[mask]
        q_resolved[mask] = self.mimic_mult[mask] * q_resolved[src] + self.mimic_off[mask]
        return q_resolved

    def is_active_joint(self, jnt_idx):
        return self.active_joint_mask[jnt_idx]

    def _build_from_structure(self, structure):
        # root
        self.link_parent_link[self.root_link_idx] = -1
        self.link_parent_joint[self.root_link_idx] = -1
        for joint in structure.joint_reg_order:
            jnt_idx = structure.joint_reg_index_map[joint]
            jnt_plnk_idx = structure.link_dfs_index_map[joint.parent_link]
            jnt_clnk_idx = structure.link_dfs_index_map[joint.child_link]
            self.joint_parent_link[jnt_idx] = jnt_plnk_idx
            self.joint_child_link[jnt_idx] = jnt_clnk_idx
            self.link_parent_link[jnt_clnk_idx] = jnt_plnk_idx
            self.link_parent_joint[jnt_clnk_idx] = jnt_idx
            # joint attributes
            self.joint_type[jnt_idx] = joint.joint_type
            self.joint_axis[jnt_idx] = joint.axis
            self.joint_origin_tfmat[jnt_idx] = joint.origin_tfmat
            # mimic
            if joint.mimic is not None:
                src_jnt, mult, off = joint.mimic
                self.mimic_src[jnt_idx] = structure.joint_reg_index_map[src_jnt]
                self.mimic_mult[jnt_idx] = float(mult)
                self.mimic_off[jnt_idx] = float(off)

    def _build_ancestor_cache(self):
        """
        ancestor_joint_indices[lnk_idx] = array of joint indices
        from root -> this link (excluding root link)
        """
        ancestor = []
        for link_idx in range(self.n_links):
            chain = []
            cur = link_idx
            while True:
                lnk_pjnt_idx = self.link_parent_joint[cur]
                if lnk_pjnt_idx < 0:
                    break
                chain.append(lnk_pjnt_idx)
                cur = self.link_parent_link[cur]
            chain.reverse()  # root -> leaf
            ancestor.append(np.asarray(chain, dtype=np.int32))
        return ancestor
