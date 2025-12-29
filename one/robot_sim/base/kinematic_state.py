import numpy as np
import one.utils.constant as const
import one.utils.math as rm


class KinematicState:

    def __init__(self, structure, root_tfmat=None, qs=None):
        self.structure = structure
        self.flat = self.structure.flat
        self.root_tfmat = rm.ensure_tfmat(root_tfmat)
        if qs is None:
            self.qs = np.zeros(self.flat.n_joints, dtype=np.float32)
        else:
            self.qs = np.asarray(qs, dtype=np.float32)
        # TODO: dirty tfmats and auto fk after set qs
        # reference_wd_frames: world transforms of kinematic reference frames after joint motion
        # (indexed by link; equivalent to joint after-motion frames)
        self.wd_link_tfmat_arr = np.zeros((self.flat.n_links, 4, 4), dtype=np.float32)
        self.runtime_links = []
        for link in structure.link_dfs_order:
            self.runtime_links.append(link.clone())

    def get_link_reference_wd_tfmat(self, link):
        idx = self.structure.link_dfs_index(link)
        return self.wd_link_tfmat_arr[idx]

    def fk(self, qs=None):
        if qs is not None:
            self._set_qs(qs)
        q_resolved = self.flat.resolve_all_qs(self.qs)
        self.wd_link_tfmat_arr[self.flat.root_link_idx] = self.root_tfmat
        for lnk in self.structure.link_dfs_order:
            lnk_idx = self.structure.link_dfs_index_map[lnk]
            if lnk_idx == self.flat.root_link_idx:
                continue
            plnk_idx = self.flat.link_parent_link[lnk_idx]
            jnt_idx = self.flat.link_parent_joint[lnk_idx]
            plnk_tfmat = self.wd_link_tfmat_arr[plnk_idx]
            loc_tfmat = (self.flat.joint_origin_tfmat[jnt_idx] @
                         self._joint_motion_tfmat(jnt_idx, q_resolved[jnt_idx]))
            self.wd_link_tfmat_arr[lnk_idx] = plnk_tfmat @ loc_tfmat
        return self.wd_link_tfmat_arr

    def update(self):
        for i, link in enumerate(self.runtime_links):
            link.tfmat = self.wd_link_tfmat_arr[i]

    def attach_to(self, scene):
        for link in self.runtime_links:
            scene.add(link)

    def remove_from(self, scene):
        for link in self.runtime_links:
            scene.remove(link)

    def clone(self):
        # TODO create new kinstate ignores runtime links
        new = KinematicState(structure=self.structure,
                             root_tfmat=self.root_tfmat,
                             qs = self.qs.copy())
        new.wd_link_tfmat_arr = self.wd_link_tfmat_arr.copy()
        return new

    @property
    def toggle_render_collision(self):
        return self.runtime_links[0].toggle_render_collision

    @toggle_render_collision.setter
    def toggle_render_collision(self, flag=True):
        for link in self.runtime_links:
            link.toggle_render_collision = flag

    def _set_qs(self, values):
        # TODO active qs only (joints with no mimic)
        values = np.asarray(values, dtype=np.float32)
        assert len(values) == len(self.qs), f"Expected {len(self.qs)} joint values, got {len(values)}"
        self.qs[:] = values

    def _joint_motion_tfmat(self, jnt_idx, q):
        jnt_type = self.flat.joint_type[jnt_idx]
        jnt_ax = self.flat.joint_axis[jnt_idx]
        if jnt_type == const.JointType.FIXED:
            return np.eye(4, dtype=np.float32)
        if jnt_type == const.JointType.REVOLUTE:
            return rm.tfmat_from_rotmat_pos(rotmat=rm.rotmat_from_axangle(jnt_ax, q))
        if jnt_type == const.JointType.PRISMATIC:
            return rm.tfmat_from_rotmat_pos(pos=jnt_ax * q)
        raise TypeError(f"Unknown joint type: {jnt_type}")
