import numpy as np
import one.utils.constant as const
import one.utils.math as rm


class RobotState:

    def __init__(self, structure, qs=None):
        self.structure = structure
        self.qs = np.zeros(len(structure.joint_order), dtype=np.float32) if qs is None else qs
        # TODO dirty tfmats and auto fk after set qs
        self.link_wd_tfmats = np.zeros((len(structure.link_order), 4, 4), dtype=np.float32)
        self.runtime_links = []
        for link in structure.link_order:
            clone = link.clone()  # clone visual/collision node
            self.runtime_links.append(clone)

    def get_link_wd_tfmat(self, link):
        idx = self._link_index(link)
        return self.link_wd_tfmats[idx]

    def fk(self, qs=None, root_tfmat=np.eye(4, dtype=np.float32)):
        if qs is not None:
            self._set_qs(qs)
        root_idx = self.structure.link_index_map[self.structure.root_link]
        self.link_wd_tfmats[root_idx] = root_tfmat

        def dfs(link):
            parent_idx = self.structure.link_index_map[link]
            parent_tfmat = self.link_wd_tfmats[parent_idx]
            for joint in link.children_joints:
                jidx = self.structure.joint_index_map[joint]
                q = self.qs[jidx]
                # origin * motion
                local_tfmat = joint.origin_tfmat @ self._tfmat_joint_motion(joint, q)
                # compute child
                child_link = joint.child_link
                child_idx = self.structure.link_index_map[child_link]
                self.link_wd_tfmats[child_idx] = parent_tfmat @ local_tfmat
                dfs(child_link)

        dfs(self.structure.root_link)
        return self.link_wd_tfmats

    def update(self):
        for i, link in enumerate(self.runtime_links):
            link.set_tfmat(self.link_wd_tfmats[i])

    def attach_to(self, scene):
        for link in self.runtime_links:
            scene.add(link)

    def remove_from(self, scene):
        for link in self.runtime_links:
            scene.remove(link)

    def clone(self):
        new = RobotState(self.structure)
        new.qs[:] = self.qs
        new.link_wd_tfmats[:] = self.link_wd_tfmats
        return new

    def _set_qs(self, values):
        """Set all joints from an array-like."""
        values = np.asarray(values, dtype=np.float32)
        assert len(values) == len(self.qs)
        self.qs[:] = values

    def _tfmat_joint_motion(self, joint, q):
        if joint.joint_type == const.JointType.FIXED:
            return np.eye(4)
        if joint.joint_type == const.JointType.REVOLUTE:
            return rm.tfmat_from_rotmat_pos(rotmat=rm.rotmat_from_axangle(joint.axis, q))
        if joint.joint_type == const.JointType.PRISMATIC:
            return rm.tfmat_from_rotmat_pos(pos=joint.axis * q)
        raise TypeError(f"Unknown joint type: {joint.joint_type}")

    def _joint_index(self, joint):
        """Accepts int, name, or Joint object."""
        if isinstance(joint, int):
            return joint
        if isinstance(joint, str):
            # find by name
            for j, idx in self.structure.joint_index_map.items():
                if j.name == joint:
                    return idx
            raise KeyError(f"No joint with name '{joint}'")
        return self.structure.joint_index_map[joint]

    def _link_index(self, link):
        """Accept int, name, or Link object."""
        if isinstance(link, int):
            return link
        if isinstance(link, str):
            # find by name
            for l, idx in self.structure.link_index_map.items():
                if l.name == link:
                    return idx
            raise KeyError(f"No link with name '{link}'")
        return self.structure.link_index_map[link]
