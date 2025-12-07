import numpy as np


class ChainState:

    def __init__(self, jlc):
        self.jlc = jlc
        self.wd_tfmat_map = {}

    def fk(self, base_T=np.eye(4, dtype=np.float32)):
        base = self.model.base_link
        self.wd_tfmat_map[base] = base_T

        def dfs(link):
            parent_T = self.world_T[link]
            for joint in self.model.get_child_joints(link):
                T_local = joint.origin_T @ self.joint_transform(joint)
                T_child = parent_T @ T_local
                self.world_T[joint.child_link] = T_child
                dfs(joint.child_link)

        dfs(base)
        return self.world_T

    def joint_transform(self, joint):
        if joint.joint_type == "fixed":
            return np.eye(4)

        axis = joint.axis / np.linalg.norm(joint.axis)

        if joint.joint_type == "revolute":
            θ = joint.q
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(θ) * K + (1 - np.cos(θ)) * (K @ K)
            T = np.eye(4);
            T[:3, :3] = R
            return T

        if joint.joint_type == "prismatic":
            T = np.eye(4)
            T[:3, 3] = axis * joint.q
            return T

    def update_scene(self):
        for link, T in self.world_T.items():
            link.node.set_tfmat(T)
