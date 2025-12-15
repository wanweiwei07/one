import numpy as np
import one.utils.math as rm
import one.utils.decorator as deco
import one.scene.scene_object as sob


class Link(sob.SceneObject):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.parent_joint = None
        self.children_joints = []


class Joint:

    def __init__(self, name, joint_type,
                 parent_link, child_link, axis,
                 origin_rotmat=np.eye(3, dtype=np.float32),
                 origin_pos=np.zeros(3, dtype=np.float32),
                 mimic=None, limit_lower=None, limit_upper=None):
        """
        :param name:
        :param joint_type: const.JointType
        :param parent_link:
        :param child_link:
        :param axis:
        :param origin_rotmat:
        :param origin_pos:
        :param mimic: None or (other_joint, multiplier, offset)
        :param limit_lower:
        :param limit_upper:
        """
        self.name = name
        self.joint_type = joint_type
        self.axis = rm.unit_vec(axis, return_length=False)
        self.origin_rotmat = origin_rotmat
        self.origin_pos = origin_pos
        # link and joints
        self.parent_link = parent_link
        self.parent_link.children_joints.append(self)
        self.child_link = child_link
        self.child_link.parent_joint = self
        # mimic
        self.mimic = mimic
        # angles and limits
        self.limit_lower = limit_lower
        self.limit_upper = limit_upper

    @property
    @deco.readonly_view
    def origin_tfmat(self):
        return rm.tfmat_from_rotmat_pos(self.origin_rotmat, self.origin_pos)


class RobotStructure:

    def __init__(self):
        self.links = []
        self.joints = []
        self.root_link = None
        self.link_order = []
        self.joint_order = []
        self.link_index_map = {}
        self.joint_index_map = {}

    def add_link(self, link):
        self.links.append(link)

    def add_joint(self, joint):
        self.joints.append(joint)

    def find_root(self):
        # Root link = one with no parent_joint
        for link in self.links:
            if link.parent_joint is None:
                return link
        raise RuntimeError("No root link found!")

    def finalize(self):
        self.root_link = self.find_root()
        self.link_order = []
        self.joint_order = []

        def dfs(link):
            self.link_order.append(link)
            for joint in link.children_joints:
                self.joint_order.append(joint)
                dfs(joint.child_link)

        dfs(self.root_link)
        # Build index maps
        self.link_index_map = {link: i for i, link in enumerate(self.link_order)}
        self.joint_index_map = {joint: i for i, joint in enumerate(self.joint_order)}

    def __repr__(self):
        return f"<RobotStructure: {len(self.links)} links, {len(self.joints)} joints>"
