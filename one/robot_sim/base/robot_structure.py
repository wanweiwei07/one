import one.utils.math as rm
import one.scene.scene_object as sob


class Link(sob.SceneObject):

    def __init__(self, parent_node=None):
        super().__init__(parent_node)
        self.parent_joint = None


class Joint:

    def __init__(self, joint_type,
                 parent_link, child_link,
                 axis, origin_tfmat,
                 limit_lower=None, limit_upper=None):
        self.joint_type = joint_type
        self.axis = rm.normalize(axis)
        self.origin_tfmat = origin_tfmat
        # link and joints
        self.parent_link = parent_link
        self.child_link = child_link
        child_link.parent_joint = self
        # angles and limits
        self.q = 0.0
        self.limit_lower = limit_lower
        self.limit_upper = limit_upper


class RobotStructure:

    def __init__(self):
        self.links=[]
        self.joints=[]
        self.children_map = {}

    def add_link(self, link):
        self.links.append(link)
        self.children_map[link] = []

    def add_joint(self, joint):
        self.joints.append(joint)
        self.children_map[joint.parent_link].append(joint)

    def get_child_joints(self, link):
        return self.children_map[link]