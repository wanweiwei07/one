import one.scene.scene_node as snd


class SceneObject:

    def __init__(self, name=None,
                 rotmat=None, pos=None,
                 parent_node=None):
        self.name = name
        self.node = snd.SceneNode(rotmat=rotmat, pos=pos, parent=parent_node)
        self.visuals = []
        self.collisions = []

    def add_visual(self, model):
        self.visuals.append(model)

    def add_collision(self, model):
        self.collisions.append(model)

    def set_pos(self, pos):
        self.node.pos = pos

    def set_rotmat(self, rotmat):
        self.node.rotmat = rotmat

    def set_pose(self, rotmat, pos):
        self.node.set_pose(rotmat, pos)

    def set_tfmat(self, tfmat):
        self.node.tfmat = tfmat
