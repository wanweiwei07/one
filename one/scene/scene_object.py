import os
import one.scene.geometry_loader as gldr
import one.scene.model as mdl
import one.scene.scene_node as snd


class SceneObject:

    @classmethod
    def from_file(cls, path, rgb=None, alpha=1.0):
        instance = cls(name=os.path.basename(path))
        instance.add_visual(mdl.Model(geometry=gldr.load_geometry(path), rgb=rgb, alpha=alpha))
        return instance

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

    def set_rotmat_pos(self, rotmat, pos):
        self.node.set_rotmat_pos(rotmat, pos)

    def set_tfmat(self, tfmat):
        self.node.set_tfmat(tfmat)

    def set_rgba(self, rgb, alpha=None):
        for model in self.visuals:
            model.rgb = rgb
            if alpha is not None:
                model.alpha = alpha
        return self

    def clone(self, new_name=None):
        name = new_name if new_name is not None else self.name + "_clone"
        new = self.__class__(name=name)
        # clone all visuals
        for model in self.visuals:
            new.add_visual(model.clone())
        # clone collisions if needed
        for col in self.collisions:
            new.add_collision(col.clone())
        return new