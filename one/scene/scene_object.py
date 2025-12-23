import os
import one.scene.model as mdl
import one.scene.scene_node as snd
import one.utils.decorator as deco
import one.scene.geometry_loader as gldr


class SceneObject:

    @classmethod
    def from_file(cls, path, name=None, rgb=None, alpha=1.0):
        instance = cls(name=os.path.basename(path))
        instance.add_visual(mdl.Model(geometry=gldr.load_geometry(path), rgb=rgb, alpha=alpha))
        instance.name = name
        return instance

    def __init__(self, name=None, rotmat=None, pos=None, parent_node=None):
        self.name = name
        self.node = snd.SceneNode(rotmat=rotmat, pos=pos, parent=parent_node)
        self.visuals = []
        self.collisions = []
        self.scene = None

    def attach_to(self, scene):
        scene.add(self)

    def remove_from(self, scene):
        scene.remove(self)

    def add_visual(self, model):
        self.visuals.append(model)

    def add_collision(self, model):
        self.collisions.append(model)

    def set_rotmat_pos(self, rotmat, pos):
        self.node.set_rotmat_pos(rotmat, pos)

    def clone(self, new_name=None):
        """
        Clone the scene object. DOES NOT clone the affiliated scene.
        :param new_name:
        :return:
        author: weiwei
        date: 20251215
        """
        name = new_name if new_name is not None else self.name
        new = self.__class__(name=name)
        # clone all visuals
        for model in self.visuals:
            new.add_visual(model.clone())
        # clone collisions if needed
        for col in self.collisions:
            new.add_collision(col.clone())
        return new

    @property
    @deco.readonly_view
    def pos(self):
        return self.node.pos

    @pos.setter
    def pos(self, value):
        self.node.pos = value

    @property
    @deco.readonly_view
    def rotmat(self):
        return self.node.rotmat

    @rotmat.setter
    def rotmat(self, value):
        self.node.rotmat = value

    @property
    @deco.readonly_view
    def tfmat(self):
        return self.node.tfmat

    @tfmat.setter
    def tfmat(self, value):
        self.node.tfmat = value

    @property
    def rgb(self):
        if not self.visuals:
            return None
        return self.visuals[0].rgb

    @rgb.setter
    def rgb(self, value):
        for model in self.visuals:
            model.rgb = value

    @property
    def alpha(self):
        if not self.visuals:
            return None
        return self.visuals[0].alpha

    @alpha.setter
    def alpha(self, value):
        for model in self.visuals:
            model.alpha = value

    @property
    def rgba(self):
        if not self.visuals:
            return None
        m = self.visuals[0]
        return (*m.rgb, m.alpha)

    @rgba.setter
    def rgba(self, value):
        r, g, b, a = value
        for model in self.visuals:
            model.rgb = (r, g, b)
            model.alpha = a