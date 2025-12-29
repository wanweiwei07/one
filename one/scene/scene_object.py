import os
import one.utils.math as rm
import one.utils.constant as const
import one.scene.render_model as mdl
import one.scene.scene_node as snd
import one.scene.geometry_loader as gldr
import one.scene.collision as sco


class SceneObject:

    @classmethod
    def from_file(cls, path,
                  local_rotmat=None, local_pos=None,
                  collision_type=None, name=None,
                  rgb=None, alpha=1.0):
        instance = cls(name=os.path.basename(path) if name is None else name,
                       collision_type=collision_type)
        instance.file_path = path
        instance.add_visual(mdl.RenderModel(geometry=gldr.load_geometry(path),
                                            rotmat=local_rotmat, pos=local_pos,
                                            rgb=rgb, alpha=alpha))
        return instance

    def __init__(self, name=None, rotmat=None, pos=None,
                 collision_type=None, parent_node=None):
        self.name = name
        self.file_path = None
        self.node = snd.SceneNode(rotmat=rotmat, pos=pos, parent=parent_node)
        self.visuals = []
        self.collisions = []
        self.collision_type = collision_type  # None means no auto collision generation
        self.toggle_render_collision = False
        self.scene = None
        self.inertia = None
        self.com = None
        self.mass = None

    def attach_to(self, scene):
        scene.add(self)

    def remove_from(self, scene):
        scene.remove(self)

    def add_visual(self, model):
        self.visuals.append(model)
        self._auto_make_collision_from_model(model)

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
        new = self.__class__(name=name,
                             rotmat=self.rotmat.copy(),
                             pos=self.pos.copy(),
                             collision_type=self.collision_type,
                             parent_node=None)
        new.toggle_render_collision = self.toggle_render_collision
        new.file_path = self.file_path
        # inertia / mass / com
        new.mass = self.mass
        new.inertia = None if self.inertia is None else self.inertia.copy()
        new.com = None if self.com is None else self.com.copy()
        # clone all visuals
        for m in self.visuals:
            new.add_visual(m.clone())
        # clone collisions if needed
        for c in self.collisions:
            new.add_collision(c.clone())
        return new

    @property
    def quat(self):
        return self.node.quat

    @property
    def pos(self):
        return self.node.pos

    @pos.setter
    def pos(self, value):
        self.node.pos = value

    @property
    def rotmat(self):
        return self.node.rotmat

    @rotmat.setter
    def rotmat(self, value):
        self.node.rotmat = value

    @property
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

    def _auto_make_collision_from_model(self, m):
        if self.collision_type is None or self.collisions:
            return
        if self.collision_type == const.CollisionType.MESH:
            shape = sco.MeshCollisionShape(file_path=self.file_path,
                                           geometry=m.geometry,
                                           rotmat=m.rotmat, pos=m.pos)
        elif self.collision_type == const.CollisionType.SPHERE:
            shape = sco.SphereCollisionShape.fit_from_model(m)
        elif self.collision_type == const.CollisionType.CAPSULE:
            shape = sco.CapsuleCollisionShape.fit_from_model(m)
        elif self.collision_type == const.CollisionType.AABB:
            shape = sco.AABBCollisionShape.fit_from_model(m)
        elif self.collision_type == const.CollisionType.OBB:
            shape = sco.OBBCollisionShape.fit_from_model(m)
        self.add_collision(shape)
