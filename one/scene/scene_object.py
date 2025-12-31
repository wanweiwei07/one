import one.utils.constant as const
import one.scene.render_model as mdl
import one.scene.scene_node as snd
import one.scene.geometry_loader as gldr
import one.scene.collision as sco


class SceneObject:
    _auto_counter = 0 #TODO thread safety

    @classmethod
    def auto_name(cls):
        name = f"{cls.__name__}_{cls._auto_counter}"
        cls._auto_counter += 1
        return name

    @classmethod
    def from_file(cls, path, local_rotmat=None, local_pos=None,  # render model offset
                  rotmat=None, pos=None,  # scene object pose
                  inertia=None, com=None, mass=None,
                  collision_type=None, parent_node=None,
                  rgb=None, alpha=1.0): #TODO do we expose rotmat/pos of render model here?
        instance = cls(rotmat=rotmat, pos=pos,
                       inertia=inertia, com=com, mass=mass,
                       collision_type=collision_type, parent_node=parent_node)
        instance.file_path = path
        instance.add_visual(mdl.RenderModel(geometry=gldr.load_geometry(path),
                                            rotmat=local_rotmat, pos=local_pos,
                                            rgb=rgb, alpha=alpha))
        return instance

    def __init__(self, rotmat=None, pos=None,
                 inertia=None, com=None, mass=None,
                 collision_type=None, parent_node=None):
        self.name = self.auto_name()
        self.file_path = None
        self.node = snd.SceneNode(rotmat=rotmat, pos=pos, parent=parent_node)
        self.visuals = []
        self.collisions = []
        self.collision_type = collision_type  # None means no auto collision generation
        self.toggle_render_collision = False
        self.scene = None
        self.inertia = inertia
        self.com = com
        self.mass = mass

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

    def clone(self):
        """DOES NOT clone the affiliated scene."""
        inertia = self.inertia.copy() if self.inertia is not None else None
        com = self.com.copy() if self.com is not None else None
        new = self.__class__(rotmat=self.rotmat.copy(), pos=self.pos.copy(),
                             inertia=inertia, com=com, mass=self.mass,
                             collision_type=self.collision_type,
                             parent_node=None)
        new.toggle_render_collision = self.toggle_render_collision
        new.file_path = self.file_path
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
        elif self.collision_type == const.CollisionType.PLANE:
            shape = sco.PlaneCollisionShape.fit_from_model(m)
        self.add_collision(shape)
