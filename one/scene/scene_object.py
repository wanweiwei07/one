import one.utils.constant as ouc
import one.scene.render_model as osrm
import one.scene.scene_node as ossn
import one.scene.geometry_loader as osgl
import one.scene.collision as osc


class SceneObject:
    _auto_counter = 0  # TODO thread safety

    @classmethod
    def auto_name(cls, flag_str=None):
        if flag_str is not None:
            name = f"{cls.__name__}_{flag_str}"
        else:
            name = f"{cls.__name__}_{cls._auto_counter}"
        cls._auto_counter += 1
        return name

    @classmethod
    def from_file(cls, path, name=None, local_rotmat=None, local_pos=None,  # render model offset
                  rotmat=None, pos=None, collision_type=None, is_fixed=False,
                  rgb=None, alpha=1.0):  # TODO do we expose rotmat/pos of render model here?
        instance = cls(name=name, rotmat=rotmat, pos=pos,
                       collision_type=collision_type, is_fixed=is_fixed)
        instance.file_path = path
        instance.add_visual(osrm.RenderModel(geometry=osgl.load_geometry(path),
                                             rotmat=local_rotmat, pos=local_pos,
                                             rgb=rgb, alpha=alpha),
                            auto_make_collision=True)
        return instance

    def __init__(self, name=None, rotmat=None, pos=None,
                 collision_type=None, is_fixed=False):
        self.name = self.auto_name(flag_str=name)
        self.file_path = None
        self.node = ossn.SceneNode(rotmat=rotmat, pos=pos)
        self.visuals = []
        self.collisions = []
        self.collision_type = collision_type  # None means no auto collider generation
        self.toggle_render_collision = False
        # self.scene = None # TODO: do we need to track the affiliated scene?
        self._inrtmat = None
        self._com = None
        self._mass = None
        self._is_fixed = is_fixed

    def attach_to(self, scene):
        scene.add(self)

    def remove_from(self, scene):
        scene.remove(self)

    def add_visual(self, model, auto_make_collision=True):
        self.visuals.append(model)
        if auto_make_collision:
            self._auto_make_collision_from_model(model)

    def add_collision(self, model):
        self.collisions.append(model)

    def set_rotmat_pos(self, rotmat, pos):
        self.node.set_rotmat_pos(rotmat, pos)

    def clone(self):
        """DOES NOT clone the affiliated scene."""
        new = self.__class__(rotmat=self.rotmat.copy(), pos=self.pos.copy(),
                             collision_type=self.collision_type)
        new.toggle_render_collision = self.toggle_render_collision
        new.file_path = self.file_path
        new.set_inertia(self._inrtmat, self._com, self._mass)
        # clone all visuals
        for m in self.visuals:
            new.add_visual(m.clone(), auto_make_collision=False)
        # clone collisions if needed
        for c in self.collisions:
            new.add_collision(c.clone())
        return new

    def set_inertia(self, inrtmat=None, com=None, mass=None):
        if inrtmat is not None:
            self._inrtmat = inrtmat.copy()
        if com is not None:
            self._com = com.copy()
        if mass is not None:
            self._mass = mass

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

    @property
    def inrtmat(self):
        if self._inrtmat is None:
            return None
        return self._inrtmat.copy()

    @property
    def com(self):
        if self._com is None:
            return None
        return self._com.copy()

    @property
    def mass(self):
        if self._mass is None:
            return None
        return self._mass

    @property
    def is_fixed(self):
        return self._is_fixed

    def _auto_make_collision_from_model(self, m):
        if self.collision_type is None or self.collisions:
            return
        if self.collision_type == ouc.CollisionType.MESH:
            shape = osc.MeshCollisionShape(file_path=self.file_path,
                                           geometry=m.geometry,
                                           rotmat=m.rotmat, pos=m.pos)
        elif self.collision_type == ouc.CollisionType.SPHERE:
            shape = osc.SphereCollisionShape.fit_from_model(m)
        elif self.collision_type == ouc.CollisionType.CAPSULE:
            shape = osc.CapsuleCollisionShape.fit_from_model(m)
        elif self.collision_type == ouc.CollisionType.AABB:
            shape = osc.AABBCollisionShape.fit_from_model(m)
        elif self.collision_type == ouc.CollisionType.OBB:
            shape = osc.OBBCollisionShape.fit_from_model(m)
        elif self.collision_type == ouc.CollisionType.PLANE:
            shape = osc.PlaneCollisionShape.fit_from_model(m)
        self.add_collision(shape)
