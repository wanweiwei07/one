import one.physics.mj_compiler as mjc


class WorldNode:
    def __init__(self):
        self.name = "world"
        self.option = None
        self.default = None
        self.assets = []
        self.root_body = None
        self.sensors = []
        self.actuators = []
        self._compiler = mjc.MJCFCompiler()

    def compile_mjcf(self):
        return self._compiler.compile_mjcf(self)


class DefaultNode:
    def __init__(self):
        self.joint = {}
        self.geom = {}
        self.motor = {}


class OptionNode:
    def __init__(self):
        self.gravity = (0, 0, -9.81)
        self.timestep = 0.002
        self.integrator = "Euler"  # 或 RK4
        self.solver = "Newton"


class AssetNode:
    def __init__(self, name):
        self.name = name


class MeshAsset(AssetNode):
    def __init__(self, name, path):
        super().__init__(name)
        self.path = path


# class TextureAsset(AssetNode):
#     def __init__(self, name, path):
#         super().__init__(name)
#         self.path = path


# class MaterialAsset(AssetNode):
#     def __init__(self, name, rgba=None):
#         super().__init__(name)
#         self.rgba = rgba


class BodyNode:
    def __init__(self, name):
        self.name = name
        self.pos = (0, 0, 0)
        self.quat = (0, 0, 0, 1)
        self.inertial = None
        self.geoms = []
        self.hosting_jnts = []
        self.children = []
        self.parent = None


class JointNode:
    def __init__(self, name: str):
        self.name = name
        self.jtype_str = "hinge"  # hinge / slide / fixed
        self.axis = (1, 0, 0)
        # self.pos = (0, 0, 0)
        # self.quat = (0, 0, 0, 1)
        self.range = None
        self.damping = 5.0
        self.frictionloss = 0.7
        self.armature = 0.05


class InertialNode:
    def __init__(self, mass, com=(0, 0, 0),
                 inertia=None):
        self.mass = mass
        self.com = com
        self.inertia = inertia


class GeomNode:
    def __init__(self, name: str):
        self.name = name
        self.gtype = "box"  # sphere, capsule, mesh...
        self.size = (1, 1, 1)
        self.pos = (0, 0, 0)
        self.quat = (0, 0, 0, 1)
        self.rgba = None
        self.mesh_ref = None
        self.friction = (.2, 0.005, 0.0001)


class ActuatorNode:
    def __init__(self, name):
        self.name = name
        self.atype = "position"  # position/velocity/motor
        self.joint = None
        self.kp = 500.0
        self.kv = None

# class SensorNode:
#     def __init__(self, name: str):
#         self.name = name
#         self.stype = "jointpos"  # jointpos/jointvel/force/accel...
#         self.source_joint: JointNode | None = None
#         self.source_body: BodyNode | None = None


# class RobotNode:
#     def __init__(self, name: str):
#         self.name = name
#         self.root_body = None
#         self.joints = []
#         self.actuators = []
