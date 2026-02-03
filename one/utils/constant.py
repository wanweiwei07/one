import numpy as np
import matplotlib.pyplot as plt


# color constants
class BasicColor:
    # RGB
    RED = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    GREEN = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    BLUE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # CMY
    CYAN = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    MAGENTA = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    YELLOW = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    # Black White
    BLACK = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    GRAY = np.array([0.67, 0.67, 0.67], dtype=np.float32)
    # Engineering
    ORANGE = np.array([1.0, 0.47, 0.0], dtype=np.float32)
    LIME = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    DEFAULT = GRAY.copy()


class ExtendedColor:
    # Cool blues
    STEEL_BLUE = np.array([0.274, 0.510, 0.706], dtype=np.float32)
    LIGHT_STEEL_BLUE = np.array([0.690, 0.769, 0.871], dtype=np.float32)
    DEEP_SKY_BLUE = np.array([0.0, 0.749, 1.0], dtype=np.float32)
    NAVY_BLUE = np.array([0.1255, 0.1843, 0.3333], dtype=np.float32)
    ORIENTAL_BLUE = np.array([0.1490, 0.2863, 0.6157], dtype=np.float32)
    # Warm reds / oranges / pinks
    ORANGE_RED = np.array([1.0, 0.27, 0.0], dtype=np.float32)
    TOMATO = np.array([1.0, 0.388, 0.278], dtype=np.float32)
    DEEP_PINK = np.array([1.0, 0.078, 0.576], dtype=np.float32)
    PINK = np.array([1.0, 0.753, 0.796], dtype=np.float32)
    # Earth / organic tones
    CHINESE_RED = np.array([0.929, 0.428, 0.275], dtype=np.float32)
    CARROT_ORANGE = np.array([0.929, 0.428, 0.208], dtype=np.float32)
    CHOCOLATE = np.array([0.424, 0.208, 0.141], dtype=np.float32)
    SALMON_PINK = np.array([0.953, 0.651, 0.549], dtype=np.float32)
    OLIVE = np.array([0.447, 0.392, 0.047], dtype=np.float32)
    BEIGE = np.array([0.961, 0.961, 0.863], dtype=np.float32)
    IVORY = np.array([0.973, 0.957, 0.902], dtype=np.float32)
    # Purples
    VIOLET = np.array([0.353, 0.267, 0.596], dtype=np.float32)
    ROYAL_PURPLE = np.array([0.498, 0.067, 0.518], dtype=np.float32)
    # Grays / Neutrals
    STEEL_GRAY = np.array([0.451, 0.427, 0.443], dtype=np.float32)
    SLATE_GRAY = np.array([0.439, 0.502, 0.565], dtype=np.float32)
    LIGHT_SLATE_GRAY = np.array([0.467, 0.533, 0.600], dtype=np.float32)
    DIM_GRAY = np.array([0.412, 0.412, 0.412], dtype=np.float32)
    SILVER_GRAY = np.array([0.686, 0.686, 0.690], dtype=np.float32)
    MOON_GRAY = np.array([0.831, 0.851, 0.863], dtype=np.float32)
    CHINA_CLAY = np.array([0.831, 0.863, 0.827], dtype=np.float32)
    DEEP_GRAY = np.array([0.251, 0.251, 0.251], dtype=np.float32)
    # Metallic
    SILVER = np.array([0.753, 0.753, 0.753], dtype=np.float32)
    GOLD = np.array([1.0, 0.843, 0.0], dtype=np.float32)
    GOLD2 = np.array([0.9, 0.77, 0.52], dtype=np.float32)
    ANTIQUE_GOLD = np.array([0.757, 0.671, 0.020], dtype=np.float32)
    ALUMINUM = np.array([0.80, 0.80, 0.82], dtype=np.float32)
    ALUMINUM_ANODIZED = np.array([0.65, 0.68, 0.72], dtype=np.float32)
    # Greens
    LAWN_GREEN = np.array([0.486, 0.988, 0.0], dtype=np.float32)
    SPRING_GREEN = np.array([0.0, 1.0, 0.498], dtype=np.float32)
    YELLOW_GREEN = np.array([0.604, 0.804, 0.196], dtype=np.float32)


class Tab20:
    _C = plt.get_cmap("tab20").colors  # raw RGB tuples
    # ===== Blue pair =====
    BLUE_DEEP = np.array(_C[0], dtype=np.float32)
    BLUE_LIGHT = np.array(_C[1], dtype=np.float32)
    # ===== Orange pair =====
    ORANGE_DEEP = np.array(_C[2], dtype=np.float32)
    ORANGE_LIGHT = np.array(_C[3], dtype=np.float32)
    # ===== Green pair =====
    GREEN_DEEP = np.array(_C[4], dtype=np.float32)
    GREEN_LIGHT = np.array(_C[5], dtype=np.float32)
    # ===== Red pair =====
    RED_DEEP = np.array(_C[6], dtype=np.float32)
    RED_LIGHT = np.array(_C[7], dtype=np.float32)
    # ===== Purple pair =====
    PURPLE_DEEP = np.array(_C[8], dtype=np.float32)
    PURPLE_LIGHT = np.array(_C[9], dtype=np.float32)
    # ===== Brown pair =====
    BROWN_DEEP = np.array(_C[10], dtype=np.float32)
    BROWN_LIGHT = np.array(_C[11], dtype=np.float32)
    # ===== Pink pair =====
    PINK_DEEP = np.array(_C[12], dtype=np.float32)
    PINK_LIGHT = np.array(_C[13], dtype=np.float32)
    # ===== Gray pair =====
    GRAY_DEEP = np.array(_C[14], dtype=np.float32)
    GRAY_LIGHT = np.array(_C[15], dtype=np.float32)
    # ===== Olive pair =====
    OLIVE_DEEP = np.array(_C[16], dtype=np.float32)
    OLIVE_LIGHT = np.array(_C[17], dtype=np.float32)
    # ===== Cyan pair =====
    CYAN_DEEP = np.array(_C[18], dtype=np.float32)
    CYAN_LIGHT = np.array(_C[19], dtype=np.float32)


class ALPHA:
    SOLID = 1.0
    NEAR_SOLID = 0.8
    SEMI = 0.6
    LIGHT_SEMI = 0.4
    TRANSPARENT = 0.3
    INVISIBLE = 0.0


# axes

class AxisColor:
    X = BasicColor.RED
    Y = BasicColor.GREEN
    Z = BasicColor.BLUE


class CoordColor:
    RGB = np.column_stack((BasicColor.RED, BasicColor.GREEN, BasicColor.BLUE))
    MYC = np.column_stack((BasicColor.MAGENTA, BasicColor.YELLOW, BasicColor.CYAN))
    TLD = np.column_stack((ExtendedColor.TOMATO,
                           ExtendedColor.LAWN_GREEN,
                           ExtendedColor.DEEP_SKY_BLUE))
    DYO = np.column_stack((ExtendedColor.DEEP_PINK,
                           ExtendedColor.YELLOW_GREEN,
                           ExtendedColor.ORIENTAL_BLUE))


class StandardAxis:
    X = np.array([1, 0, 0], dtype=np.float32)
    Y = np.array([0, 1, 0], dtype=np.float32)
    Z = np.array([0, 0, 1], dtype=np.float32)
    ARROW_SHAFT_RADIUS = 0.005
    ARROW_HEAD_RADIUS = 0.01
    ARROW_HEAD_LENGTH = 0.04
    ARROW_LENGTH = 0.2


class StandardCoord:
    XYZ = np.column_stack((StandardAxis.X, StandardAxis.Y, StandardAxis.Z))


# graphics

class ArrowSize:
    HEAD_LENGTH = 0.04
    HEAD_RADIUS = 0.01
    SHAFT_RADIUS = 0.005


class ForceArrowSize:
    BASE_LENGTH = 0.05
    GAIN = 3.0  # scale force (length or color)
    SHAFT_RADIUS = 0.003
    HEAD_RADIUS = 0.007
    HEAD_LENGTH = 0.03


# robotics
class JntType:
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2


class CollisionType:
    SPHERE = 0
    CAPSULE = 1
    AABB = 2
    OBB = 3
    MESH = 4
    PLANE = 5


class CollisionGroup:
    ROBOT = 1 << 0  # 1
    ENV = 1 << 1  # 2
    OBJECT = 1 << 2  # 4
    ALL = ROBOT | ENV | OBJECT


class CollisionMatrix:
    """Default collision permission table"""
    DEFAULT = {
        CollisionGroup.ROBOT:  CollisionGroup.ENV | CollisionGroup.OBJECT | CollisionGroup.ROBOT,
        CollisionGroup.ENV:    CollisionGroup.ROBOT | CollisionGroup.OBJECT,
        CollisionGroup.OBJECT: CollisionGroup.ROBOT | CollisionGroup.ENV | CollisionGroup.OBJECT,
    }

class DefaultPhy:
    INERTIA = np.eye(3, dtype=np.float32)
    COM = np.zeros(3, dtype=np.float32)
    MASS = 1.0  # kg
    DENSITY = 1000.0  # kg/m3
    FRICTION = 0.5
    RESTITUTION = 0.0
    LINEAR_DAMPING = 0.01
    ANGULAR_DAMPING = 0.01
    SLEEP_VELOCITY = 0.01
    SLEEP_ANGULAR_VELOCITY = 0.01
    SLEEP_TIME = 0.5  # seconds