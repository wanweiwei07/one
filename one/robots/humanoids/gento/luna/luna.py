import os

import one.utils.constant as ouc
import one.robots.base.mech_base as orbmb
import one.robots.base.urdf_loader as orul


_LUNA_URDF = os.path.join(
    os.path.dirname(__file__),
    'urdf',
    'Gento Luna_URDF.urdf',
)
_PKG_PREFIX = 'package://Gento Luna_URDF/'


def prepare_mechstruct(collision_type=ouc.CollisionType.MESH):
    """Load the Gento Luna URDF into a MechStruct, via the shared urdf_loader.
    The SolidWorks-exported URDF refers to meshes as
    ``package://Gento Luna_URDF/meshes/...``; map that package root to this
    directory."""
    urdf_path = os.path.abspath(_LUNA_URDF)
    urdf_dir = os.path.dirname(urdf_path)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))

    def mesh_resolver(filename):
        if filename.startswith(_PKG_PREFIX):
            return os.path.join(pkg_dir, filename[len(_PKG_PREFIX):])
        return os.path.abspath(os.path.join(urdf_dir, filename))

    urdf = orul.load_robot_from_xacro(urdf_path, base_dir=urdf_dir)
    return orul.urdf_to_mechstruct(
        urdf, urdf_dir,
        collision_type=collision_type,
        res_dir=pkg_dir,
        mesh_resolver=mesh_resolver)


class Luna(orbmb.MechBase):
    """Gento Luna wheeled humanoid, bare (no end-effectors): a mobile underpan,
    a 6-dof torso (``Gento Luna_Leg_J1..J6``), two 7-dof arms tipping at the
    ``Arm_[LR]7_Link`` flanges, and a 2-dof head. All chains use the numerical
    IK solver (7-dof arms, no analytic solver); the 13-dof ``*_arm_torso``
    chains converge much better with a larger budget, e.g. ``max_iter=50``."""

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None, home_qs=None, is_floating=True):
        super().__init__(rotmat=rotmat, pos=pos,
                         home_qs=home_qs, is_floating=is_floating)
        lm = self.structure.lnk_map
        # named chains (which joints move) -- base/tip are structure links
        torso = lm['Gento Luna_Leg_J6_Link']
        self.add_chain('left_arm', torso, lm['Arm_L7_Link'])
        self.add_chain('right_arm', torso, lm['Arm_R7_Link'])
        self.add_chain('left_arm_torso', lm['Underpan_Base_link'], lm['Arm_L7_Link'])
        self.add_chain('right_arm_torso', lm['Underpan_Base_link'], lm['Arm_R7_Link'])
        self.add_chain('torso', lm['Underpan_Base_link'], torso)
        self.add_chain('head', torso, lm['Head_J2_Link'])

    def lnk(self, name):
        lidx = self.structure.compiled.lidx_map[self.structure.lnk_map[name]]
        return self.runtime_lnks[lidx]

    def jnt(self, name):
        return self.structure.jnt_map[name]


if __name__ == '__main__':
    import one.viewer.world as ovw
    import one.robots.base.kine_visualizer as orbkv

    base = ovw.World(cam_pos=(2.2, 1.4, 1.6),
                     cam_lookat_pos=(0.0, 0.0, 0.9))
    robot = Luna()
    robot.attach_to(base.scene)
    # render the meshes semi-transparent so the kinematic skeleton shows through
    robot.alpha = 0.3
    # stick (skeleton) model of the whole mechanism (chain=None -> all joints)
    kv = orbkv.KineVisualizer(robot, alpha=0.9)
    kv.attach_to(base.scene)
    base.run()
