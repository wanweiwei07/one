import numpy as np
import one.utils.constant as ouc
import one.scene.render_model_primitive as osrmp
import one.robots.base.mech_base as orbmb
import one.robots.base.mech_structure as osrbms


def prepare_mechstruct():
    structure = osrbms.MechStruct()
    wd_lnk = osrbms.Link()
    dummy_xlnk = osrbms.Link()
    dummy_ylnk = osrbms.Link()
    body_lnk = osrbms.Link(collision_type=ouc.CollisionType.AABB)
    body_lnk.add_visual(osrmp.gen_box((.1, .1, .1)))
    body_lnk.set_inertia(mass=0.1)
    joint_x = osrbms.Joint(jnt_type=ouc.JntType.PRISMATIC,
                           parent_lnk=wd_lnk,
                           child_lnk=dummy_xlnk,
                           axis=ouc.StandardAxis.X,
                           lmt_low=-5.0, lmt_up=5.0)
    joint_y = osrbms.Joint(jnt_type=ouc.JntType.PRISMATIC,
                           parent_lnk=dummy_xlnk,
                           child_lnk=dummy_ylnk,
                           axis=ouc.StandardAxis.Y,
                           lmt_low=-5.0, lmt_up=5.0)
    joint_t = osrbms.Joint(jnt_type=ouc.JntType.REVOLUTE,
                           parent_lnk=dummy_ylnk,
                           child_lnk=body_lnk,
                           axis=ouc.StandardAxis.Z,
                           lmt_low=-np.pi, lmt_up=np.pi)
    structure.add_lnk(wd_lnk)
    structure.add_lnk(dummy_xlnk)
    structure.add_lnk(dummy_ylnk)
    structure.add_lnk(body_lnk)
    structure.add_jnt(joint_x)
    structure.add_jnt(joint_y)
    structure.add_jnt(joint_t)
    structure.compile()
    return structure


class XYThetaRobot(orbmb.MechBase):

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(rotmat, pos)
