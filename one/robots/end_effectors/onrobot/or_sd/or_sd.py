import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb


class ORSD(oreb.EndEffectorBase, oreb.PointMixin):
    """Minimal OnRobot ScrewDriver implementation."""

    @classmethod
    def _build_structure(cls):
        structure = orbms.MechStruct()
        mesh_dir = structure.default_mesh_dir
        # Single link - screwdriver body
        lnk = orbms.Link.from_file(
            os.path.join(mesh_dir, "or_screwdriver.stl"),
            collision_type=ouc.CollisionType.MESH,
            rgb=ouc.ExtendedColor.SILVER)
        structure.add_lnk(lnk)
        structure.compile()
        return structure

    def __init__(self):
        # TCP at tip, Z-axis along screwdriver axis
        tcp_pos = np.array([0.16855, 0, 0.09509044], dtype=np.float32)
        tcp_rotmat = oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi / 2)
        super().__init__(loc_tcp_tf=oum.tf_from_rotmat_pos(tcp_rotmat, tcp_pos))
        self._is_activated = False


if __name__ == '__main__':
    import one.viewer.world as wd
    import one.scene.scene_object_primitive as ossop
    import one.robots.manipulators.kawasaki.rs007l.rs007l as rs007l

    base = wd.World(cam_pos=(2,0.5,2), cam_lookat_pos=(0, 0, .75))
    # world frame
    ossop.frame().attach_to(base.scene)
    manipulator = rs007l.RS007L()
    manipulator.attach_to(base.scene)
    screwdriver = ORSD()
    screwdriver.attach_to(base.scene)
    manipulator.engage(screwdriver, engage_tfmat=oum.tf_from_rotmat_pos(pos=(0.0, 0.0, 0.05)))
    tgt_pos = (0.3, 0.5, 0.5)
    tgt_rotmat = oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi)
    ossop.frame(pos=tgt_pos, rotmat=tgt_rotmat, color_mat=ouc.CoordColor.DYO).attach_to(base.scene)
    qs = manipulator.ik_tcp_nearest(tgt_rotmat, tgt_pos)
    print(qs)
    manipulator.fk(qs)
    ossop.frame(pos=screwdriver.gl_tcp_tf[:3, 3],
                rotmat=screwdriver.gl_tcp_tf[:3, :3],
                color_mat=ouc.CoordColor.MYC).attach_to(base.scene)
    ossop.frame(pos=screwdriver.pos, rotmat=screwdriver.rotmat).attach_to(base.scene)
    base.run()
