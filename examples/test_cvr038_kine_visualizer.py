import numpy as np

import one.utils.math as oum
import one.utils.constant as ouc
import one.viewer.world as ovw
import one.scene.scene_object_primitive as ossop
import one.robots.base.kine_visualizer as orbkv
import one.robots.manipulators.denso.cvr038.cvr038 as ormdc


if __name__ == '__main__':
    base = ovw.World(cam_pos=(1.6, 1.2, 1.1), cam_lookat_pos=(0.0, 0.0, 0.3))
    scene = base.scene
    ossop.frame().attach_to(scene)

    robot = ormdc.CVR038()
    robot.attach_to(scene)
    robot.alpha = 0.3

    # qs = np.array([0.2, -0.6, 1.0, -0.8, 0.7, 0.3], dtype=np.float32)
    # robot.fk(qs=qs)

    jviz = orbkv.KineVisualizer(robot, mode='chain')
    jviz.attach_to(scene)

    ossop.frame(pos=robot.gl_tcp_tf[:3, 3], rotmat=robot.gl_tcp_tf[:3, :3],
                color_mat=ouc.CoordColor.MYC).attach_to(scene)
    base.run()

    tgt_pos = (0.25, 0.15, 0.25)
    tgt_rotmat = (oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 6.0) @
                  oum.rotmat_from_axangle(ouc.StandardAxis.Y, np.pi))
    ossop.frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(scene)

    base.run()