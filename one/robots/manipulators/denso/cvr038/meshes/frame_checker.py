import argparse
import os

import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.utils.constant as ouc
import one.viewer.world as ovw


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load one CVR038 link mesh and show world frame.')
    parser.add_argument('--link', default='j4.stl',
                        help='Mesh file name under this folder, e.g. base_link.stl, j1.stl ... j6.stl')
    args = parser.parse_args()

    mesh_path = os.path.join(os.path.dirname(__file__), args.link)
    if not os.path.exists(mesh_path):
        raise ValueError(f'Link mesh not found: {mesh_path}')

    base = ovw.World(cam_pos=(0.9, 0.6, 0.6), cam_lookat_pos=(0.0, 0.0, 0.15))
    scene = base.scene
    ossop.frame().attach_to(scene)

    link_sobj = osso.SceneObject.from_file(
        path=mesh_path,
        collision_type=ouc.CollisionType.MESH,
        rgb=ouc.ExtendedColor.BEIGE,
    )
    link_sobj.attach_to(scene)

    base.run()
