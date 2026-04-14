import numpy as np
import one.utils.math as oum
from one import ovw, ouc, osso, osrm, or_2fg7, ocm

base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)
# ossop.gen_frame().attach_to(base.scene)
# gripper
gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
# # box (object to collide)
# box = ossop.gen_box(half_extents=(0.03, 0.03, 0.03),
#                     rgb=ouc.BasicColor.ORANGE,
#                     collision_type=ouc.CollisionType.AABB,
#                     is_free=True)
# box.set_rotmat_pos(rotmat=np.eye(3), pos=np.array([0.0, 0.0, 0.05]))
# box.attach_to(base.scene)
# bunny (object to collide)
bunny = osso.SceneObject.from_file(
    "bunny.stl",
    collision_type=ouc.CollisionType.MESH)
bunny.attach_to(base.scene)
# target grasp pose (scene coordinates)
tgt_pos = np.array([0.00257776, -0.01682014, 0.14954071], dtype=np.float32)
tgt_rotmat = np.array([[0.92450643, 0.38116568, 0.00000005],
                       [0.32940900, -0.79897177, 0.50312352],
                       [0.19177355, -0.46514106, -0.86421424]], dtype=np.float32)
tgt_jw = 0.03603375
# move gripper to target
gripper.grip_at(tgt_pos, tgt_rotmat, tgt_jw)
# collider
collider = ocm.MJCollider()
collider.append(gripper)
collider.append(bunny)
collider.actors = [gripper]
collider.compile()
# collision check
collided = collider.is_collided(gripper.qs)
print("Collided:", collided)
collider._mjenv.save("collided.xml")
model = collider._mjenv.model
body = collider._mjenv.sync.sobj2bdy[bunny]
bid = model.body(body.name).id
geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == bid]
geom_id = geom_ids[0]
mesh_id = model.geom_dataid[geom_id]
vadr = model.mesh_vertadr[mesh_id]
vnum = model.mesh_vertnum[mesh_id]
fadr = model.mesh_faceadr[mesh_id]
fnum = model.mesh_facenum[mesh_id]
verts = model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3).copy()
faces = model.mesh_face[fadr:fadr + fnum].reshape(-1, 3).copy()
bunny_mj = osso.SceneObject()
bunny_mj.add_visual(osrm.RenderModel(
    geom=(verts, faces),
    rgb=ouc.BasicColor.GREEN,
    alpha=0.3), auto_make_collision=False)
bunny_mj.attach_to(base.scene)
base.run()
