import numpy as np
import mujoco
import one.collider.mj_collider as ocm
from one import oum, ouc, ovw, ossop, khi_rs007l, or_2fg7

print("="*60)
print("Analyzing Robot Collision Geometry")
print("="*60)

base = ovw.World()

robot = khi_rs007l.RS007L(pos=(0, 0, 0))
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.engage(gripper)

# Check robot's actual geometry bounds
print(f"\nRobot position: {robot.pos}")
print(f"Robot has {len(robot.visuals)} visual models")
print(f"Robot structure has {robot.structure.compiled.n_lnks} links")

# Get bounding box of robot in home position
robot.fk(qs=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))

# Check each link's geometry
print("\nLink geometries:")
for i, lnk in enumerate(robot.runtime_lnks):
    if hasattr(lnk, 'collisions') and lnk.collisions:
        for c in lnk.collisions:
            aabb_min, aabb_max = c.aabb
            print(f"  Link {i}: AABB min={aabb_min}, max={aabb_max}")
            break

# Create ground at different heights
ground = ossop.box(
    half_extents=(50, 50, 0.05),
    pos=(0, 0, -0.05),
    collision_type=ouc.CollisionType.AABB
)
ground.attach_to(base.scene)

mjc = ocm.MJCollider()
mjc.append(robot)
mjc.append(ground)
mjc.actors = [robot]
mjc.compile(margin=0.0)

model = mjc._mjenv.runtime.model
data = mjc._mjenv.runtime.data

# Get all body positions in the scene
print("\n" + "="*60)
print("MuJoCo Body Positions (at home config):")
print("="*60)

qs_home = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
mjc._mjenv.sync.push_one_mecba_freebase_pose(robot, robot.quat, robot.pos)
mjc._mjenv.sync.push_one_mecba_qpos(robot, qs_home)
mjc._mjenv.sync.push_all_sobj_qpos()
mujoco.mj_kinematics(model, data)

for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    body_pos = data.xpos[i]
    print(f"  Body {i:2d} ({body_name:12s}): pos = {body_pos}")

# Get geom positions
print("\n" + "="*60)
print("MuJoCo Geom Positions and AABBs:")
print("="*60)

mujoco.mj_collision(model, data)

for i in range(min(model.ngeom, 12)):  # Print first 12 geoms
    gtype = model.geom_type[i]
    gtype_name = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"][gtype]
    bodyid = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodyid)
    geom_pos = data.geom_xpos[i]
    
    # Get AABB
    aabb = data.geom_xpos[i]  # This is center, not AABB
    print(f"  geom {i:2d} ({gtype_name:8s}, {body_name:12s}): center={geom_pos}")

# Test collision with ground at various heights
print("\n" + "="*60)
print("Testing Ground at Different Heights:")
print("="*60)

for z_ground in [0, 0.1, 0.2, 0.3, 0.35, 0.36, 0.37]:
    ground.pos = (0, 0, z_ground - 0.05)  # Center is at z - half_thickness
    mjc2 = ocm.MJCollider()
    mjc2.append(robot)
    mjc2.append(ground)
    mjc2.actors = [robot]
    mjc2.compile(margin=0.0)
    
    collided = mjc2.is_collided(qs_home)
    print(f"  Ground top at z={z_ground:.2f}: collided = {collided}")

print("="*60)
