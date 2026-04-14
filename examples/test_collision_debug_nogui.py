import numpy as np
import mujoco
import one.collider.mj_collider as ocm
from one import oum, ovw, ossop, khi_rs007l, or_2fg7

# Debug test WITHOUT GUI
print("="*60)
print("Collision Detection Debug Test")
print("="*60)

base = ovw.World(cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, .75),
                 toggle_auto_cam_orbit=False)

robot = khi_rs007l.RS007L(pos=(.5, 0, 0.01))
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.engage(gripper)

# create ground plane at z=0
ground = ossop.plane(pos=(0, 0, 0))
ground.attach_to(base.scene)

print(f"\nGround collision shapes: {len(ground.collisions)}")
if ground.collisions:
    c = ground.collisions[0]
    print(f"  Type: {type(c).__name__}")
    print(f"  Pos: {c.pos}")
    print(f"  Normal: {c.normal if hasattr(c, 'normal') else 'N/A'}")

# setup mj collider
mjc = ocm.MJCollider()
mjc.append(robot)
mjc.append(ground)
mjc.actors = [robot]
mjc.compile(margin=0.0)

# Save XML
mjc._mjenv.save("debug_collision.xml")
print("\nSaved MuJoCo XML to: debug_collision.xml")

# Check geom info from MuJoCo model
model = mjc._mjenv.runtime.model
print(f"\nMuJoCo model has {model.ngeom} geoms:")
plane_geoms = []
for i in range(model.ngeom):
    gtype = model.geom_type[i]
    gtype_name = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"][gtype]
    bodyid = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodyid)
    contype = model.geom_contype[i]
    conaffinity = model.geom_conaffinity[i]
    if gtype == 0:  # plane
        plane_geoms.append(i)
        print(f"  geom {i}: {gtype_name:8s} body={body_name:12s} contype={contype} conaffinity={conaffinity} <<< PLANE")
    elif i < 20:  # Only print first 20 geoms
        print(f"  geom {i}: {gtype_name:8s} body={body_name:12s} contype={contype} conaffinity={conaffinity}")

if len(plane_geoms) == 0:
    print("\n*** WARNING: No plane geoms found in model! ***")
else:
    print(f"\n*** Found {len(plane_geoms)} plane geom(s) ***")

# Debug collision detection
def debug_collision(qs, label):
    print(f"\n{label}:")
    print(f"  Joint config: {qs}")
    
    # Manually update and check
    runtime = mjc._mjenv.runtime
    data = runtime.data
    
    # Update robot pose
    for actor, sl in mjc._actor_qs_slice.items():
        mjc._mjenv.sync.push_one_mecba_freebase_pose(actor, actor.quat, actor.pos)
        mjc._mjenv.sync.push_one_mecba_qpos(actor, qs[sl])
    mjc._mjenv.sync.push_all_sobj_qpos()
    
    # Run collision detection
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    
    print(f"  Number of contacts: {data.ncon}")
    if data.ncon > 0:
        for i in range(min(data.ncon, 10)):  # Print first 10 contacts
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]
            body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
            body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
            gtype1 = model.geom_type[g1]
            gtype2 = model.geom_type[g2]
            gtype1_name = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"][gtype1]
            gtype2_name = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"][gtype2]
            print(f"    Contact {i}: {body1}({gtype1_name}) <-> {body2}({gtype2_name}), dist={c.dist:.6f}, pos={c.pos}")
    
    collided = mjc.is_collided(qs)
    print(f"  is_collided() result: {collided}")
    return collided

# Test 1: Safe pose
print("\n" + "="*60)
qs_safe = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
debug_collision(qs_safe, "Test 1 - Safe pose (all joints at 0)")

# Test 2: Joint 1 bent down (should collide with ground)
qs_collision = np.array([0, np.pi/3, 0, 0, 0, 0], dtype=np.float32)
debug_collision(qs_collision, "Test 2 - Joint 1 bent down (π/3)")

# Test 3: Robot base at z=0
print("\n" + "="*60)
print("Lowering robot base to z=0...")
robot.pos = (.5, 0, 0)
mjc._mjenv.sync.push_one_mecba_freebase_pose(robot, robot.quat, robot.pos)

qs_low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
debug_collision(qs_low, "Test 3 - Robot base at z=0 (should collide)")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("If contacts are shown above, collision detection is working!")
print("If no contacts but geometry should collide, there's a bug.")
print("="*60)
