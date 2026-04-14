import numpy as np
import mujoco
import one.collider.mj_collider as ocm
from one import oum, ovw, ossop, khi_rs007l, or_2fg7

# Debug test with detailed collision info
base = ovw.World(cam_pos=(2, 2, 1.5), cam_lookat_pos=(0, 0, .75),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

robot = khi_rs007l.RS007L(pos=(.5, 0, 0.01))
robot.attach_to(base.scene)

gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
robot.engage(gripper)

# create ground plane at z=0
ground = ossop.plane(pos=(0, 0, 0))
ground.attach_to(base.scene)

print(f"Ground collision shapes: {len(ground.collisions)}")
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

# Save and print XML
mjc._mjenv.save("debug_collision.xml")

# Check geom info from MuJoCo model
model = mjc._mjenv.runtime.model
print(f"\nMuJoCo model has {model.ngeom} geoms:")
for i in range(model.ngeom):
    gtype = model.geom_type[i]
    gtype_name = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"][gtype]
    bodyid = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bodyid)
    contype = model.geom_contype[i]
    conaffinity = model.geom_conaffinity[i]
    print(f"  geom {i}: {gtype_name:8s} body={body_name:12s} contype={contype} conaffinity={conaffinity}")

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
        for i in range(min(data.ncon, 5)):  # Print first 5 contacts
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]
            body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
            body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
            print(f"    Contact {i}: {body1} <-> {body2}, dist={c.dist:.6f}, pos={c.pos}")
    
    collided = mjc.is_collided(qs)
    print(f"  is_collided() result: {collided}")
    return collided

# Test 1: Safe pose
qs_safe = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
debug_collision(qs_safe, "Test 1 - Safe pose")

# Test 2: Joint 1 bent down (should collide with ground)
qs_collision = np.array([0, np.pi/3, 0, 0, 0, 0], dtype=np.float32)
debug_collision(qs_collision, "Test 2 - Joint 1 bent (π/3)")

# Test 3: Robot base at z=0
print("\n" + "="*60)
print("Lowering robot base to z=0...")
robot.pos = (.5, 0, 0)
mjc._mjenv.sync.push_one_mecba_freebase_pose(robot, robot.quat, robot.pos)

qs_low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
debug_collision(qs_low, "Test 3 - Robot at z=0")

print("\n" + "="*60)
print("If contacts are shown above, collision detection is working!")
print("If no contacts but should collide, check plane definition.")

base.run()
