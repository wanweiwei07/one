# Collision

`one` ships two collision backends with the same interface:

- `one.collider.cpu_simd` (`occs`) — numpy/SIMD triangle-mesh detection.
- `one.collider.gpu_simd_batch` — the GPU batch equivalent (grasp planners try
  this first and fall back to CPU).

For robot-level queries (a whole robot against the environment across many
configurations) there is `one.collider.mj_collider.MJCollider` (`ocm`), which
compiles a robot + static obstacles once and then answers `is_collided(qs)`.

## Robot-vs-ground with MJCollider

Build the collider from the robot and the obstacles, mark which bodies are the
moving **actors**, `compile()` once, then probe configurations:

```python
--8<-- "examples/test_collision_check.py"
```

Things to notice:

- `mjc.append(robot)` / `mjc.append(ground)` register bodies; `mjc.actors = [robot]`
  marks what moves; `mjc.compile(margin=...)` finalizes the broadphase.
- `mjc.is_collided(qs)` drives the robot to `qs` and returns a boolean — cheap to
  call in a planning loop.
- Register only the robot, its mounted gripper, and the static environment with
  `MJCollider`. Manipulated/held parts are *not* added as collider bodies.

## Lower-level mesh-vs-mesh

For one-off mesh pairs, `occs.create_detector()` + `occs.build_batch(items, pairs)`
+ `detector.detect_collision_batch(batch)` returns contact points (or `None`).
This is exactly what the grasp planners use to reject colliding poses (see
[`one.grasp._common.build_ee_target_detector`](../api/grasp.md)).

## See also

- [`one.collider`](../api/collider.md) — CPU/GPU detectors, MJCollider, batches.
- [Motion planning](motion_planning.md) — `is_collided` as the planner's oracle.
