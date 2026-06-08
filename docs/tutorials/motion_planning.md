# Motion Planning

`one.motion` provides sampling-based planners (`one.motion.probabilistic`) and
trajectory tools (`one.motion.trajectory`). They are driven by a planning context
that wraps the robot and a collision oracle (typically `MJCollider.is_collided`).

## RRT-Connect

A bidirectional tree planner for joint-space queries between a start and goal
configuration, with the collider rejecting invalid samples and edges:

```python
--8<-- "examples/test_rrtc_rs007l.py"
```

Things to notice:

- The robot, ground, and any obstacles are registered with a collider; the
  planning context turns `is_collided(qs)` into the planner's validity check.
- The result is a joint-space path; neighbouring variants
  (`test_prm_rs007l`, `test_rrtc_rs007l_dual`, `test_rrtc_rs007l_gpu_aabb`) swap
  the planner or the collision backend.

## Cartesian trajectories

When you need the tcp to follow a straight Cartesian path (e.g. a tool sweeping
over a workpiece), `one.motion.trajectory.cartesian.cartesian_to_jtraj` solves
per-step IK along the line, and `time_param.retime_trapezoidal` adds a velocity/
acceleration profile. This example mounts a screwdriver and drives its tip along
a Cartesian segment:

```python
--8<-- "examples/test_rs007l_mount_orsd_jtraj.py"
```

Things to notice:

- `cartesian_to_jtraj(robot, start_pos, start_rotmat, goal_pos, goal_rotmat,
  tcp=...)` returns joint waypoints plus the sampled poses; pass a biased
  `ref_qs` so the per-step IK stays on a smooth branch.
- `retime_trapezoidal(q_seq, v_max, a_max, dt)` time-parameterizes the waypoints.

## See also

- [`one.motion`](../api/motion.md) — RRT, PRM, planning context, trajectory tools.
- [Collision](collision.md) — building the `is_collided` oracle.
