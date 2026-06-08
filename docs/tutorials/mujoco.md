# MuJoCo Integration

`one.physics` bridges the scene to **MuJoCo** for rigid-body simulation. An
`MJEnv` is built from a `one` scene: it translates the scene's bodies into a
MuJoCo model, steps the physics, and syncs the resulting states back so the
viewer shows the simulated motion.

## Dropping robots onto a plane

This example stacks several RS007L clones above a ground plane and lets MuJoCo
simulate them falling. `mjenv.step` is scheduled on the viewer loop, so physics
advances every frame:

```python
--8<-- "examples/test_mujoco_rs007l.py"
```

Things to notice:

- `MJEnv(scene=base.scene)` builds the simulation from whatever is attached to
  the scene (robots, primitives, planes).
- `mjenv.save("scene.xml")` exports the generated MuJoCo model — handy for
  inspecting or reusing the converted scene.
- `base.schedule_interval(mjenv.step)` runs the simulation in step with the
  viewer.
- Register robots, grippers, and the static environment; a held/manipulated part
  is handled through the grasp attachment, not added as a separate sim body.

Related examples: `test_mujoco_bunny`, `test_mujoco_rs007l_and_bunny`,
`test_mujoco_rs007l_engage_2fg7`, `test_mujoco_xytheta`.

## See also

- [`one.physics`](../api/physics.md) — `MJEnv`, contact, model conversion.
- [Collision](collision.md) — the analytic (non-physics) collision path.
