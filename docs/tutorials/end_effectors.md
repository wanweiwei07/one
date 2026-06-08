# End-Effectors & Tool Change

End-effectors are `MechBase` mechanisms too. They mix in behavior and register a
working tcp:

- **Grippers** (`GripperMixin`) register a `'grasp_center'` tcp and expose
  `open()` / `close()` / `set_jaw_width()` and `grasp(child)` / `release(child)`.
- **Point tools** (`PointMixin`, e.g. a screwdriver) register a `'tip'` tcp and
  expose `touch_at(...)` / `attach(...)`.

## Mounting

`arm.mount(ee, parent_lnk, loc_tf, update=True)` attaches an end-effector to a
link; from then on every `arm.fk(...)` propagates to the mounted tool. `update`
controls only whether the tool snaps into place *at the moment of mounting* — it
always follows during motion. `arm.unmount(ee)` detaches it (the tool stays where
it was). To pick a target with the mounted tool, solve **cross-object IK** to the
tool's tcp:

```python
qs = arm.ik(pos, rotmat, tcp=gripper.tcp('grasp_center'))
```

## A full tool-change sequence

This example starts with a bare RS007L, mounts a 2FG7 gripper to pick and place a
bunny, returns it to its stand and unmounts it, then mounts a screwdriver to work
above the bunny — exercising mount / grasp / release / unmount end to end:

```python
--8<-- "examples/test_rs007l_toolchange.py"
```

Things to notice:

- A tool "stand" pose is just a flange pose; mounting with an identity `loc_tf`
  while the flange sits at the stand is a no-jump pickup.
- IK targets the active tool's tcp: `gripper.tcp('grasp_center')` while the
  gripper is on, `screwdriver.tcp('tip')` while the screwdriver is on, and the
  default `'flange'` while the arm is bare.
- `SceneObject` defaults to `is_free = False`; set `bunny.is_free = True` before
  `grasp` so it can be mounted onto the gripper.

## See also

- [`one.robots`](../api/robots.md) — `end_effectors` (grippers, point tools).
- [Grasp planning](grasp_planning.md) — where the grasp poses come from.
- [Kinematics](kinematics.md) — chains and tcps.
