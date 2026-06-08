# Humanoid (Linx L1)

A humanoid is the same `MechBase` as an arm — it just registers several chains.
The Linx **L1** upper body registers `'left_arm'`, `'right_arm'`,
`'left_arm_waist'`, `'right_arm_waist'`, and `'neck'`. Two classes are provided:

- `L1` — the bare body (arms tip at the `*_arm_link_6` flanges; mount your own
  end-effectors).
- `L1O6` — `L1` with two dexterous Linkerbot **O6** hands pre-mounted on the
  flanges.

Because the hands are mounted children, you reach a target by solving
**cross-object IK** down a body chain to a hand tcp:

```python
robot.ik(pos, rotmat, chain='left_arm', tcp=robot.left_hand.tcp('power_center'))
```

The `'left_arm_waist'` chain adds the waist joint, letting the torso assist the
reach (weight it in the nearest-solution metric to keep the torso still and let
the arm do the work, like a person).

## Pinch-and-place with planned grasps

This example has the L1O6 left hand pick a small bunny using **antipodal** grasps
(not a hand-picked pose). A dexterous hand presents itself to the parallel-jaw
planner via `spawn_jaw('pinch')`, which returns a calibrated jaw view; each grasp
becomes pick-and-place keyframes, and the free-space hops are planned with RRT:

```python
--8<-- "examples/test_l1o6_pinch_bunny.py"
```

Things to notice:

- `hand.spawn_jaw('pinch')` adapts the multi-finger hand into the interface
  `antipodal` expects; each returned grasp is a world pinch-center pose + width.
- Only the `'left_arm_waist'` chain moves; the rest of the body is frozen.
- Set `ONE_HEADLESS=1` to run the planning/IK validation without a window.

## Visualizing the skeleton

`L1`'s own `__main__` renders the body translucent (`robot.alpha = 0.3`) and
overlays the `KineVisualizer` skeleton — which reads per-link world transforms,
so it draws the *branched* humanoid tree correctly (both arms branch from the
waist, not strung in series). See [Kinematics](kinematics.md#visualizing-the-mechanism).

## See also

- [`one.robots`](../api/robots.md) — humanoids and end-effectors.
- [End-effectors & tool change](end_effectors.md) — mounting and cross-object IK.
- [Grasp planning](grasp_planning.md) — antipodal grasps used here.
