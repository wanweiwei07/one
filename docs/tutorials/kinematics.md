# Kinematics

Every robot in `one` — arm, gripper, hand, or humanoid — inherits a single base,
`MechBase`. They differ only in which **chains** and **tcps** they register:

- A **chain** names a serial run of joints (a `base → tip` link span) that move
  together. A 6-DOF arm registers `'main'`; a humanoid registers `'left_arm'`,
  `'right_arm'`, `'neck'`, …
- A **tcp** (tool center point) is a named frame rigidly attached to a link, e.g.
  `'flange'` on an arm or `'grasp_center'` on a gripper. Look it up by name:
  `robot.tcp('flange')`.

## FK and IK

`robot.fk(qs=...)` drives the joints and updates every link's world transform.
`robot.ik(tgt_pos, tgt_rotmat, chain='main', tcp='flange')` is **position-first**
(ROS Pose order) and returns a list of joint solutions — empty if unreachable,
several when multiple branches reach the target.

This roundtrip example drives FK with known angles, reads back the flange pose,
then asks IK to recover joint configurations and checks the error:

```python
--8<-- "examples/test_ik_roundtrip.py"
```

Things to notice:

- `robot.tcp('flange').tf` is the 4×4 world transform of the flange frame.
- `robot.ik(pos, rotmat)` takes the position first, then the rotation matrix.
- Analytic solvers return *all* branches; a roundtrip should recover the input
  configuration (up to joint-limit / wrap equivalence) with sub-millimetre error.

## Visualizing the mechanism

`KineVisualizer` draws the kinematic **skeleton** (joint axes as stator/rotor
cylinders, links as rods). Pass a `chain` to draw just that chain, or `chain=None`
to draw the whole mechanism — it reads the robot's already-computed per-link
world transforms, so it is correct for branched humanoids as well as serial arms.
Set `robot.alpha` to make the meshes translucent so the skeleton shows through:

```python
--8<-- "examples/test_cvr038_kine_visualizer.py"
```

## See also

- [`one.utils.math`](../api/utils.md) — `rotmat_from_*`, `tf_from_pos_rotmat`,
  `frame_from_normal`, quaternion/euler/rotvec conversions, slerp.
- [`one.robots`](../api/robots.md) — robot classes and `kine_visualizer`.
- [End-effectors & tool change](end_effectors.md) — cross-object IK to a mounted
  gripper's `grasp_center`.
