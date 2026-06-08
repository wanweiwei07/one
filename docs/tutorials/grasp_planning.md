# Grasp Planning

`one.grasp` has three surface-sampling planners. The naming is systematic — the
**prefix is the contact count, the suffix is the mechanism**:

| Planner | Contacts | Mechanism | Gripper |
|---|---|---|---|
| `antipodal` | 2 | opposing pinch (force closure) | parallel jaw |
| `polypodal` | N | opposing N-point pattern (force closure) | multi-pad / dexterous |
| `monocontact` | 1 | single-side adhesion / press | suction / tip |

All three sample the target surface, build candidate poses, and reject
gripper-vs-target collisions, returning scored `(pose, pre_pose, …)` tuples.

## Antipodal (parallel-jaw)

`antipodal(gripper, target_sobj, ...)` finds 2-point opposing grasps. Each result
is `(pose, pre_pose, jaw_width, score)`; `pose` is the `grasp_center` frame the
gripper's `grip_at` expects.

```python
--8<-- "examples/test_2fg7_antipodal.py"
```

## Monocontact (suction / tip)

`monocontact(tool, target_sobj, tcp='tip', ...)` aligns a single contact axis to
the inward surface normal — for suction cups or a tool tip. Results are
`(pose, pre_pose, score)`; `approach_bias` (default world +z) rewards top-facing
surfaces.

```python
--8<-- "examples/test_orsd_monocontact.py"
```

## Stable placement

Grasping is half the story — `one.grasp.placement` computes the **stable resting
poses** of an object on a flat support (from its convex hull), which is where you
place what you picked up:

```python
--8<-- "examples/test_placement_bunny.py"
```

## See also

- [`one.grasp`](../api/grasp.md) — `antipodal`, `polypodal`, `monocontact`,
  `placement`, and the shared `_common` collision helper.
- [End-effectors & tool change](end_effectors.md) — mounting the gripper and
  solving IK to its `grasp_center`.
