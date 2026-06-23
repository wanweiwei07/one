# Grasp Planning

`one.grasp` has three surface-sampling planners. The naming is systematic — the
**prefix is the contact count, the suffix is the mechanism**:

| Planner | Contacts | Mechanism | Gripper |
|---|---|---|---|
| `antipodal` | 2 | opposing pinch (force closure) | parallel jaw |
| `polypodal` | N | opposing N-point pattern (force closure) | multi-pad / dexterous |
| `monocontact` | 1 | single-side adhesion / press | suction / tip |

All three sample the target surface, build candidate poses, and reject
gripper-vs-target collisions, returning a list of scored
[`Grasp`](../api/grasp.md) records.

A `Grasp` is **self-contained and gripper-agnostic**: it freezes the grasp
`pose` / `pre_pose` (the tcp frame in the object's local coords), the `tcp`
loc_tf relative to the hand root, and the hand `qpos` / `pre_qpos` — so replay
needs no re-derivation from the gripper's live state. `g.make_tcp(gripper)`
rebuilds the IK tcp; `g.transformed(obj.wd_tf)` maps it into the world;
`g.provenance` carries optional, non-authoritative metadata (`jaw_width`,
`mode`, …). See [`one.grasp.grasp.Grasp`](../api/grasp.md) for the full record.

## Antipodal (parallel-jaw)

`antipodal(gripper, target_sobj, ...)` finds 2-point opposing grasps. Each result
is a `Grasp` whose `pose` is the `grasp_center` frame the gripper's `grip_at`
expects (its `jaw_width` is in `g.provenance`).

```python
--8<-- "examples/test_2fg7_antipodal.py"
```

## Monocontact (suction / tip)

`monocontact(tool, target_sobj, tcp='tip', ...)` aligns a single contact axis to
the inward surface normal — for suction cups or a tool tip. Results are `Grasp`
records (no opposing closure, so `qpos` is the tool's fixed config and the
`tcp_name` is in `g.provenance`); `approach_bias` (default world +z) rewards
top-facing surfaces.

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
