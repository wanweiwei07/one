# `one` — project guide

`one` is a self-contained robotics library (kinematics, scene, mesh geometry,
collision, grasp planning, motion, viewer). It deliberately implements its own
mesh / transform / collision math.

## House rule: prefer in-house utilities

**Before importing a third-party library for mesh, transform, collision, or
grasp work, check the in-house equivalent first.** `one` already has these, and
mixing in trimesh / open3d / scipy.spatial.transform / fcl creates redundant,
inconsistent code. When unsure what exists, consult **[docs/API_INDEX.md](docs/API_INDEX.md)**
(auto-generated module → function map) or `grep` the modules below.

| Need | Use in `one` | Don't reach for |
|---|---|---|
| Mesh ops: convex hull, surface sampling, ray–mesh, clip, subdivide, revolve | `one.scene.geometry_ops` (`osgop`) | trimesh, open3d |
| Rotations / poses: `rotmat_from_*`, `tf_from_pos_rotmat`, quat/euler/rotvec, slerp, `frame_from_normal` | `one.utils.math` (`oum`) | scipy.spatial.transform, transforms3d |
| Mesh file loading (STL/DAE) | `one.geom.loader` (`ogl`) | trimesh.load |
| Collision detection (CPU/GPU batch) | `one.collider.cpu_simd` (`occs`), `gpu_simd_batch` | fcl, trimesh.collision |
| Grasp planning | `one.grasp` — `antipodal` (`ogab`), `polypodal` (`ogpp`), `monocontact` (`ogmc`), `placement` (`ogpl`) | — |
| Constants / colors / axes | `one.utils.constant` (`ouc`) | — |
| Scene objects / viewer | `one.scene.scene_object` (`osso`), `scene_object_primitive` (`ossop`), `one.viewer.world` (`ovw`) | — |

Convenience aliases are exported from the package: `from one import oum, osgop, occs, ouc, ossop, ovw, ogab, ogmc, ...` (see `one/__init__.py`).

## Key API conventions

- **pos-first**, ROS Pose order: `oum.tf_from_pos_rotmat(pos, rotmat)`,
  `mech.set_pos_rotmat(pos, rotmat)`, `mech.ik(tgt_pos, tgt_rotmat, chain='main', tcp='flange')`.
- **Robots** all inherit `MechBase` (arms, hands, grippers, humanoids alike);
  differences are which `chains`/`tcps` are registered. Grippers add
  `GripperMixin` + a `'grasp_center'` tcp; point tools add `PointMixin` + a
  `'tip'` tcp. tcps are looked up by string name: `mech.tcp('flange')`.
- **Grasp planner naming**: suffix = mechanism, prefix = contact count.
  `-podal` = opposing pinch / force-closure (`antipodal` 2-pt, `polypodal` N-pt);
  `-contact` = single-side adhesion / press (`monocontact` 1-pt suction/tip).
- **Collision**: `ocm.MJCollider().is_collided(qs)` is the high-level robot
  self/scene check used by motion planning; `occs` (`cpu_simd`) is the low-level
  mesh-vs-mesh SIMD path, `gpu_simd_batch` the batched GPU variant.
- **Motion planning**: build an `omppc.PlanningContext(collider=mjc, cd_step_size=...)`,
  pass it to `ompr.RRTConnectPlanner(pln_ctx=ctx, extend_step_size=...)`, then
  `planner.solve(start, goal, max_iters=...)` (returns `None` if no path).

## Environment

- Run with **`py -3.12`** (has pyglet/numpy/scipy/trimesh). The default `python`
  is 3.14 and lacks pyglet. Examples set up a `World` and call `base.run()`
  (GUI) — for headless checks, import modules and call the planning/IK functions
  directly.
- Tests / examples live under **`examples/`** (`py -3.12 examples/test_<name>.py`);
  several ship a `*_nogui.py` headless variant.

## Keeping the index fresh

After adding/removing public functions, regenerate the map:

```
py -3.12 tools/gen_api_index.py
```
