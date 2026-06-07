# `one`

`one` is a self-contained robotics library: **kinematics, scene, mesh geometry,
collision, grasp planning, motion, and a viewer**. It deliberately implements its
own mesh / transform / collision math, so you rarely need to reach for trimesh,
open3d, scipy.spatial.transform, or fcl.

## What's inside

| Need | Module | Alias |
|---|---|---|
| Mesh ops: convex hull, surface sampling, ray–mesh, clip, subdivide, revolve | `one.scene.geometry_ops` | `osgop` |
| Rotations / poses: `rotmat_from_*`, `tf_from_pos_rotmat`, quat/euler/rotvec, slerp | `one.utils.math` | `oum` |
| Mesh file loading (STL/DAE) | `one.geom.loader` | `ogl` |
| Collision detection (CPU/GPU batch) | `one.collider.cpu_simd`, `gpu_simd_batch` | `occs` |
| Grasp planning | `one.grasp.antipodal` / `polypodal` / `monocontact` | `ogab` / `ogpp` / `ogmc` |
| Scene objects / viewer | `one.scene.scene_object`, `one.viewer.world` | `osso` / `ovw` |

Convenience aliases are exported from the package:
`from one import oum, osgop, occs, ouc, ossop, ovw, ogab, ogmc, ...`

See the **[API Index](API_INDEX.md)** for the full, auto-generated module →
function/class map.

## Key conventions

- **pos-first, ROS Pose order**: `oum.tf_from_pos_rotmat(pos, rotmat)`,
  `mech.set_pos_rotmat(pos, rotmat)`,
  `mech.ik(tgt_pos, tgt_rotmat, chain='main', tcp='flange')`.
- **Every robot inherits `MechBase`** (arms, hands, grippers, humanoids alike);
  they differ only in which `chains` / `tcps` are registered. TCPs are looked up
  by name: `mech.tcp('flange')`.
- **Grasp planner naming**: prefix = contact count, suffix = mechanism.
  `antipodal` (2-point opposing pinch), `polypodal` (N-point), `monocontact`
  (1-point suction / press).
- **Dexterous hands** present themselves to the parallel-jaw planner via
  `hand.spawn_jaw('pinch')`, which returns a calibrated, immutably-bound view
  exposing the gripper interface antipodal expects.

## Install

```bash
pip install -e .
```

Run examples with a Python that has pyglet / numpy / scipy (e.g. `py -3.12`);
examples set up a `World` and call `base.run()` for the GUI.

## Regenerating this site's API index

The [API Index](API_INDEX.md) is produced by a static AST scan (no imports, no
hardware side effects):

```bash
py -3.12 tools/gen_api_index.py
```

This runs automatically on every push via the docs workflow, so the published
index never goes stale.
