# AGENTS.md - Coding Guide for `one`

Guidelines for AI coding agents working on the `one` framework — a
self-contained robotics research toolkit (kinematics, scene, mesh geometry,
collision, grasp planning, motion, viewer) with optional MuJoCo physics.

This file is the cross-tool guide; the Claude-specific house rules live in
[CLAUDE.md](CLAUDE.md). Read both.

## Project Snapshot

- Python 3.12 only. Run with **`py -3.12`** — the default `python` is 3.14 and
  lacks pyglet. Pure-Python codebase.
- Core deps: numpy, scipy, mujoco, pyglet, trimesh.
- Tests/examples are standalone scripts under `examples/`.
- No build step or lint pipeline configured.

## House rule: prefer in-house utilities

`one` deliberately implements its own mesh / transform / collision / grasp math.
**Before importing trimesh / open3d / scipy.spatial.transform / fcl, check the
in-house equivalent first.** When unsure what exists, consult
[docs/API_INDEX.md](docs/API_INDEX.md) (auto-generated module → function map) or
`grep` the modules. See [CLAUDE.md](CLAUDE.md) for the full need → module table.

## Build, Lint, and Test

### Install
```bash
py -3.12 -m pip install -e .
```

### Run a Single Test/Example
Examples live in `examples/` and double as tests. They set up a `World` and call
`base.run()` (GUI); for headless checks import the module and call the
planning/IK functions directly.
```bash
py -3.12 examples/test_2fg7.py
py -3.12 examples/test_collider_bunny.py
py -3.12 examples/test_collision_debug_nogui.py   # headless variant
py -3.12 examples/test_mujoco_bunny.py
```

### Regenerate the API index
After adding/removing public functions:
```bash
py -3.12 tools/gen_api_index.py
```

### Lint/Format
No formal linter or formatter is wired up. Match existing style and keep changes
minimal.

## Repository Layout

```text
one/
├── collider/        # Collision: cpu_simd (occs), gpu_simd_batch, mj_collider (MJCollider)
├── devices/         # Hardware device interfaces
├── geom/            # Mesh geometry (ogg) + STL/DAE loader (ogl)
├── grasp/           # Grasp planning: antipodal, polypodal, monocontact, placement
├── motion/          # Motion planning: probabilistic (rrt/prm) + trajectory
├── physics/         # MuJoCo integration
├── robots/          # Robot models (manipulators, end_effectors, vehicle)
├── scene/           # Scene graph, scene objects, geometry_ops (osgop), rendering
├── stream/          # Streaming/data pipelines
├── utils/           # Math (oum), constants (ouc), helpers (ouh)
└── viewer/          # Pyglet viewer / World (ovw)
```

## Code Style

### Imports
- Order: standard library, third-party, then local.
- Prefer the canonical abbreviations exported from `one/__init__.py`.
- Avoid wildcard imports; import specific modules.

Canonical aliases (kept in sync with `one/__init__.py`):

```python
import one.utils.math as oum
import one.utils.helper as ouh
import one.utils.constant as ouc
import one.geom.geometry as ogg
import one.geom.loader as ogl
import one.scene.scene as oss
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.scene.render_model as osrm
import one.scene.geometry_ops as osgop
import one.viewer.world as ovw
import one.collider.mj_collider as ocm
import one.collider.cpu_simd as occs
import one.grasp.antipodal as ogab
import one.grasp.polypodal as ogpp
import one.grasp.monocontact as ogmc
import one.grasp.placement as ogpl
import one.motion.probabilistic.planning_context as omppc
import one.motion.probabilistic.rrt as ompr
import one.motion.probabilistic.prm as ompp
import one.motion.trajectory.cartesian as omtr
import one.motion.trajectory.time_param as omttp
```

### Naming
- Files/modules: `snake_case`.
- Classes: `PascalCase`.
- Functions/vars: `snake_case`.
- Constants: `UPPER_CASE` (prefer `one.utils.constant`).
- Private attrs: leading underscore.

### Formatting
- Indent with 4 spaces.
- Keep lines ~100-120 chars when practical.
- Prefer single quotes; keep string style consistent in file.
- Add spaces around operators and after commas.
- Use numpy-style aligned matrices when multi-line.
- Follow [docs/COMMENT_STYLE.md](docs/COMMENT_STYLE.md) for comments and
  docstrings.

```python
rotmat = np.array([[aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac), 0],
                   [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab), 0],
                   [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
```

### Types and Arrays
- Type hints are minimal; use when helpful.
- Use `numpy.typing` for array annotations.
- Document array shapes in docstrings `(N,3)` or `(4,4)`.
- Prefer `dtype=np.float32` for numeric arrays.

### Error Handling
- Return `None` for expected failure cases.
- Raise `ValueError` for invalid inputs.
- Avoid broad exception catching.

### NumPy Practices
- Prefer vectorized operations over Python loops.
- Use `np.einsum` for batched math when appropriate.
- Guard unit-vector division by zero.

```python
length = np.linalg.norm(vec)
unit = vec / length if length > 0 else np.zeros_like(vec)
```

## Key API conventions

- **pos-first**, ROS Pose order. Poses are 4×4 homogeneous matrices.
  ```python
  tf = oum.tf_from_pos_rotmat(pos, rotmat)      # pos first, then rotmat
  inv = oum.tf_inverse(tf)
  mech.set_pos_rotmat(pos, rotmat)
  q = mech.ik(tgt_pos, tgt_rotmat, chain='main', tcp='flange')
  ```
- **Robots** all inherit `MechBase`; differences are which `chains`/`tcps` are
  registered. Grippers add `GripperMixin` + a `'grasp_center'` tcp; point tools
  add `PointMixin` + a `'tip'` tcp. tcps are looked up by name: `mech.tcp('flange')`.
- **Grasp planner naming**: prefix = contact count, suffix = mechanism.
  `-podal` = opposing pinch / force-closure (`antipodal` 2-pt, `polypodal` N-pt);
  `-contact` = single-side adhesion / press (`monocontact` 1-pt suction/tip).

### Collision Detection Pattern
`MJCollider` is the high-level robot self/scene collision check used by motion
planning; `cpu_simd` (`occs`) is the low-level mesh-vs-mesh SIMD path.
```python
mjc = ocm.MJCollider()
if mjc.is_collided(qs):          # qs = joint configuration
    ...
```

### Motion Planning Pattern
```python
ctx = omppc.PlanningContext(collider=mjc, cd_step_size=np.pi / 180)
planner = ompr.RRTConnectPlanner(pln_ctx=ctx, extend_step_size=np.pi / 36)
path = planner.solve(start=q0, goal=q1, max_iters=3000)   # None if no path found
```

### Scene / Viewer Pattern
- Attach objects to `base.scene`; call `base.run()` to launch the GUI.
- Use `ovw.World` for visualization.

## Testing / Examples

- Keep new tests/examples in `examples/` as `test_<feature>.py`.
- Tests are also examples; keep them runnable and readable.
- Provide a `*_nogui.py` headless variant when a test needs to run in CI/headless.

## General Guidance

- Keep code simple; clarity over cleverness.
- Use constants from `one.utils.constant`.
- Prefer small, composable functions.
- When modifying existing modules, follow local style.
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` in use.
