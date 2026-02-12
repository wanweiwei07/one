# AGENTS.md - Coding Guide for `one`

Guidelines for AI coding agents working on the `one` framework.
The project is a lightweight MuJoCo-based robotics research toolkit.

## Project Snapshot

- Python 3.12+ only; pure Python codebase.
- Core deps: numpy, scipy, mujoco, pyglet.
- Tests are standalone scripts in the repo root.
- No build step or lint pipeline configured.

## Build, Lint, and Test

### Install
```bash
pip install -e .
```

### Run a Single Test
```bash
python test_<name>.py
```

Examples:
```bash
python test_2fg7.py
python test_collider_bunny.py
python test_rrtc_rs007l.py
python test_mujoco_bunny.py
```

### Lint/Format
No formal linter or formatter is wired up.
Match existing style and keep changes minimal.

## Repository Layout

```text
one/
├── collider/        # Collision detection
├── grasp/           # Grasp planning
├── motion/          # Motion planning
├── physics/         # MuJoCo integration
├── robots/          # Robot models
├── scene/           # Scene graph/rendering
├── utils/           # Math/constants/helpers
└── viewer/          # Pyglet viewer
```

## Code Style

### Imports
- Order: standard library, third-party, then local.
- Prefer canonical abbreviations from `one/__init__.py`.
- Avoid wildcard imports; import specific modules.

Canonical aliases:

```python
import one.utils.math as oum
import one.utils.helper as ouh
import one.utils.constant as ouc
import one.scene.scene_object as osso
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
import one.collider.mj_collider as ocm
import one.motion.probabilistic.space_provider as ompsp
import one.motion.probabilistic.rrt as ompr
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
- Always pass `dtype=np.float32` for numeric arrays.

```python
import numpy.typing as npt

def gen_points(...) -> npt.NDArray:
    pts = np.zeros((n, 3), dtype=np.float32)
    return pts
```

### Docstrings
- Use short docstrings with params/return descriptions.
- For complex math, include author/date if pattern exists.

### Error Handling
- Return `None` for expected failure cases.
- Raise `ValueError` for invalid inputs.
- Avoid broad exception catching.

### NumPy Practices
- Prefer vectorized operations over Python loops.
- Use `np.einsum` for batched math when appropriate.
- Use unit-vector guard patterns to avoid division by zero.

```python
length = np.linalg.norm(vec)
unit = vec / length if length > 0 else np.zeros_like(vec)
```

### Scene and Transform Patterns
- Use 4x4 homogeneous matrices for poses.
- Helpers: `oum.tf_from_rotmat_pos`, `oum.tf_inverse`.
- Attach objects to `base.scene`; call `base.run()`.

### Collision Detection Pattern
- Use the CPU SIMD path by default; GPU is optional.
```python
from one.collider import cpu_simd

hit_points = cpu_simd.is_sobj_collided(obj_a, obj_b)
if hit_points is not None:
    pass
```

### Motion Planning Pattern
```python
planner = ompr.RRTConnectPlanner(ssp_provider, step_size=np.pi / 36)
path = planner.solve(start, goal, max_iters=1000)
```

### Testing/Examples
- Keep tests in repo root as `test_<feature>.py`.
- Tests are also examples; keep them runnable and readable.
- Use visualization (`ovw.World`) when needed.

## Cursor/Copilot Rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` found.

## General Guidance
- Keep code simple; clarity over cleverness.
- Use constants from `one.utils.constant`.
- Prefer small, composable functions.
- When modifying existing modules, follow local style.
