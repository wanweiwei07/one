# AGENTS.md - Coding Guide for `one`

This document provides guidelines for AI coding agents working on the `one` framework - a lightweight robot motion planning and learning framework built on MuJoCo.

## Project Overview

`one` is a Python 3.12+ research framework for robot motion planning and learning with:
- MuJoCo-native collision detection and physics simulation
- Custom visualization built on pyglet
- Support for both URDF and MJCF robot descriptions
- Modular design: collider, grasp, motion, physics, robots, scene, utils, viewer

## Build, Lint, and Test Commands

### Installation
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.12+
- Core: numpy, scipy, pyglet, mujoco
- No external collision libraries (uses MuJoCo native)

### Running Tests
There is no formal test framework (pytest/unittest) configured. Tests are standalone Python scripts:

```bash
# Run a single test/demo script
python test_<name>.py

# Examples:
python test_2fg7.py
python test_cpusimd_collider_bunny.py
python test_rrtc_rs007l.py
python test_mujoco_bunny.py
```

Test files are located in the repository root and demonstrate specific features or components.

### No Build Step
This is a pure Python package with no build/compile step required.

### Linting/Formatting
No formal linter configuration (pylint, flake8, black) is present. Follow the code style patterns observed in the codebase.

## Code Style Guidelines

### Import Conventions

**Standard library first, then third-party, then local imports:**
```python
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
import one.utils.math as oum
import one.utils.constant as ouc
```

**Use abbreviated import aliases consistently:**
- `import one.utils.math as oum`
- `import one.utils.helper as ouh`
- `import one.utils.constant as ouc`
- `import one.scene.scene as oss`
- `import one.viewer.world as ovw`
- `import one.collider.mj_collider as ocm`
- etc.

See `one/__init__.py` for canonical abbreviations.

### Naming Conventions

**Files and modules:** `snake_case`
- `cpu_simd.py`, `mj_collider.py`, `space_provider.py`

**Classes:** `PascalCase`
- `RRTTree`, `RRTConnectPlanner`, `MJEnv`, `SceneObject`

**Functions and methods:** `snake_case`
- `is_sobj_collided()`, `build_triangles()`, `compute_mesh_aabb()`

**Variables:** `snake_case`
- `tris_a`, `hit_points`, `max_radius`, `edge_length`

**Constants:** `UPPER_CASE` (in `one.utils.constant`)
- Classes as namespaces: `BasicColor.RED`, `CollisionType.MESH`

**Private attributes:** Prefix with single underscore
- `self._sspp`, `self._cvter`, `self._ssnp`

### Type Hints

Type hints are **minimal** in this codebase. Use numpy typing where helpful:
```python
import numpy.typing as npt

def gen_2d_spiral_points(
    max_radius: float = .002,
    radial_granularity: float = .0001,
    tangential_granularity: float = .0003,
    toggle_origin: bool = False
) -> npt.NDArray:
```

For most functions, rely on docstrings for type documentation.

### Formatting

**Line length:** No strict limit, but keep reasonable (~100-120 chars preferred)

**Indentation:** 4 spaces

**String quotes:** Single quotes `'` preferred, but double quotes `"` acceptable

**Array/matrix alignment:** Use numpy-style formatting:
```python
rotmat = np.array([[aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac), 0],
                   [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab), 0],
                   [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc, 0],
                   [0, 0, 0, 1]], dtype=np.float32)
```

**Whitespace:** Space around operators, after commas

### Docstrings

Use simple docstrings with parameter descriptions:
```python
def tripair_planeprojection_filter(tris_a, tris_b, eps=1e-4):
    """
    Find first intersecting triangle pair.
    tris_a, tris_b: (N,3,3)
    return: (hit_found, (ia, ib)) or (False, None)
    """
```

For complex functions, include author/date:
```python
def rotmat_from_normalandpoints(facet_normal, facet_first_pnt, facet_second_pnt):
    '''
    Compute the rotation matrix of a 3D facet using
    facet_normal and the first two points on the facet
    :param facet_normal: 1x3 nparray
    :param facet_first_pnt: 1x3 nparray
    :param facet_second_pnt: 1x3 nparray
    :return: 3x3 rotmat
    date: 20160624
    author: weiwei
    '''
```

### NumPy Usage

**Always specify dtype for consistency:**
```python
np.array([x, y, z], dtype=np.float32)
np.zeros(3, dtype=np.float32)
np.eye(4, dtype=np.float32)
```

**Use vectorized operations:**
```python
# Good - vectorized
dists = state_space.vectorized_distance(state, self._ssnp)

# Avoid loops where vectorization is possible
```

**Common patterns:**
```python
# Unit vector calculation
length = np.linalg.norm(vec)
unit = vec / length if length > 0 else np.zeros_like(vec)

# Einsum for batched operations
dist_to_plane_a = np.einsum("nij,nj->ni", tris_b, normals_a) + offsets_a[:, None]
```

## Architecture Patterns

### Module Structure
```
one/
├── collider/        # Collision detection (CPU SIMD, GPU, MuJoCo, raycast)
├── grasp/          # Grasp planning (antipodal grasps)
├── motion/         # Motion planning (RRT, PRM, chain planners)
├── physics/        # MuJoCo physics (env, contact, compiler, naming)
├── robots/         # Robot models (manipulators, end effectors, vehicles)
├── scene/          # Scene management (geometry, objects, rendering)
├── utils/          # Utilities (math, constants, helpers)
└── viewer/         # Visualization (world viewer)
```

### Scene and Visualization
- Use `ovw.World` for creating visualization environments
- Attach objects to `base.scene`
- Call `base.run()` to start interactive loop

### Error Handling

**Return None for failure cases rather than exceptions:**
```python
def is_sobj_collided(sobj_a, sobj_b, eps=1e-9):
    # ... checks ...
    if not aabb_intersect(min_a, max_a, min_b, max_b):
        return None  # No collision
    # ... more checks ...
    return points[valid]  # Collision points
```

**Raise exceptions for invalid inputs:**
```python
if np.linalg.norm(d1) < eps or np.linalg.norm(d2) < eps:
    raise ValueError("Direction vector cannot be zero!")
```

## Common Patterns

### Transformation Matrices
Use 4x4 homogeneous matrices for poses:
```python
tf = oum.tf_from_rotmat_pos(rotmat, pos)
inv_tf = oum.tf_inverse(tf)
transformed = oum.transform_points_by_tf(tf, points)
```

### Collision Detection
```python
from one.collider import cpu_simd

hit_points = cpu_simd.is_sobj_collided(obj_a, obj_b)
if hit_points is not None:
    # Handle collision
```

### Motion Planning
```python
planner = ompr.RRTConnectPlanner(ssp_provider, step_size=np.pi/36)
path = planner.solve(start, goal, max_iters=1000)
```

## Testing Practices

- Write standalone test scripts in the root directory
- Name tests: `test_<feature>.py`
- Include visualization in tests using `ovw.World` when helpful
- Test files serve as examples and documentation

## Best Practices

1. **Use typed arrays consistently:** Always specify `dtype=np.float32` for consistency
2. **Vectorize operations:** Prefer numpy vectorized ops over Python loops
3. **Document shape contracts:** Annotate array shapes in docstrings `(N,3)`, `(4,4)`
4. **Fail gracefully:** Return `None` for expected failures, raise for programmer errors
5. **Use constants:** Import from `one.utils.constant` rather than hardcoding
6. **Follow abbreviation patterns:** Use canonical module abbreviations from `one/__init__.py`
7. **Keep it simple:** This is a research framework - clarity over complexity
