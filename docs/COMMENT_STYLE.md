# Comment Style

This project uses comments and docstrings as part of the engineering contract:
they should clarify geometry, transforms, units, array shapes, and failure
behavior without repeating obvious Python syntax.

## Language

- Use English for new comments and docstrings.
- Keep existing Chinese comments until the surrounding code is being touched;
  translate them only when the meaning is clear.
- Prefer concise technical wording over long prose.

## Docstrings

Use NumPy-style docstrings for public APIs, complex private helpers, and
mathematical routines whose behavior is not obvious from the implementation.
Short getters, simple setters, and tiny local helpers do not need docstrings
unless they encode an important convention.

Document the following when relevant:

- Array shapes, such as `(N, 3)`, `(M, 3)`, `(4, 4)`, or `(B, N, 3)`.
- Coordinate frames, pose order, and transform conventions.
- Units, especially radians, meters, and normalized vectors.
- `dtype` expectations when the function depends on them.
- Expected failure cases, especially functions that return `None`.
- Side effects, mutation, caching, or in-place behavior.

Example:

```python
def tf_from_pos_rotmat(pos, rotmat):
    """
    Build a homogeneous transform from position and rotation matrix.

    Parameters
    ----------
    pos : array-like, shape (3,)
        Translation in world coordinates.
    rotmat : array-like, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    np.ndarray, shape (4, 4)
        Homogeneous transform matrix.
    """
```

For functions that can fail as part of normal operation, make that explicit:

```python
def solve(start, goal, max_iters=3000):
    """
    Plan a joint-space path from start to goal.

    Returns
    -------
    list[np.ndarray] or None
        Joint configurations from start to goal if planning succeeds,
        otherwise None.
    """
```

## Module Docstrings

Use a short module docstring when a file exposes reusable behavior or a
non-obvious convention. A good module docstring usually says what the module
owns and which data conventions callers should know.

Example:

```python
"""
Mesh geometry helpers.

This module provides low-level mesh operations used by scene objects,
collision checks, and grasp planning. Mesh arrays generally use float32
vertices with shape (N, 3) and int triangle indices with shape (M, 3).
"""
```

## Inline Comments

Inline comments should explain intent, constraints, or edge cases. Avoid
comments that only restate the next line of code.

Good:

```python
# Avoid division by zero for degenerate normals.
normals = normals / np.maximum(lengths[:, None], 1e-8)
```

Avoid:

```python
# Loop over points.
for point in points:
    ...
```

Use inline comments sparingly around:

- Degenerate geometry handling.
- Numerical tolerances and thresholds.
- Coordinate-frame conversions.
- MuJoCo, OpenGL, or robot-model conventions.
- Performance-sensitive vectorization.

## Conventions To Spell Out

The project has several conventions that are easy to misuse. Mention them in
docstrings when a function's API depends on them:

- Poses are 4x4 homogeneous matrices.
- Position comes first: `tf_from_pos_rotmat(pos, rotmat)`.
- Rotation matrices are 3x3.
- Joint-space values are usually radians.
- Mesh vertices are generally `(N, 3)` arrays.
- Triangle indices are generally `(M, 3)` arrays.
- Expected planner or IK failure should return `None`.

## Migration

When normalizing old comments, keep the diff focused:

1. Update one module or one subpackage at a time.
2. Do not refactor behavior just to improve comments.
3. Preserve comments that contain uncertain domain knowledge until verified.
4. Run the closest example or headless check after touching executable code.
5. Regenerate `docs/API_INDEX.md` only when public functions or classes change.
