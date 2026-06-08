# Installation

## Requirements

- **Python 3.12** is the supported interpreter. On Windows, invoke it as
  `py -3.12` — the bare `python` may point at a newer build that lacks a
  working `pyglet` wheel.
- A working OpenGL stack for the viewer. Running the GUI examples (`base.run()`)
  needs a display; the planning/geometry/IK code itself runs headless.

Core dependencies (declared in `pyproject.toml`, installed automatically):
`numpy`, `scipy`, `pyglet`, and the mesh/URDF tooling used by the loaders.

## Install

From the repository root, an editable install picks up local changes without
reinstalling:

```bash
py -3.12 -m pip install -e .
```

## Verify

A headless smoke test — imports the package and solves one inverse-kinematics
query, no window required:

```bash
py -3.12 -c "import numpy as np; from one import khi_rs007l; \
r = khi_rs007l.RS007L(); \
print('IK solutions:', len(r.ik((0.4, 0.0, 0.3), np.eye(3, dtype=np.float32))))"
```

If that prints a non-zero number of solutions, the kinematics stack is wired up.
To check the viewer, run any GUI example:

```bash
py -3.12 examples/test_rs007l_spawn.py
```

## Next

- [Quick Start](quickstart.md) — spawn a robot, view a mesh, solve IK.
- [Tutorials](tutorials/kinematics.md) — topic-by-topic walkthroughs.
- [API reference](api/index.md) — every module's public functions and classes.
