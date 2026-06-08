# Quick Start

Three small, runnable examples. Each lives under `examples/` and is embedded
here verbatim — run it with `py -3.12 examples/<name>.py`.

Two import styles appear throughout the codebase and are interchangeable:

```python
from one import oum, ovw, osso, ossop, khi_rs007l   # convenience aliases
import one.utils.math as oum                          # or the full path
```

The aliases are exported from `one/__init__.py` (`oum` = `one.utils.math`,
`osgop` = `one.scene.geometry_ops`, `ovw` = `one.viewer.world`, …).

## Spawn a robot and view it

A `World` opens the viewer; `attach_to(base.scene)` makes an object visible;
`robot.fk(qs=...)` drives the joints; `robot.clone()` makes an independent copy.
`base.schedule_interval(fn, dt)` runs `fn` every `dt` seconds, and `base.run()`
starts the loop.

```python
--8<-- "examples/test_rs007l_spawn.py"
```

## Solve inverse kinematics

`robot.ik(tgt_pos, tgt_rotmat)` is **position-first** (ROS Pose order) and
returns a list of joint solutions (empty if unreachable). It defaults to
`chain='main'`, `tcp='flange'`. Here every solution is cloned and drawn, so you
see the IK branches fan out.

```python
--8<-- "examples/test_rs007l_ik.py"
```

## Load and show a mesh

`SceneObject.from_file` reads a mesh (STL/DAE) via the in-house loader. A
`collision_type` (`MESH`, `CAPSULE`, …) builds the collision proxy;
`toggle_render_collision` shows it. `.pos` / `.rotmat` place the object, and
`.clone()` duplicates it.

```python
--8<-- "examples/test_bunny.py"
```

## Where to next

- [Kinematics](tutorials/kinematics.md) — chains, tcps, FK/IK, the skeleton viewer.
- [Scene & Geometry](tutorials/scene_geometry.md) — meshes, convex hull, sampling.
- [Grasp planning](tutorials/grasp_planning.md) — antipodal / monocontact / placement.
- [Motion planning](tutorials/motion_planning.md) — RRT-Connect, PRM, Cartesian.
- [API reference](api/index.md).
