# one

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`one` is a lightweight robot motion planning and learning framework with a strong emphasis on simplicity, consistency, and controllability.

The framework depends solely on MuJoCo and does not rely on external collision libraries. For kinematic planning, collision checking is performed using MuJoCo's native pipeline (mj_kinematics + mj_collision). For dynamic simulation, MuJoCo's standard time stepping (mj_step) is used directly.

Visualization is implemented from scratch on top of pyglet, providing full control from low-level rendering primitives such as device buffers, shaders, and render passes to high-level abstractions including scenes, kinematic chains, and articulated mechanisms. Users can flexibly intervene at any layer depending on their needs.

The internal data structures are designed to be compatible with both URDF and MJCF XML descriptions, enabling unified handling of robot models across different representations without forcing premature conversions.

Overall, `one` is intended as a lightweight yet expressive research framework for studying motion planning, learning, and their interaction under a single, coherent simulation backend.


<table style="width:100%; border:none; border-collapse:collapse;">
<tr style="border:none;">
<td style="width:60%; border:none; padding:0; vertical-align:top;">

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Visualizing a Robot](#visualizing-a-robot)
  - [Collision Detection](#collision-detection)
  - [Motion Planning](#motion-planning)
- [Project Structure](#project-structure)
- [Running Examples](#running-examples)
- [Supported Robots](#supported-robots)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

</td>
<td style="width:40%; border:none; padding:0; text-align:center; vertical-align:middle;">
<img src="grasp.gif" width="720">
</td>
</tr>
</table>

## Features

### Collision Detection
- **MuJoCo-native collision checking** - No external collision libraries required
- **CPU SIMD accelerated detection** - Fast triangle-triangle intersection tests (~3ms per check)
- **GPU collision support** - OpenGL compute shader acceleration (~0.3ms per check), shares device buffers with rendering for zero-copy performance
- **Automatic CPU fallback** - Seamlessly degrades to CPU SIMD when OpenGL compute shaders are unavailable
- **Raycasting capabilities** - Ray-mesh intersection utilities

### Motion Planning
- **RRT/RRT-Connect algorithms** - Rapidly-exploring random tree planners
- **Probabilistic Roadmap (PRM)** - Multi-query motion planning
- **State space providers** - Integrated collision checking during planning
- **Path post-processing** - Shortcutting and densification

### Grasp Planning
- **Antipodal grasp generation** - Surface sampling and contact analysis
- **Force closure evaluation** - Grasp quality metrics
- **Ray-based contact detection** - Efficient grasp candidate generation

### Physics Simulation
- **Direct MuJoCo integration** - Native mj_step and mj_collision APIs
- **Contact force visualization** - Real-time contact force rendering
- **MJCF and URDF compilation** - Unified robot model handling
- **Free-floating and fixed-base support** - Flexible robot configurations

### Visualization
- **Custom pyglet-based renderer** - Built from the ground up for robotics
- **Low to high-level control** - Access buffers, shaders, or use scene abstractions
- **Interactive 3D world viewer** - Camera controls and real-time rendering
- **Scene graph management** - Hierarchical object organization

### Robot Models
- **Industrial manipulators** - Kawasaki RS007L, Denso Cobotta
- **End effectors** - OnRobot 2FG7 parallel jaw gripper
- **Mobile robots** - xy-theta vehicle models
- **Extensible design** - Easy to add new robots via URDF/MJCF

## Installation

### Prerequisites
- **Python**: 3.12 or later
- **Package manager**: `pip`

It is recommended to use a virtual environment (e.g., `venv`, `conda`, or similar) to avoid dependency conflicts.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/wanweiwei07/one.git
   cd one
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```
   This installs the current repository as an editable site package in your environment, so changes in this folder are immediately reflected without reinstalling.

### Dependencies Explained

- **numpy** - Core numerical operations, array handling, and linear algebra
- **scipy** - Spatial transformations (rotations, interpolations) and scientific computing utilities
- **mujoco** - Physics simulation backend and native collision detection pipeline
- **pyglet** - OpenGL-based visualization, windowing, and event handling

## Quick Start

### Visualizing a Robot

```python
import numpy as np
from one import ovw, ouc, ossop, or_2fg7

# Create a 3D world with camera settings
base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .2),
                 toggle_auto_cam_orbit=True)

# Add coordinate frame
oframe = ossop.frame().attach_to(base.scene)

# Load and display a gripper
gripper = or_2fg7.OR2FG7()
gripper.attach_to(base.scene)
gripper.fk()

# Create a cylinder object
box = ossop.cylinder(spos=(.3, 0, 0), epos=(.3, 0, .1), radius=.03,
                     collision_type=ouc.CollisionType.AABB,
                     is_free=True)
box.attach_to(base.scene)

# Compute grasp
gripper.grasp(box)

# Start interactive viewer
base.run()
```

### Collision Detection

This example demonstrates CPU SIMD-accelerated collision detection between two mesh objects:

```python
import builtins
import numpy as np
import time
from one import ovw, ouc, osso, ossop
from one.collider import cpu_simd

# Create visualization world
base = ovw.World(cam_pos=(.5, .5, .5), cam_lookat_pos=(0, 0, .1),
                 toggle_auto_cam_orbit=True)
builtins.base = base

# Load two mesh objects
bunny1 = osso.SceneObject.from_file(
    "bunny.stl",
    collision_type=ouc.CollisionType.MESH)
bunny1.alpha = .5

bunny2 = osso.SceneObject.from_file(
    "bunny.stl",
    collision_type=ouc.CollisionType.MESH)
bunny2.alpha = .3

# Position them to potentially collide
bunny1.set_rotmat_pos(rotmat=np.eye(3), pos=np.array([0.0, 0.0, 0.0]))
bunny2.set_rotmat_pos(rotmat=np.eye(3), pos=np.array([0.03, 0.05, 0.0]))

# Check for collisions using CPU SIMD
tic = time.time()
hit_points = cpu_simd.is_sobj_collided(bunny1, bunny2)
toc = time.time()
print(f"Collision check time: {toc - tic} seconds")

# Visualize objects
bunny1.attach_to(base.scene)
bunny2.attach_to(base.scene)

# Visualize collision points as red spheres
if hit_points is not None:
    for hit_point in hit_points:
        s = ossop.sphere(
            pos=hit_point, radius=0.002,
            rgb=ouc.BasicColor.RED, alpha=ouc.ALPHA.SOLID,
            collision_type=None, is_free=False)
        s.attach_to(base.scene)

base.run()
```

#### GPU Collision Support

The framework provides GPU-accelerated collision detection using OpenGL compute shaders (OpenGL 4.3+). The GPU implementation achieves **~10× speedup** over CPU SIMD by:

- **Zero-copy optimization**: Shares device buffers with the rendering pipeline, eliminating CPU↔GPU memory transfers
- **Parallel processing**: Leverages compute shaders for massively parallel triangle-triangle tests
- **No external dependencies**: Uses only OpenGL (already required for visualization)

**Performance comparison** (typical collision check):
- **GPU collision**: ~0.3ms per check
- **CPU SIMD**: ~3ms per check

**Automatic fallback**: If OpenGL compute shaders are unavailable (e.g., older hardware, missing drivers), the system automatically falls back to CPU SIMD without code changes.

**Usage:**

```python
from one.collider import tbd_collider

# High-level API (GPU-first with automatic CPU fallback)
hit_points = collider.is_collided(obj1, obj2, max_points=200)

# Or use specific backend explicitly
from one.collider import gpu_simd_batch, cpu_simd

hit_points = gpu_simd.is_sobj_collided(obj1, obj2)  # GPU only
hit_points = cpu_simd.is_sobj_collided(obj1, obj2)  # CPU only
```

Run `python benchmark_collider.py` to measure performance on your hardware.

### Motion Planning

This example shows RRT-Connect path planning for a 6-DOF manipulator with obstacles:

```python
import builtins
import numpy as np
from one import oum, ovw, ouc, ossop, ocm, ompsp, ompr, khi_rs007l

# Setup world and robot
base = ovw.World(cam_pos=(-2, 2, 2), cam_lookat_pos=(0, 0, 0.5),
                 toggle_auto_cam_orbit=False)
builtins.base = base

oframe = ossop.frame()
oframe.attach_to(base.scene)

robot = khi_rs007l.RS007L()
robot.is_free = True
robot.rotmat = oum.rotmat_from_euler(0, 0, -oum.pi / 2)
robot.attach_to(base.scene)

# Add obstacles
box = ossop.box(half_extents=(1, .01, .15), pos=(.0, -0.3, 1),
                collision_type=ouc.CollisionType.AABB)
box.attach_to(base.scene)

box2 = ossop.box(half_extents=(.15, .01, 1), pos=(-.5, -0.3, 0.5),
                 collision_type=ouc.CollisionType.AABB)
box2.attach_to(base.scene)

box3 = ossop.box(half_extents=(.01, 1, .15), pos=(.3, 0.0, 1),
                 collision_type=ouc.CollisionType.AABB)
box3.attach_to(base.scene)

# Setup collision checker
collider = ocm.MJCollider()
collider.append(robot)
collider.append(box)
collider.append(box2)
collider.append(box3)
collider.actors = [robot]
collider.compile()

# Configure state space with joint limits
jlmt_low = robot.structure.compiled.jlmt_low_by_idx
jlmt_high = robot.structure.compiled.jlmt_high_by_idx
sspp = ompsp.SpaceProvider.from_box_bounds(
    lmt_low=jlmt_low,
    lmt_high=jlmt_high,
    collider=collider,
    cd_step_size=np.pi / 180)

# Plan path with RRT-Connect
planner = ompr.RRTConnectPlanner(pln_ctx=sspp, extend_step_size=np.pi / 36)
start = np.array([0, 0, 0, 0, 0, 0])
goal = np.array([-oum.pi / 2, -oum.pi / 4, oum.pi / 2,
                 -oum.pi / 2, oum.pi / 4, oum.pi / 3])
state_list = planner.solve(start=start, goal=goal, verbose=True)
print(f"Path found with {len(state_list)} waypoints")

# Visualize start and goal configurations
robot1 = robot.clone()
robot1.fk(qs=start)
robot1.rgba = (1, 0, 0, 0.5)
robot1.attach_to(base.scene)

robot2 = robot.clone()
robot2.fk(qs=goal)
robot2.rgba = (0, 0, 1, 0.5)
robot2.attach_to(base.scene)

# Animate the planned path
counter = [0]


def update_pose(dt, counter):
    if counter[0] < len(state_list):
        robot.fk(qs=state_list[counter[0]])
        counter[0] += 1
    else:
        counter[0] = 0


base.schedule_interval(update_pose, interval=0.1, counter=counter)
base.run()
```

![Motion Planning Demo](docs/images/motion_planning_demo.gif)
*RRT-Connect planning around obstacles with start (red) and goal (blue) configurations*

## Project Structure

```
one/
├── collider/          # Collision detection systems
│   ├── cpu_simd.py    # SIMD-accelerated CPU collision detection
│   ├── gpu_simd.py    # GPU-based collision detection
│   ├── mj_collider.py # MuJoCo collision wrapper
│   └── raycast.py     # Ray casting utilities
├── grasp/             # Grasp planning
│   └── antipodal.py   # Antipodal grasp generation
├── motion/            # Motion planning algorithms
│   ├── chain_planner.py        # Sequential planner for manipulator chains
│   └── probabilistic/          # Sampling-based planners
│       ├── rrt.py              # RRT and RRT-Connect
│       ├── prm.py              # Probabilistic Roadmap
│       ├── space_provider.py   # State space with collision checking
│       └── post_processor.py   # Path shortcutting and densification
├── physics/           # MuJoCo physics integration
│   ├── mj_env.py      # Environment wrapper
│   ├── mj_compiler.py # MJCF/URDF compilation
│   ├── mj_contact.py  # Contact force handling and visualization
│   └── mj_naming.py   # Body and geometry naming utilities
├── robots/            # Robot model library
│   ├── base/          # Base classes for robot structures
│   ├── manipulators/  # Industrial manipulator arms
│   │   └── kawasaki/  # Kawasaki RS007L 6-DOF arm
│   ├── end_effectors/ # Grippers and tools
│   │   └── onrobot/   # OnRobot 2FG7 gripper
│   └── vehicle/       # Mobile robot bases
│       └── xytheta.py # Planar mobile robot
├── scene/             # Scene management and geometry
│   ├── scene.py           # Scene graph management
│   ├── scene_object.py    # Renderable scene objects
│   ├── geometry.py        # Geometric primitives
│   ├── geometry_loader.py # STL/mesh loading
│   └── render_model.py    # Rendering abstractions
├── utils/             # Utilities and helpers
│   ├── math.py        # Transformation matrices, rotations, vectors
│   ├── constant.py    # Color palettes, collision types, enums
│   └── helper.py      # Miscellaneous helper functions
└── viewer/            # Visualization system
    └── world.py       # Interactive 3D world viewer
```

## Running Examples

The repository includes 25 example scripts demonstrating various features. All examples are located in the repository root.

### Basic Visualization
```bash
python test_2fg7.py                    # Gripper visualization and grasping
python test_rs007l_spawn.py            # Robot spawning and display
python test_bunny.py                   # Loading and rendering mesh objects
```

### Collision Detection
```bash
python test_collider_bunny.py  # CPU SIMD collision detection demo
python test_collider_bunny.py      # GPU collision detection demo
python test_cpusimd_collider_box.py    # Box collision detection
python test_2fg7_collision.py          # Gripper collision checking
python benchmark_collider.py           # CPU vs GPU performance comparison
```

### Motion Planning
```bash
python test_rrtc_rs007l.py             # RRT-Connect with obstacles
python test_prm_rs007l.py              # Probabilistic Roadmap planning
python test_rrtc_rs007l_dual.py        # Dual-arm motion planning
python test_rrtc_rs007l_engage_2fg7.py # Planning with end effector attached
```

### Physics Simulation
```bash
python test_mujoco_bunny.py            # Basic MuJoCo physics simulation
python test_mujoco_rs007l.py           # Robot physics simulation
python test_mujoco_rs007l_and_bunny.py # Robot-object interaction
python test_mujoco_xytheta.py          # Mobile robot simulation
```

### Grasp Planning
```bash
python test_2fg7_grip_at.py            # Target pose grasping
python test_2fg7_antipodal_debug.py    # Antipodal grasp generation
```

### Robot Kinematics
```bash
python test_rs007l_ik.py               # Inverse kinematics
python test_rs007l_engage_2fg7.py      # Attaching end effectors
python test_rs007l_dual.py             # Dual arm coordination
```

## Supported Robots

### Manipulators
- **Kawasaki RS007L** - 6-DOF industrial robotic arm with kinematics and collision models
- **Denso Cobotta** - Collaborative robot arm

### End Effectors
- **OnRobot 2FG7** - Parallel jaw gripper with adjustable jaw width and grasp planning

### Mobile Robots
- **xy-theta vehicle** - Planar mobile robot model for navigation tasks

### Extensibility
The framework is designed to easily incorporate new robot models. Simply provide URDF or MJCF descriptions and follow the patterns in the `robots/` module. See [AGENTS.md](AGENTS.md) for coding guidelines.

## Documentation

- **Coding Guidelines:** See [AGENTS.md](AGENTS.md) for detailed coding standards, import conventions, naming patterns, and best practices for contributors and AI coding agents
- **Examples:** All test files (`test_*.py`) serve as comprehensive usage examples demonstrating specific features
- **API Reference:** Refer to inline docstrings in the source code for detailed API documentation

## Contributing

Contributions are welcome! To contribute:

1. **Follow the coding style** defined in [AGENTS.md](AGENTS.md)
2. **Write test scripts** demonstrating new features (following the `test_*.py` pattern)
3. **Keep code simple and well-documented** - clarity over complexity
4. **Maintain compatibility** with Python 3.12+
5. **Respect the design philosophy** - simplicity, consistency, controllability

When adding new features:
- Ensure they integrate cleanly with existing modules
- Document array shapes and transformation conventions
- Use `numpy.float32` consistently for numerical arrays
- Follow the established import abbreviation patterns

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MuJoCo** - High-performance physics simulation and collision detection backend
- **pyglet** - Cross-platform OpenGL-based windowing and visualization framework
- **NumPy & SciPy** - Foundational numerical computing and scientific computing libraries that power all mathematical operations

---

*Built with a focus on simplicity, consistency, and controllability for robotics research.*
