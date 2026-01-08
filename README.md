# one

`one` aims at seamless integration of robot motion planning and learning, with a strong emphasis on simplicity, consistency, and controllability.

The framework only depends on MuJoCo and does not rely on external collision libraries.
For kinematic planning, collision checking is performed using MuJoCo’s native pipeline (mj_kinematics + mj_collision).
For dynamic simulation, MuJoCo’s standard time stepping (mj_step) is used directly.

Visualization is implemented from scratch on top of pyglet, providing full control from low-level rendering primitives such as device buffers, shaders, and render passes to high-level abstractions including scenes, kinematic chains, and articulated mechanisms. Users can flexibly intervene at any layer depending on their needs.

The internal data structures are designed to be compatible with both URDF and MJCF XML descriptions, enabling unified handling of robot models across different representations without forcing premature conversions.

Overall, one is intended as a lightweight yet expressive research framework for studying motion planning, learning, and their interaction under a single, coherent simulation backend.

## Environment

- **Python**: 3.12 or later  
- **Package manager**: `pip`

It is recommended to use a virtual environment (e.g. `venv`, `conda`, or similar) to avoid dependency conflicts.

## Installation

Clone the repository and install the dependencies listed in `requirements.txt`.

```bash
git clone https://github.com/wanweiwei07/one.git
cd one
pip install -r requirements.txt