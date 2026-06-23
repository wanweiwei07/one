# 快速开始

三个简短、可运行的示例。每个都位于 `examples/` 下，并在
此处逐字嵌入 —— 用 `py -3.12 examples/<name>.py` 运行它。

代码库中通篇出现两种导入风格，二者可互换：

```python
from one import oum, ovw, osso, ossop, khi_rs007l   # convenience aliases
import one.utils.math as oum                          # or the full path
```

这些别名从 `one/__init__.py` 导出（`oum` = `one.utils.math`、
`osgop` = `one.scene.geometry_ops`、`ovw` = `one.viewer.world`、…）。

## 生成机器人并查看它

`World` 打开查看器；`attach_to(base.scene)` 使对象可见；
`robot.fk(qs=...)` 驱动关节；`robot.clone()` 制作一个独立副本。
`base.schedule_interval(fn, dt)` 每 `dt` 秒运行一次 `fn`，而 `base.run()`
启动循环。

```python
--8<-- "examples/test_rs007l_spawn.py"
```

## 求解逆运动学

`robot.ik(tgt_pos, tgt_rotmat)` 是**位置优先**的（ROS Pose 顺序），并
返回一个关节解的列表（不可达时为空）。它默认
`chain='main'`、`tcp='flange'`。这里每个解都被克隆并绘制出来，因此你
能看到 IK 分支扇形展开。

```python
--8<-- "examples/test_rs007l_ik.py"
```

## 加载并显示网格

`SceneObject.from_file` 通过内置加载器读取网格（STL/DAE）。
`collision_type`（`MESH`、`CAPSULE`、…）会构建碰撞代理；
`toggle_render_collision` 显示它。`.pos` / `.rotmat` 放置对象，而
`.clone()` 复制它。

```python
--8<-- "examples/test_bunny.py"
```

## 接下来去哪里

- [运动学](tutorials/kinematics.md) —— 链、tcp、FK/IK、骨架查看器。
- [场景与几何](tutorials/scene_geometry.md) —— 网格、凸包、采样。
- [抓取规划](tutorials/grasp_planning.md) —— antipodal / monocontact / placement。
- [运动规划](tutorials/motion_planning.md) —— RRT-Connect、PRM、笛卡尔。
- [API 参考](api/index.md)。
