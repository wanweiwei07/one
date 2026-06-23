# 末端执行器与换工具

末端执行器同样是 `MechBase` 机构。它们混入行为并注册一个
工作 tcp：

- **夹爪**（`GripperMixin`）注册一个 `'grasp_center'` tcp，并暴露
  `open()` / `close()` / `set_jaw_width()` 以及 `grasp(child)` / `release(child)`。
- **点工具**（`PointMixin`，例如螺丝刀）注册一个 `'tip'` tcp，并
  暴露 `touch_at(...)` / `attach(...)`。

## 安装

`arm.mount(ee, parent_lnk, loc_tf, update=True)` 将末端执行器附着到某个
连杆上；此后每次 `arm.fk(...)` 都会传播到所安装的工具。`update`
仅控制工具是否在*安装的那一刻*就吸附到位——它
在运动过程中始终跟随。`arm.unmount(ee)` 将其卸下（工具停留在原处
不动）。要用所安装的工具拾取目标，请对
工具的 tcp 求解**跨物体 IK**：

```python
qs = arm.ik(pos, rotmat, tcp=gripper.tcp('grasp_center'))
```

## 完整的换工具序列

本示例从一台裸 RS007L 开始，安装一个 2FG7 夹爪去拾取和放置一只
兔子，将其放回支架并卸下夹爪，然后安装一把螺丝刀在兔子
上方作业——端到端地演练 mount / grasp / release / unmount：

```python
--8<-- "examples/test_rs007l_toolchange.py"
```

需要注意的事项：

- 工具“支架”位姿只是一个法兰位姿；在法兰位于支架处时
  以单位 `loc_tf` 进行安装，即可实现无跳变的拾取。
- IK 以当前激活工具的 tcp 为目标：夹爪处于安装状态时为 `gripper.tcp('grasp_center')`，
  螺丝刀处于安装状态时为 `screwdriver.tcp('tip')`，而
  机械臂为裸臂时则为默认的 `'flange'`。
- `SceneObject` 默认 `is_free = False`；在
  `grasp` 之前先设置 `bunny.is_free = True`，这样它才能被安装到夹爪上。

## 另请参阅

- [`one.robots`](../api/robots.md) —— `end_effectors`（夹爪、点工具）。
- [抓取规划](grasp_planning.md) —— 抓取位姿的来源。
- [运动学](kinematics.md) —— 链与 tcp。
