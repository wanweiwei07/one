# 运动学

`one` 中的每个机器人 —— 机械臂、夹爪、灵巧手或人形机器人 —— 都继承单一基类，
`MechBase`。它们仅在所注册的 **chains** 和 **tcps** 上有所不同：

- 一个 **chain** 命名一段串联的关节序列（一个 `base → tip` 连杆跨度），它们一起
  运动。一个 6-DOF 机械臂注册 `'main'`；一个人形机器人注册 `'left_arm'`、
  `'right_arm'`、`'neck'`、…
- 一个 **tcp**（工具中心点）是刚性附着于某连杆的具名坐标系，例如
  机械臂上的 `'flange'` 或夹爪上的 `'grasp_center'`。通过名称查找它：
  `robot.tcp('flange')`。

## FK 与 IK

`robot.fk(qs=...)` 驱动关节并更新每个连杆的世界变换。
`robot.ik(tgt_pos, tgt_rotmat, chain='main', tcp='flange')` 是**位置优先**的
（ROS Pose 顺序），并返回一个关节解的列表 —— 不可达时为空，
当多个分支可到达目标时会有多个解。

这个往返示例用已知角度驱动 FK，读回法兰位姿，
然后请求 IK 恢复关节配置并检查误差：

```python
--8<-- "examples/test_ik_roundtrip.py"
```

需要注意的事项：

- `robot.tcp('flange').tf` 是法兰坐标系的 4×4 世界变换。
- `robot.ik(pos, rotmat)` 先取位置，再取旋转矩阵。
- 解析求解器返回*所有*分支；一次往返应当恢复出输入
  配置（在关节限位 / 缠绕等价意义下），误差在亚毫米级。

## 可视化机构 { #visualizing-the-mechanism }

`KineVisualizer` 绘制运动学**骨架**（关节轴绘为定子/转子
圆柱，连杆绘为杆）。传入一个 `chain` 只绘制该链，或 `chain=None`
绘制整个机构 —— 它读取机器人已计算好的逐连杆
世界变换，因此对分支型人形机器人和串联机械臂同样正确。
设置 `robot.alpha` 使网格半透明，从而透出骨架：

```python
--8<-- "examples/test_cvr038_kine_visualizer.py"
```

## 另见

- [`one.utils.math`](../api/utils.md) —— `rotmat_from_*`、`tf_from_pos_rotmat`、
  `frame_from_normal`、四元数/欧拉角/旋转矢量转换、slerp。
- [`one.robots`](../api/robots.md) —— 机器人类和 `kine_visualizer`。
- [末端执行器与工具更换](end_effectors.md) —— 跨对象 IK 到所安装
  夹爪的 `grasp_center`。
