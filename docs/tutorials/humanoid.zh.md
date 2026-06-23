# 人形机器人（Linx L1）

人形机器人与机械臂是同一个 `MechBase`——它只是注册了多条链。
Linx **L1** 上半身注册了 `'left_arm'`、`'right_arm'`、
`'left_arm_waist'`、`'right_arm_waist'` 和 `'neck'`。提供了两个类：

- `L1` —— 裸机身（机械臂末端位于 `*_arm_link_6` 法兰处；自行安装你的
  末端执行器）。
- `L1O6` —— 在法兰上预先安装了两只 Linkerbot **O6** 灵巧手的 `L1`。

由于这些手是已安装的子机构，你需要沿着某条机身链向下求解到
手部 tcp 的**跨物体 IK** 来到达目标：

```python
robot.ik(pos, rotmat, chain='left_arm', tcp=robot.left_hand.tcp('power_center'))
```

`'left_arm_waist'` 链加入了腰部关节，让躯干协助
伸展（在最近解度量中对其加权，以保持躯干静止并让
机械臂完成工作，就像人一样）。

## 用规划好的抓取进行夹捏放置

本示例让 L1O6 左手用**对置**抓取拾取一只小兔子
（而不是手工挑选的位姿）。灵巧手通过 `spawn_jaw('pinch')`
将自己呈现给平行钳口规划器，该方法返回一个校准过的钳口视图；每个抓取
都成为拾取-放置关键帧，而自由空间的跳跃则用 RRT 规划：

```python
--8<-- "examples/test_l1o6_pinch_bunny.py"
```

需要注意的事项：

- `hand.spawn_jaw('pinch')` 将多指手适配成 `antipodal` 所期望的
  接口；每个返回的抓取都是一个世界夹捏中心位姿 + 开合宽度。
- 只有 `'left_arm_waist'` 链运动；机身的其余部分保持冻结。
- 设置 `ONE_HEADLESS=1` 即可在无窗口的情况下运行规划/IK 验证。

## 可视化骨架

`L1` 自身的 `__main__` 将机身渲染为半透明（`robot.alpha = 0.3`），并
叠加 `KineVisualizer` 骨架——它读取每个连杆的世界变换，
因此能正确绘制*分支*的人形树（两条机械臂从
腰部分支，而非串联排列）。参见 [运动学](kinematics.md#visualizing-the-mechanism)。

## 另请参阅

- [`one.robots`](../api/robots.md) —— 人形机器人与末端执行器。
- [末端执行器与换工具](end_effectors.md) —— 安装与跨物体 IK。
- [抓取规划](grasp_planning.md) —— 此处使用的对置抓取。
