# 运动规划

`one.motion` 提供基于采样的规划器（`one.motion.probabilistic`）和
轨迹工具（`one.motion.trajectory`）。它们由一个规划上下文驱动，
该上下文封装了机器人和一个碰撞预言机（通常是 `MJCollider.is_collided`）。

## RRT-Connect

一种双向树规划器，用于在起始配置和目标配置之间进行关节空间查询，
碰撞器会剔除无效的采样和边：

```python
--8<-- "examples/test_rrtc_rs007l.py"
```

需要注意的事项：

- 机器人、地面和任何障碍物都注册到一个碰撞器；
  规划上下文将 `is_collided(qs)` 转化为规划器的有效性检查。
- 结果是一条关节空间路径；相邻的变体
  （`test_prm_rs007l`、`test_rrtc_rs007l_dual`、`test_rrtc_rs007l_gpu_aabb`）会替换
  规划器或碰撞后端。

## 笛卡尔轨迹

当你需要 tcp 沿直线笛卡尔路径运动时（例如工具在工件上
扫掠），`one.motion.interpolation.cartesian.linear_to_jpath` 沿该直线
逐步求解 IK，而 `time_param.retime_trapezoidal` 则添加速度/
加速度曲线。本示例安装一把螺丝刀，并驱动其尖端沿
笛卡尔线段运动：

```python
--8<-- "examples/test_rs007l_mount_orsd_jtraj.py"
```

需要注意的事项：

- `linear_to_jpath(robot, start_pos, start_rotmat, goal_pos, goal_rotmat,
  tcp=...)` 返回关节路径以及采样得到的位姿；传入一个有偏置的
  `ref_qs`，使逐步 IK 保持在平滑的分支上。
- `retime_trapezoidal(q_seq, v_max, a_max, dt)` 对路点进行时间参数化。

## 另请参阅

- [`one.motion`](../api/motion.md) —— RRT、PRM、规划上下文、轨迹工具。
- [碰撞](collision.md) —— 构建 `is_collided` 预言机。
