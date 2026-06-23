# 碰撞

`one` 提供两种具有相同接口的碰撞后端：

- `one.collider.cpu_simd`（`occs`）—— numpy/SIMD 三角网格检测。
- `one.collider.gpu_simd_batch` —— GPU 批量等价实现（抓取规划器先尝试
  它，并回退到 CPU）。

对于机器人级别的查询（整个机器人在许多配置下与环境的碰撞）
有 `one.collider.mj_collider.MJCollider`（`ocm`），它
一次性编译机器人 + 静态障碍物，然后回答 `is_collided(qs)`。

## 用 MJCollider 进行机器人对地面检测

从机器人和障碍物构建碰撞器，标记哪些物体是
运动的**执行体（actors）**，`compile()` 一次，然后探测各配置：

```python
--8<-- "examples/test_collision_check.py"
```

需要注意的事项：

- `mjc.append(robot)` / `mjc.append(ground)` 注册物体；`mjc.actors = [robot]`
  标记什么在运动；`mjc.compile(margin=...)` 最终确定宽相位（broadphase）。
- `mjc.is_collided(qs)` 将机器人驱动到 `qs` 并返回一个布尔值 —— 在
  规划循环中调用开销很低。
- 只向 `MJCollider` 注册机器人、其所安装的夹爪以及静态环境。
  被操作/被持有的部件*不*作为碰撞器物体加入。

## 更底层的网格对网格

对于一次性的网格对，`occs.create_detector()` + `occs.build_batch(items, pairs)`
+ `detector.detect_collision_batch(batch)` 返回接触点（或 `None`）。
这正是抓取规划器用来剔除碰撞位姿的方式（参见
[`one.grasp._common.build_ee_target_detector`](../api/grasp.md)）。

## 另见

- [`one.collider`](../api/collider.md) —— CPU/GPU 检测器、MJCollider、批处理。
- [运动规划](motion_planning.md) —— 以 `is_collided` 作为规划器的判定器。
