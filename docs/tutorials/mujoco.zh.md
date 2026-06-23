# MuJoCo 集成

`one.physics` 将场景桥接到 **MuJoCo** 以进行刚体仿真。一个
`MJEnv` 由 `one` 场景构建而成：它将场景中的物体转化为
MuJoCo 模型，推进物理仿真，并将得到的状态同步回来，使
查看器显示出仿真的运动。

## 让机器人落到平面上

本示例在地面平面上方堆叠多个 RS007L 副本，并让 MuJoCo
仿真它们下落。`mjenv.step` 被调度到查看器循环上，因此物理
在每一帧都向前推进：

```python
--8<-- "examples/test_mujoco_rs007l.py"
```

需要注意的事项：

- `MJEnv(scene=base.scene)` 从附着到场景上的一切
  （机器人、基本体、平面）构建仿真。
- `mjenv.save("scene.xml")` 导出生成的 MuJoCo 模型——便于
  检查或复用转换后的场景。
- `base.schedule_interval(mjenv.step)` 让仿真与
  查看器同步运行。
- 注册机器人、夹爪和静态环境；被持有/操作的部件
  通过抓取附着来处理，而不是作为单独的仿真体添加。

相关示例：`test_mujoco_bunny`、`test_mujoco_rs007l_and_bunny`、
`test_mujoco_rs007l_engage_2fg7`、`test_mujoco_xytheta`。

## 另请参阅

- [`one.physics`](../api/physics.md) —— `MJEnv`、接触、模型转换。
- [碰撞](collision.md) —— 解析式（非物理）碰撞路径。
