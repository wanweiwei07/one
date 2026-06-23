# 抓取规划

`one.grasp` 提供三种表面采样规划器。命名是系统化的——**前缀是接触点数量，后缀是机制**：

| 规划器 | 接触点 | 机制 | 夹爪 |
|---|---|---|---|
| `antipodal` | 2 | 对置夹捏（力封闭） | 平行钳口 |
| `polypodal` | N | 对置 N 点模式（力封闭） | 多接触垫 / 灵巧手 |
| `monocontact` | 1 | 单侧吸附 / 按压 | 吸盘 / 工具尖端 |

这三者都对目标表面进行采样，构建候选位姿，并剔除夹爪与目标之间的碰撞，返回一组带评分的
[`Grasp`](../api/grasp.md) 记录。

`Grasp` 是**自包含且与夹爪无关的**：它冻结了抓取
`pose` / `pre_pose`（物体局部坐标系中的 tcp 坐标系）、相对于手部根部的 `tcp`
loc_tf，以及手部的 `qpos` / `pre_qpos`——因此回放
无需从夹爪的实时状态重新推导。`g.make_tcp(gripper)`
重建 IK tcp；`g.transformed(obj.wd_tf)` 将其映射到世界坐标系；
`g.provenance` 携带可选的、非权威性的元数据（`jaw_width`、
`mode`、……）。完整记录参见 [`one.grasp.grasp.Grasp`](../api/grasp.md)。

## Antipodal（平行钳口）

`antipodal(gripper, target_sobj, ...)` 查找 2 点对置抓取。每个结果
都是一个 `Grasp`，其 `pose` 是夹爪的 `grip_at`
所期望的 `grasp_center` 坐标系（其 `jaw_width` 位于 `g.provenance` 中）。

```python
--8<-- "examples/test_2fg7_antipodal.py"
```

## Monocontact（吸附 / 尖端）

`monocontact(tool, target_sobj, tcp='tip', ...)` 将单一接触轴对齐到
向内的表面法线——用于吸盘或工具尖端。结果是 `Grasp`
记录（没有对置闭合，因此 `qpos` 是工具的固定配置，且
`tcp_name` 位于 `g.provenance` 中）；`approach_bias`（默认世界 +z）会奖励
朝上的表面。

```python
--8<-- "examples/test_orsd_monocontact.py"
```

## 稳定放置

抓取只是故事的一半——`one.grasp.placement` 计算物体在平面支撑上的**稳定静止
位姿**（基于其凸包），这正是你
放置所拾取物体的位置：

```python
--8<-- "examples/test_placement_bunny.py"
```

## 另请参阅

- [`one.grasp`](../api/grasp.md) —— `antipodal`、`polypodal`、`monocontact`、
  `placement`，以及共享的 `_common` 碰撞辅助工具。
- [末端执行器与换工具](end_effectors.md) —— 安装夹爪并
  求解到其 `grasp_center` 的 IK。
