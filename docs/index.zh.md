# `one`

`one` 是一个自包含的机器人库：**运动学、场景、网格几何、碰撞、抓取规划、运动以及查看器**。它有意实现了自己的网格 / 变换 / 碰撞数学，因此你很少需要借助 trimesh、open3d、scipy.spatial.transform 或 fcl。

## 内含功能

| 需求 | 模块 | 别名 |
|---|---|---|
| 网格操作：凸包、表面采样、射线–网格、裁剪、细分、旋转生成 | `one.scene.geometry_ops` | `osgop` |
| 旋转 / 位姿：`rotmat_from_*`、`tf_from_pos_rotmat`、quat/euler/rotvec、slerp | `one.utils.math` | `oum` |
| 网格文件加载（STL/DAE） | `one.geom.loader` | `ogl` |
| 碰撞检测（CPU/GPU 批量） | `one.collider.cpu_simd`、`gpu_simd_batch` | `occs` |
| 抓取规划 | `one.grasp.antipodal` / `polypodal` / `monocontact` | `ogab` / `ogpp` / `ogmc` |
| 场景对象 / 查看器 | `one.scene.scene_object`、`one.viewer.world` | `osso` / `ovw` |

便捷别名已从包中导出：
`from one import oum, osgop, occs, ouc, ossop, ovw, ogab, ogmc, ...`

完整的、自动生成的模块 → 函数/类映射，请参见 **[API Index](API_INDEX.md)**。

## 关键约定

- **位置优先（pos-first）、ROS Pose 顺序**：`oum.tf_from_pos_rotmat(pos, rotmat)`、
  `mech.set_pos_rotmat(pos, rotmat)`、
  `mech.ik(tgt_pos, tgt_rotmat, chain='main', tcp='flange')`。
- **每个机器人都继承 `MechBase`**（机械臂、灵巧手、夹爪、人形机器人皆然）；
  它们仅在所注册的 `chains` / `tcps` 上有所不同。TCP 通过名称查找：
  `mech.tcp('flange')`。
- **抓取规划器命名**：前缀 = 接触点数，后缀 = 机构。
  `antipodal`（2 点对置夹捏）、`polypodal`（N 点）、`monocontact`
  （1 点吸附 / 按压）。
- **灵巧手**通过 `hand.spawn_jaw('pinch')` 向平行夹爪规划器呈现自身，
  该方法返回一个经过标定、不可变绑定的视图，暴露 antipodal 所期望的夹爪接口。

## 安装

```bash
pip install -e .
```

使用具有 pyglet / numpy / scipy 的 Python 运行示例（例如 `py -3.12`）；
示例会建立一个 `World` 并调用 `base.run()` 来启动 GUI。

## 重新生成本站点的 API 索引

[API Index](API_INDEX.md) 由一次静态 AST 扫描生成（无导入、无硬件副作用）：

```bash
py -3.12 tools/gen_api_index.py
```

它会在每次推送时通过文档工作流自动运行，因此发布的索引永远不会过时。
