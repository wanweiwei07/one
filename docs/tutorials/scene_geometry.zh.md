# 场景与几何

`one` 自带一套网格几何栈，因此你很少需要 trimesh 或 open3d。
网格就是朴素的 `(vertices, faces)` 数组，而 `one.scene.geometry_ops`
（`osgop`）直接对它们进行操作。

## 场景对象

一个 `SceneObject` 封装一个网格加一个碰撞代理。`from_file` 通过内置加载器
加载 STL/DAE；`collision_type` 选择代理（`MESH`、
`CAPSULE`、…）；`.pos` / `.rotmat` 放置它；`.alpha` 和
`toggle_render_collision` 控制渲染；`.clone()` 复制它。

```python
--8<-- "examples/test_bunny.py"
```

## 凸包与表面分割

`one.geom.fitting.convex_hull` 构建凸包；`one.geom.surface.segment_surface`
将共面的面（在法向容差内）归组为平坦面片 —— 这是
稳定放置和抓取推理的基础。这里 bunny 的凸包被分割，
每个面片在按 `SPACE` 时逐一显现：

```python
--8<-- "examples/test_segment_bunny.py"
```

需要注意的事项：

- `bunny.collisions[0].geom` 是底层几何（`.vs`、`.fs`）。
- `ossop.mesh(vs, fs, rgb=...)` 绘制任意三角网格。
- `segment_surface(geom, normal_tol_deg=...)` 返回多个面索引列表，每个
  平坦面片一个。

## 有用的入口

`osgop` 还提供 `sample_surface`、`sample_count_from_area`、
`ray_triangles_batch_far` / `ray_shoot`、`clip_mesh`、`subdivide_once`、
`revolve` 和 `icosahedron`。

## 另见

- [`one.scene.geometry_ops`](../api/scene.md) —— 网格操作。
- [`one.geom`](../api/geom.md) —— 加载器、拟合、表面、基本体。
- [抓取规划](grasp_planning.md) —— 表面采样为规划器提供输入。
