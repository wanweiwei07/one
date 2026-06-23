# シーンと幾何

`one` は独自のメッシュ幾何スタックを備えているため、trimesh や open3d はほとんど
必要ありません。メッシュは単純な `(vertices, faces)` の配列であり、
`one.scene.geometry_ops`（`osgop`）はそれらを直接操作します。

## シーンオブジェクト

`SceneObject` はメッシュと干渉プロキシをラップします。`from_file` は独自ローダを
通じて STL/DAE を読み込みます。`collision_type` がプロキシ（`MESH`、`CAPSULE`、…）を
選択します。`.pos` / `.rotmat` で配置します。`.alpha` と
`toggle_render_collision` がレンダリングを制御します。`.clone()` で複製します。

```python
--8<-- "examples/test_bunny.py"
```

## 凸包と表面セグメンテーション

`one.geom.fitting.convex_hull` は凸包を構築します。`one.geom.surface.segment_surface`
は同一平面上の面を（法線の許容差内で）平坦なパッチにグループ化します — これは
安定配置と把持の推論の基礎です。ここでは bunny の凸包をセグメント化し、
各パッチを `SPACE` で順に表示します:

```python
--8<-- "examples/test_segment_bunny.py"
```

注目すべき点:

- `bunny.collisions[0].geom` は背後にある幾何（`.vs`、`.fs`）です。
- `ossop.mesh(vs, fs, rgb=...)` は任意の三角形メッシュを描画します。
- `segment_surface(geom, normal_tol_deg=...)` は平坦なパッチごとに1つずつ、
  面インデックスのリストを返します。

## 便利なエントリポイント

`osgop` は `sample_surface`、`sample_count_from_area`、
`ray_triangles_batch_far` / `ray_shoot`、`clip_mesh`、`subdivide_once`、
`revolve`、`icosahedron` も提供します。

## 関連項目

- [`one.scene.geometry_ops`](../api/scene.md) — メッシュ操作。
- [`one.geom`](../api/geom.md) — ローダ、フィッティング、表面、プリミティブ。
- [Grasp planning](grasp_planning.md) — 表面サンプリングが計画器に供給されます。
