# Quick Start

実行可能な小さな例を3つ紹介します。それぞれ `examples/` 配下にあり、ここでは
そのまま埋め込んでいます。`py -3.12 examples/<name>.py` で実行してください。

コードベース全体を通じて2つのインポートスタイルが現れ、相互に置き換え可能です:

```python
from one import oum, ovw, osso, ossop, khi_rs007l   # convenience aliases
import one.utils.math as oum                          # or the full path
```

エイリアスは `one/__init__.py` からエクスポートされています（`oum` = `one.utils.math`、
`osgop` = `one.scene.geometry_ops`、`ovw` = `one.viewer.world`、…）。

## ロボットをスポーンして表示する

`World` がビューアを開きます。`attach_to(base.scene)` でオブジェクトが表示されます。
`robot.fk(qs=...)` が関節を駆動します。`robot.clone()` が独立したコピーを作成します。
`base.schedule_interval(fn, dt)` は `dt` 秒ごとに `fn` を実行し、`base.run()` が
ループを開始します。

```python
--8<-- "examples/test_rs007l_spawn.py"
```

## 逆運動学を解く

`robot.ik(tgt_pos, tgt_rotmat)` は **位置優先**（ROS Pose 順序）で、関節解のリストを
返します（到達不能なら空）。デフォルトは `chain='main'`、`tcp='flange'` です。
ここではすべての解をクローンして描画するため、IK の分岐が扇状に広がる様子が
見えます。

```python
--8<-- "examples/test_rs007l_ik.py"
```

## メッシュを読み込んで表示する

`SceneObject.from_file` は独自ローダ経由でメッシュ（STL/DAE）を読み込みます。
`collision_type`（`MESH`、`CAPSULE`、…）が干渉プロキシを構築し、
`toggle_render_collision` がそれを表示します。`.pos` / `.rotmat` でオブジェクトを
配置し、`.clone()` で複製します。

```python
--8<-- "examples/test_bunny.py"
```

## 次はどこへ

- [Kinematics](tutorials/kinematics.md) — チェーン、tcp、FK/IK、スケルトンビューア。
- [Scene & Geometry](tutorials/scene_geometry.md) — メッシュ、凸包、サンプリング。
- [Grasp planning](tutorials/grasp_planning.md) — antipodal / monocontact / placement。
- [Motion planning](tutorials/motion_planning.md) — RRT-Connect、PRM、Cartesian。
- [API reference](api/index.md)。
