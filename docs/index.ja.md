# `one`

`one` は自己完結型のロボティクスライブラリです。**運動学、シーン、メッシュ幾何、
干渉、把持計画、動作、そしてビューア**を備えます。メッシュ／変換／干渉の数学を
意図的に独自実装しているため、trimesh、open3d、scipy.spatial.transform、fcl に
頼る必要はほとんどありません。

## 内容

| 必要なもの | モジュール | エイリアス |
|---|---|---|
| メッシュ操作: 凸包、表面サンプリング、レイ–メッシュ、クリップ、サブディビジョン、回転体生成 | `one.scene.geometry_ops` | `osgop` |
| 回転／姿勢: `rotmat_from_*`、`tf_from_pos_rotmat`、quat/euler/rotvec、slerp | `one.utils.math` | `oum` |
| メッシュファイル読み込み (STL/DAE) | `one.geom.loader` | `ogl` |
| 干渉チェック (CPU/GPU バッチ) | `one.collider.cpu_simd`、`gpu_simd_batch` | `occs` |
| 把持計画 | `one.grasp.antipodal` / `polypodal` / `monocontact` | `ogab` / `ogpp` / `ogmc` |
| シーンオブジェクト／ビューア | `one.scene.scene_object`、`one.viewer.world` | `osso` / `ovw` |

便利なエイリアスはパッケージからエクスポートされています:
`from one import oum, osgop, occs, ouc, ossop, ovw, ogab, ogmc, ...`

完全な自動生成のモジュール → 関数/クラス対応表については
**[API Index](API_INDEX.md)** を参照してください。

## 主要な規約

- **pos-first、ROS Pose 順序**: `oum.tf_from_pos_rotmat(pos, rotmat)`、
  `mech.set_pos_rotmat(pos, rotmat)`、
  `mech.ik(tgt_pos, tgt_rotmat, chain='main', tcp='flange')`。
- **すべてのロボットは `MechBase` を継承します**（アーム、ハンド、グリッパ、
  ヒューマノイドも同様）。違いはどの `chains` / `tcps` が登録されているかだけです。
  TCP は名前で参照します: `mech.tcp('flange')`。
- **把持計画器の命名**: 接頭辞 = 接触点数、接尾辞 = 機構。
  `antipodal`（2点の対向ピンチ）、`polypodal`（N点）、`monocontact`
  （1点の吸着／押し付け）。
- **多指ハンド**は `hand.spawn_jaw('pinch')` を通じて平行ジョー計画器に
  自身を提示します。これは校正済みで不変に束縛されたビューを返し、
  antipodal が期待するグリッパインターフェースを公開します。

## インストール

```bash
pip install -e .
```

例は pyglet / numpy / scipy を備えた Python（例: `py -3.12`）で実行してください。
例は `World` をセットアップし、GUI のために `base.run()` を呼び出します。

## このサイトの API インデックスの再生成

[API Index](API_INDEX.md) は静的な AST スキャン（インポートなし、ハードウェアへの
副作用なし）で生成されます:

```bash
py -3.12 tools/gen_api_index.py
```

これはドキュメントワークフローによってプッシュごとに自動実行されるため、公開される
インデックスが古くなることはありません。
