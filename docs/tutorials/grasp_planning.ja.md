# 把持計画

`one.grasp` には表面サンプリングを行う3つのプランナがあります。命名規則は体系的で、
**接頭辞が接触点数、接尾辞が機構**を表します:

| プランナ | 接触点数 | 機構 | グリッパ |
|---|---|---|---|
| `antipodal` | 2 | 対向ピンチ（力閉じ込め） | 平行ジョー |
| `polypodal` | N | 対向N点パターン（力閉じ込め） | 多パッド / 多指 |
| `monocontact` | 1 | 片側吸着 / 押し付け | 吸着 / ティップ |

3つすべてが対象の表面をサンプリングし、候補姿勢を構築し、グリッパと対象の干渉を
棄却して、スコア付けされた [`Grasp`](../api/grasp.md) レコードのリストを返します。

`Grasp` は**自己完結的でグリッパ非依存**です。把持の `pose` / `pre_pose`
（対象のローカル座標系における tcp 座標系）、ハンド根本に対する `tcp` の loc_tf、
そしてハンドの `qpos` / `pre_qpos` を固定して保持するため、リプレイ時にグリッパの
ライブ状態から再導出する必要がありません。`g.make_tcp(gripper)` は IK 用の tcp を
再構築し、`g.transformed(obj.wd_tf)` はそれをワールドへマッピングします。
`g.provenance` はオプションかつ非正規のメタデータ（`jaw_width`、`mode` など）を
保持します。完全なレコードについては [`one.grasp.grasp.Grasp`](../api/grasp.md) を
参照してください。

## 対向把持（平行ジョー）

`antipodal(gripper, target_sobj, ...)` は2点対向の把持を求めます。各結果は、
グリッパの `grip_at` が期待する `grasp_center` 座標系を `pose` に持つ `Grasp`
です（その `jaw_width` は `g.provenance` にあります）。

```python
--8<-- "examples/test_2fg7_antipodal.py"
```

## 単一接触（吸着 / ティップ）

`monocontact(tool, target_sobj, tcp='tip', ...)` は単一の接触軸を内向きの表面法線に
合わせます。吸着カップやツールのティップ向けです。結果は `Grasp` レコードです
（対向閉じ込めがないため `qpos` はツールの固定コンフィグであり、`tcp_name` は
`g.provenance` にあります）。`approach_bias`（デフォルトはワールド +z）は上向きの
面を優先します。

```python
--8<-- "examples/test_orsd_monocontact.py"
```

## 安定設置

把持は話の半分にすぎません。`one.grasp.placement` は平らな支持面上での物体の
**安定静止姿勢**を（その凸包から）計算します。これは拾い上げたものを置く先となります:

```python
--8<-- "examples/test_placement_bunny.py"
```

## 関連項目

- [`one.grasp`](../api/grasp.md) — `antipodal`、`polypodal`、`monocontact`、
  `placement`、および共有の `_common` 干渉ヘルパ。
- [エンドエフェクタとツールチェンジ](end_effectors.md) — グリッパの取り付けと、
  その `grasp_center` への IK の求解。
