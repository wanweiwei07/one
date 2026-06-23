# 運動学

`one` のすべてのロボット — アーム、グリッパ、ハンド、ヒューマノイド — は単一の
基底クラス `MechBase` を継承します。違いはどの **chains** と **tcps** を登録するか
だけです:

- **chain** は一緒に動く関節の直列の連なり（`base → tip` のリンク区間）を指します。
  6 自由度アームは `'main'` を登録し、ヒューマノイドは `'left_arm'`、`'right_arm'`、
  `'neck'`、… を登録します。
- **tcp**（ツールセンターポイント）はリンクに剛体的に取り付けられた名前付き座標系で、
  例えばアームの `'flange'` やグリッパの `'grasp_center'` です。名前で参照します:
  `robot.tcp('flange')`。

## FK と IK

`robot.fk(qs=...)` は関節を駆動し、各リンクのワールド変換を更新します。
`robot.ik(tgt_pos, tgt_rotmat, chain='main', tcp='flange')` は **位置優先**
（ROS Pose 順序）で、関節解のリストを返します。到達不能なら空、複数の分岐が
ターゲットに到達する場合は複数返します。

このラウンドトリップの例は、既知の角度で FK を駆動し、フランジの姿勢を読み戻し、
それから IK に関節構成を復元させて誤差をチェックします:

```python
--8<-- "examples/test_ik_roundtrip.py"
```

注目すべき点:

- `robot.tcp('flange').tf` はフランジ座標系の 4×4 ワールド変換です。
- `robot.ik(pos, rotmat)` は位置を最初に、続いて回転行列を取ります。
- 解析的ソルバは *すべて* の分岐を返します。ラウンドトリップは入力構成を
  （関節限界 / ラップの等価性を除いて）サブミリメートルの誤差で復元するはずです。

## 機構を可視化する { #visualizing-the-mechanism }

`KineVisualizer` は運動学的 **スケルトン**（関節軸を固定子/回転子のシリンダとして、
リンクをロッドとして）を描画します。`chain` を渡すとそのチェーンだけを描画し、
`chain=None` で機構全体を描画します。ロボットの既に計算済みのリンクごとの
ワールド変換を読み取るため、直列アームだけでなく分岐したヒューマノイドでも正しく
動作します。`robot.alpha` を設定してメッシュを半透明にすると、スケルトンが
透けて見えます:

```python
--8<-- "examples/test_cvr038_kine_visualizer.py"
```

## 関連項目

- [`one.utils.math`](../api/utils.md) — `rotmat_from_*`、`tf_from_pos_rotmat`、
  `frame_from_normal`、クォータニオン/オイラー/rotvec 変換、slerp。
- [`one.robots`](../api/robots.md) — ロボットクラスと `kine_visualizer`。
- [End-effectors & tool change](end_effectors.md) — 取り付けられたグリッパの
  `grasp_center` へのクロスオブジェクト IK。
