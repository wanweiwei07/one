# エンドエフェクタとツールチェンジ

エンドエフェクタも `MechBase` 機構です。挙動をミックスインし、作業用の tcp を
登録します:

- **グリッパ**（`GripperMixin`）は `'grasp_center'` tcp を登録し、
  `open()` / `close()` / `set_jaw_width()` および `grasp(child)` / `release(child)`
  を公開します。
- **ポイントツール**（`PointMixin`、例: ドライバ）は `'tip'` tcp を登録し、
  `touch_at(...)` / `attach(...)` を公開します。

## 取り付け

`arm.mount(ee, parent_lnk, loc_tf, update=True)` はエンドエフェクタをリンクに
取り付けます。以降は `arm.fk(...)` のたびに取り付けたツールへ伝播します。`update`
は*取り付けた瞬間に*ツールが所定の位置にスナップするかどうかのみを制御します。
動作中は常に追従します。`arm.unmount(ee)` はそれを取り外します（ツールはあった場所に
とどまります）。取り付けたツールで対象を拾うには、ツールの tcp に対して
**クロスオブジェクト IK** を解きます:

```python
qs = arm.ik(pos, rotmat, tcp=gripper.tcp('grasp_center'))
```

## 完全なツールチェンジシーケンス

この例では素の RS007L から始め、2FG7 グリッパを取り付けてバニーをピックアンド
プレースし、スタンドに戻して取り外し、それからドライバを取り付けてバニーの上で
作業します。mount / grasp / release / unmount を端から端まで実行します:

```python
--8<-- "examples/test_rs007l_toolchange.py"
```

注目すべき点:

- ツールの「スタンド」姿勢は単なるフランジ姿勢です。フランジがスタンドに位置する
  状態で恒等な `loc_tf` で取り付けると、跳ねのないピックアップになります。
- IK はアクティブなツールの tcp を狙います。グリッパが付いている間は
  `gripper.tcp('grasp_center')`、ドライバが付いている間は
  `screwdriver.tcp('tip')`、アームが素の間はデフォルトの `'flange'` です。
- `SceneObject` はデフォルトで `is_free = False` です。グリッパに取り付けられる
  よう、`grasp` の前に `bunny.is_free = True` を設定してください。

## 関連項目

- [`one.robots`](../api/robots.md) — `end_effectors`（グリッパ、ポイントツール）。
- [把持計画](grasp_planning.md) — 把持姿勢の出どころ。
- [運動学](kinematics.md) — チェーンと tcp。
