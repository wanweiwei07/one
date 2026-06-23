# MuJoCo 連携

`one.physics` はシーンを剛体シミュレーションのために **MuJoCo** へ橋渡しします。
`MJEnv` は `one` のシーンから構築されます。シーンのボディを MuJoCo モデルへ変換し、
物理をステップ実行し、得られた状態を同期して戻すことで、ビューアがシミュレートされた
動きを表示します。

## 平面上にロボットを落とす

この例では複数の RS007L クローンを地面平面の上に積み上げ、MuJoCo に落下を
シミュレートさせます。`mjenv.step` はビューアループ上にスケジュールされるため、
物理はフレームごとに進行します:

```python
--8<-- "examples/test_mujoco_rs007l.py"
```

注目すべき点:

- `MJEnv(scene=base.scene)` は、シーンに取り付けられているもの（ロボット、
  プリミティブ、平面）からシミュレーションを構築します。
- `mjenv.save("scene.xml")` は生成された MuJoCo モデルをエクスポートします。変換された
  シーンの検査や再利用に便利です。
- `base.schedule_interval(mjenv.step)` はシミュレーションをビューアと歩調を合わせて
  実行します。
- ロボット、グリッパ、静的環境を登録します。保持/操作される部品は、別個のシミュ
  ボディとして追加するのではなく、把持アタッチメントを通じて扱われます。

関連する例: `test_mujoco_bunny`、`test_mujoco_rs007l_and_bunny`、
`test_mujoco_rs007l_engage_2fg7`、`test_mujoco_xytheta`。

## 関連項目

- [`one.physics`](../api/physics.md) — `MJEnv`、接触、モデル変換。
- [干渉](collision.md) — 解析的な（非物理の）干渉パス。
