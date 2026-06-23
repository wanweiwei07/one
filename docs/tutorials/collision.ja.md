# 干渉

`one` は同じインターフェースを持つ2つの干渉バックエンドを提供します:

- `one.collider.cpu_simd`（`occs`） — numpy/SIMD による三角形メッシュ検出。
- `one.collider.gpu_simd_batch` — GPU バッチ版（把持計画器はまずこれを試し、
  CPU にフォールバックします）。

ロボットレベルのクエリ（ロボット全体を環境に対して多数の構成で）については
`one.collider.mj_collider.MJCollider`（`ocm`）があり、ロボット + 静的障害物を
一度コンパイルしてから `is_collided(qs)` に応答します。

## MJCollider によるロボット対地面

ロボットと障害物から collider を構築し、どのボディが動く **アクター** かを示し、
`compile()` を一度行い、それから構成をプローブします:

```python
--8<-- "examples/test_collision_check.py"
```

注目すべき点:

- `mjc.append(robot)` / `mjc.append(ground)` はボディを登録します。
  `mjc.actors = [robot]` は何が動くかを示します。`mjc.compile(margin=...)` は
  ブロードフェーズを確定します。
- `mjc.is_collided(qs)` はロボットを `qs` に駆動してブール値を返します — 計画
  ループ内で呼ぶのに安価です。
- `MJCollider` にはロボット、その取り付けられたグリッパ、および静的環境のみを
  登録してください。操作される／保持される部品は collider ボディとして *追加しません*。

## より低レベルのメッシュ対メッシュ

単発のメッシュペアには、`occs.create_detector()` + `occs.build_batch(items, pairs)`
+ `detector.detect_collision_batch(batch)` が接触点（または `None`）を返します。
これはまさに把持計画器が干渉する姿勢を棄却するために使うものです
（[`one.grasp._common.build_ee_target_detector`](../api/grasp.md) を参照）。

## 関連項目

- [`one.collider`](../api/collider.md) — CPU/GPU 検出器、MJCollider、バッチ。
- [Motion planning](motion_planning.md) — 計画器のオラクルとしての `is_collided`。
