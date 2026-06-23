# 動作計画

`one.motion` はサンプリングベースのプランナ（`one.motion.probabilistic`）と
軌道ツール（`one.motion.trajectory`）を提供します。これらは、ロボットと干渉オラクル
（通常は `MJCollider.is_collided`）をラップする planning context によって駆動されます。

## RRT-Connect

開始コンフィグと目標コンフィグの間の関節空間クエリのための双方向ツリープランナで、
コライダが無効なサンプルとエッジを棄却します:

```python
--8<-- "examples/test_rrtc_rs007l.py"
```

注目すべき点:

- ロボット、地面、および任意の障害物はコライダに登録されます。planning context は
  `is_collided(qs)` をプランナの妥当性チェックに変換します。
- 結果は関節空間の経路です。近縁のバリアント
  （`test_prm_rs007l`、`test_rrtc_rs007l_dual`、`test_rrtc_rs007l_gpu_aabb`）は
  プランナまたは干渉バックエンドを差し替えます。

## デカルト軌道

tcp に直線的なデカルト経路をたどらせる必要がある場合（例: ワークピース上を掃引する
ツール）、`one.motion.interpolation.cartesian.linear_to_jpath` は直線に沿ってステップ
ごとの IK を解き、`time_param.retime_trapezoidal` が速度/加速度プロファイルを付加します。
この例ではドライバを取り付け、そのティップをデカルト区間に沿って駆動します:

```python
--8<-- "examples/test_rs007l_mount_orsd_jtraj.py"
```

注目すべき点:

- `linear_to_jpath(robot, start_pos, start_rotmat, goal_pos, goal_rotmat,
  tcp=...)` は関節経路とサンプリングされた姿勢を返します。ステップごとの IK が
  滑らかな分岐上にとどまるよう、バイアスのかかった `ref_qs` を渡してください。
- `retime_trapezoidal(q_seq, v_max, a_max, dt)` はウェイポイントを時間
  パラメータ化します。

## 関連項目

- [`one.motion`](../api/motion.md) — RRT、PRM、planning context、軌道ツール。
- [干渉](collision.md) — `is_collided` オラクルの構築。
