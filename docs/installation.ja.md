# インストール

## 要件

- **Python 3.12** がサポートされるインタプリタです。Windows では
  `py -3.12` として呼び出してください。素の `python` は動作する `pyglet`
  ホイールを欠く新しいビルドを指している可能性があります。
- ビューア用に動作する OpenGL スタック。GUI の例（`base.run()`）の実行には
  ディスプレイが必要ですが、計画／幾何／IK のコード自体はヘッドレスで動作します。

コア依存関係（`pyproject.toml` で宣言され、自動的にインストールされます）:
`numpy`、`scipy`、`pyglet`、およびローダで使用されるメッシュ/URDF ツール群。

## インストール

リポジトリのルートから、editable インストールを行うと再インストールなしに
ローカルの変更が反映されます:

```bash
py -3.12 -m pip install -e .
```

## 検証

ヘッドレスのスモークテスト — パッケージをインポートして逆運動学クエリを
1つ解くだけで、ウィンドウは不要です:

```bash
py -3.12 -c "import numpy as np; from one import khi_rs007l; \
r = khi_rs007l.RS007L(); \
print('IK solutions:', len(r.ik((0.4, 0.0, 0.3), np.eye(3, dtype=np.float32))))"
```

これが非ゼロの解の数を表示すれば、運動学スタックが正しく接続されています。
ビューアを確認するには、任意の GUI の例を実行してください:

```bash
py -3.12 examples/test_rs007l_spawn.py
```

## 次へ

- [Quick Start](quickstart.md) — ロボットをスポーンし、メッシュを表示し、IK を解きます。
- [Tutorials](tutorials/kinematics.md) — トピックごとのウォークスルー。
- [API reference](api/index.md) — 各モジュールの公開関数とクラス。
