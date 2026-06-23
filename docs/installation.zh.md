# 安装

## 要求

- **Python 3.12** 是受支持的解释器。在 Windows 上，请通过
  `py -3.12` 调用它 —— 裸 `python` 可能指向一个缺少可用 `pyglet`
  wheel 的较新版本。
- 查看器需要可用的 OpenGL 栈。运行 GUI 示例（`base.run()`）
  需要显示器；而规划/几何/IK 代码本身可以无头运行。

核心依赖（在 `pyproject.toml` 中声明，自动安装）：
`numpy`、`scipy`、`pyglet`，以及加载器所用的网格/URDF 工具。

## 安装

在仓库根目录下，可编辑安装会自动接收本地更改而无需
重新安装：

```bash
py -3.12 -m pip install -e .
```

## 验证

一个无头冒烟测试 —— 导入包并求解一次逆运动学
查询，无需窗口：

```bash
py -3.12 -c "import numpy as np; from one import khi_rs007l; \
r = khi_rs007l.RS007L(); \
print('IK solutions:', len(r.ik((0.4, 0.0, 0.3), np.eye(3, dtype=np.float32))))"
```

如果它打印出非零数量的解，说明运动学栈已正确连接。
要检查查看器，运行任意 GUI 示例：

```bash
py -3.12 examples/test_rs007l_spawn.py
```

## 下一步

- [快速开始](quickstart.md) —— 生成机器人、查看网格、求解 IK。
- [教程](tutorials/kinematics.md) —— 逐个主题的演练。
- [API 参考](api/index.md) —— 每个模块的公开函数和类。
