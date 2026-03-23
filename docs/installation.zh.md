# 安装

安装最新的稳定核心包：

```bash
pip install evotoolkit
```

这已经足够用于：

- 开发你自己的 `PythonTask` 或 `StringTask`
- 在自定义目标上运行内置算法
- 作为另一个任务包的运行时依赖

如果你是在当前仓库里开发：

```bash
uv sync --group dev
```

核心包本身不再附带具体应用任务依赖。如果你是在 EvoToolkit 之上构建任务包，请在自己的项目里单独声明这些依赖。
