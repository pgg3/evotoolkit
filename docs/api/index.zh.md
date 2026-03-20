# 归档 API 入口

这个 API 目录属于旧的 `docs/` 文档树，很多页面仍然反映的是 `3.0.0` 之前的结构。

当前版本的核心 API 请改看 `docs_core/`，优先入口是：

- `docs_core/quickstart.md`
- `docs_core/extensions.md`
- `docs_core/migration.md`

当前真正的公开表面是：

- 显式算法类：`EoH`、`EvoEngineer`、`FunSearch`
- `Method`、`MethodState`、`RunStore`
- 通用 `PythonTask` / `StringTask`
- 通过 `load_checkpoint()` 显式恢复 checkpoint

- 浏览 [核心 API](core.md) 文档
- 探索 [任务 API](tasks.md) 了解内置任务
- 查看 [方法 API](methods.md) 了解进化算法
- 了解 [接口 API](interfaces.md) 的算法集成
