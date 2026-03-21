# API 文档状态

这个 API 目录里仍然保留了一些 `1.0.0` 之前形成的页面，因此部分细节可能落后于当前稳定核心运行时。

当前公开接口建议先从下面这些页面开始：

- `../quickstart.md`
- `../extensions.md`
- `../migration.md`

当前真正的公开表面是：

- 显式算法类：`EoH`、`EvoEngineer`、`FunSearch`
- `Method`、`MethodState`、`RunStore`
- 通用 `PythonTask` / `StringTask`
- 通过 `load_checkpoint()` 显式恢复 checkpoint

- 浏览 [核心 API](core.md) 文档
- 探索 [任务 API](tasks.md) 了解通用任务抽象
- 查看 [方法 API](methods.md) 了解进化算法
- 了解 [接口 API](interfaces.md) 的算法集成
