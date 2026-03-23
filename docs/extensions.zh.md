# 扩展

核心包刻意保持精简。推荐从两个方向扩展：

- 在 `PythonTask` 或 `StringTask` 之上定义你自己的 task
- 在 `Method`、`IterativeMethod` 或 `PopulationMethod` 之上定义你自己的算法

## 自定义 Task

推荐的 task 扩展流程是：

1. 创建一个 `PythonTask` 或 `StringTask` 子类。
2. 在 `build_python_spec()` 或 `build_string_spec()` 中返回 `TaskSpec`。
3. 实现对应模态的评估 hook。
4. 优先复用 `evotoolkit.task` 中的通用 interface；只有在提示词或响应契约确实不同的时候，才添加自定义 `MethodInterface`。
5. 在你自己的包里通过显式导入暴露这些 task。

## 自定义算法

新的算法通常从以下基类开始：

- `IterativeMethod`：适合一般的逐步搜索
- `PopulationMethod`：适合基于种群的代际搜索

如果你需要非标准生命周期，再退回到底层 `Method`。具体方法自行管理初始化策略，运行时只负责把 `task.spec` 传入方法状态。
