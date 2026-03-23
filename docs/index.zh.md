# EvoToolkit Core

EvoToolkit 现在定位为面向 LLM 驱动进化搜索的核心运行时层。

核心包提供：

- 内置算法：`EoH`、`EvoEngineer`、`FunSearch`
- 生命周期基类：`Method`、`IterativeMethod`、`PopulationMethod`
- 通过 `RunStore` 提供的 checkpoint 与可读运行产物
- 通用 `PythonTask` 与 `StringTask` SDK
- 面向内置方法的通用 Python / 字符串 interface
- `evotoolkit.tools` 中的 OpenAI 兼容 HTTP 客户端工具

当前分支正在准备 `1.0.1rc1`，它是建立在 `v1.0.0` 稳定版本之上的 RC 体验线。这个仓库只描述可复用的 core 运行时。具体领域 task、数据集和硬件相关工作流应放在独立包或你自己的仓库里，在这个 core 之上扩展。
