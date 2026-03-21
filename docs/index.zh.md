# EvoToolkit Core

EvoToolkit 现在定位为面向 LLM 驱动进化搜索的稳定核心运行时层。

核心包提供：

- 内置算法：`EoH`、`EvoEngineer`、`FunSearch`
- 生命周期基类：`Method`、`IterativeMethod`、`PopulationMethod`
- 通过 `RunStore` 提供的 checkpoint 与可读运行产物
- 通用 `PythonTask` 与 `StringTask` SDK
- 面向 Python 与字符串优化的通用 interface

核心包本身不再承载具体领域任务。具体 task 应该放在独立包或你自己的仓库里，在这个 core 之上扩展。
