# 自定义任务教程

学习如何在 EvoToolkit 核心运行时之上构建一个自定义 `PythonTask`。

## 概述

本教程只保留一条最小路径：

- 定义一个 `PythonTask`
- 用 `TaskSpec` 描述任务
- 实现评估逻辑
- 用内置方法运行它

!!! tip "完整示例代码"
    可运行的参考示例是：

    - [:material-download: my_custom_task.py](https://github.com/pgg3/evotoolkit/blob/master/examples/custom_task/my_custom_task.py)

    本地运行：
    ```bash
    cd examples/custom_task
    uv run python my_custom_task.py
    ```

## 前置条件

- 具备基础的 Python 类与继承知识
- 已准备好可用的 LLM API 地址和密钥

## 步骤 1：定义 PythonTask

```python
import numpy as np

from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import PythonTask


class FunctionApproximationTask(PythonTask):
    def __init__(self, data, target, timeout_seconds=30.0):
        self.target = np.asarray(target, dtype=float)
        super().__init__(np.asarray(data, dtype=float), timeout_seconds=timeout_seconds)

    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="function_approximation",
            prompt=(
                "Write a Python function `my_function(x)` that maps a scalar input to a scalar output. "
                "The goal is to match the hidden target function as closely as possible."
            ),
            modality="python",
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        namespace = {"np": np}
        exec(candidate_code, namespace)  # noqa: S102

        if "my_function" not in namespace:
            return EvaluationResult(
                valid=False,
                score=float("-inf"),
                additional_info={"error": "Function `my_function` was not defined."},
            )

        predictions = np.array([namespace["my_function"](x) for x in self.data], dtype=float)
        mse = float(np.mean((predictions - self.target) ** 2))
        return EvaluationResult(valid=True, score=-mse, additional_info={"mse": mse})
```

## 步骤 2：用内置方法运行

```python
import os
import numpy as np

from evotoolkit import EvoEngineer
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi


data = np.linspace(0.0, 10.0, 50)
target = np.sin(data)

task = FunctionApproximationTask(data, target)
interface = EvoEngineerPythonInterface(task)
llm_api = HttpsApi(
    api_url=os.environ["LLM_API_URL"],
    key=os.environ["LLM_API_KEY"],
    model=os.environ.get("LLM_MODEL", "gpt-4o"),
)

algo = EvoEngineer(
    interface=interface,
    output_path="./results/custom_task_python",
    running_llm=llm_api,
    max_generations=5,
)
best_solution = algo.run()
```

## 关键点

- 用 `TaskSpec.prompt` 描述优化目标。
- 在 `__init__()` 中保存任务自己的数据。
- 所有打分逻辑都放在 `_evaluate_code_impl()` 里。
- 更好的候选解要返回更高的分数。

## 最佳实践

### 在打分前先验证候选代码

先检查目标函数是否存在、返回值是否合法，以及数值任务里是否出现 `NaN` 或 `Inf`。

### 尽量保持评估稳定

如果可以，评估器应尽量确定性。这样更容易比较实验、恢复 checkpoint 和调试。

### 不要把初始化逻辑塞进 task

如果你想给模型一个示例实现，把它放进 method prompt 或自定义 interface 里。task 只负责定义问题和评估器。

## 完整示例

参见 `examples/custom_task/my_custom_task.py` 获取完整的可运行示例。

## 下一步

- 查看 `docs/extensions.md` 了解当前扩展方式
- 如果在迁移旧任务，查看 `docs/migration.md`
- 查看 `docs/api/tasks.md` 和 `docs/api/interfaces.md` 了解底层 API
