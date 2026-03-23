# 从 1.0 之前的版本迁移

`v1.0.1` 是独立 EvoToolkit core 的当前稳定版本。`1.0.0` 首次确定了稳定的显式运行时表面，而 `1.0.1` 保持该运行时不变。

如果你之前使用的是 `1.0` 之前的版本或旧接口，最重要的变化如下：

- `evotoolkit.solve(...)` 已移除；现在需要显式实例化方法类并调用 `run()`
- core 包不再内置具体领域 task
- task 现在通过 `TaskSpec` 描述自己
- interface 不再负责构造初始解
- 初始化策略由具体方法或其提示设计负责

## Task API 对照

如果你有基于旧预发布 API 编写的自定义 task，可以按下面方式迁移：

- `get_base_task_description()` -> `TaskSpec.prompt`
- `_process_data()` -> 普通 `__init__()` 状态加 `build_*_spec()`
- `evaluate_code(...)` / `evaluate_string(...)` -> `evaluate()` 或 `_evaluate_*_impl()`

示例：

```python
from evotoolkit.core import TaskSpec
from evotoolkit.task.python_task import PythonTask


class MyTask(PythonTask):
    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="my_task",
            prompt="Describe the optimization target here.",
            modality="python",
        )
```

如果你以前在 task 层返回 baseline candidate，请移除这类 hook，把任何 bootstrap 示例直接放到方法提示或自定义 interface 逻辑里。

## 运行时用法对照

旧写法：

```python
result = evotoolkit.solve(interface=interface, output_path="./results", running_llm=llm_api)
```

新写法：

```python
from evotoolkit import EvoEngineer

algo = EvoEngineer(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=5,
)
result = algo.run()
```

如果你之前依赖早期预发布分支内置的领域 task，请先把这些 task 搬到你自己的包里，再在当前 core 运行时之上重新接入。
