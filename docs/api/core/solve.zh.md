# 显式 Runtime 工作流

`evotoolkit` 不再通过 `evotoolkit.solve(...)` 自动组装运行。

当前推荐的低层工作流是显式的：

1. 创建 task。
2. 用 interface 包装 task。
3. 显式实例化 `EoH`、`EvoEngineer` 或 `FunSearch`。
4. 调用 `run()`。

## 示例

```python
from evotoolkit import EvoEngineer
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit_tasks.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.tools import HttpsApi

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)

llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key",
    model="gpt-4o",
)

algo = EvoEngineer(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=5,
    max_sample_nums=10,
    pop_size=5,
)

result = algo.run()
print(f"Best score: {result.evaluation_res.score}")
```

如果需要继续已有运行，请重新创建 method 对象，然后在 `run()` 之前调用 `load_checkpoint()`。
