# evotoolkit.solve()

::: evotoolkit.solve

---

## 示例

```python
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

# 创建任务
task = ScientificRegressionTask(dataset_name="bactgrow")

# 创建接口
interface = EvoEngineerPythonInterface(task)

# 配置 LLM
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key",
    model="gpt-4o"
)

# 求解
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    max_sample_nums=10,
    pop_size=5
)

print(f"最佳得分: {result.evaluation_res.score}")
```
