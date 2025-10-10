# evotoolkit.solve()

::: evotoolkit.solve

---

## Example

```python
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

# Create task
task = ScientificRegressionTask(dataset_name="bactgrow")

# Create interface
interface = EvoEngineerPythonInterface(task)

# Configure LLM
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key",
    model="gpt-4o"
)

# Solve
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    max_sample_nums=10,
    pop_size=5
)

print(f"Best score: {result.evaluation_res.score}")
```
