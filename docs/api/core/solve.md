# Explicit Runtime Workflow

`evotoolkit` no longer assembles runs through `evotoolkit.solve(...)`.

The low-level workflow is now explicit:

1. Create a task.
2. Wrap it with an interface.
3. Instantiate a method such as `EoH`, `EvoEngineer`, or `FunSearch`.
4. Call `run()`.

## Example

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

To resume a run, create the method again with the same task and interface, then call `load_checkpoint()` before `run()`.
