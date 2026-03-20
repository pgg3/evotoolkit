# Quick Start

The intended `3.0.0` workflow is:

1. Define or import a task object.
2. Wrap it with a generic interface.
3. Instantiate an algorithm explicitly.
4. Call `run()`.

```python
from evotoolkit import EoH
from evotoolkit.task.python_task import EoHPythonInterface
from evotoolkit_tasks.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.tools import HttpsApi

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EoHPythonInterface(task)
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key",
    model="gpt-4o",
)

algo = EoH(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=5,
)
result = algo.run()
```
