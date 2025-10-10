# EvoEngineer

::: evotoolkit.evo_method.evoengineer.EvoEngineer

---

## Configuration

::: evotoolkit.evo_method.evoengineer.EvoEngineerConfig

---

## Usage

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
import evotoolkit

interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8
)
```
