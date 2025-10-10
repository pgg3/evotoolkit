# FunSearch

::: evotoolkit.evo_method.funsearch.FunSearch

---

## Configuration

::: evotoolkit.evo_method.funsearch.FunSearchConfig

---

## Usage

```python
from evotoolkit.task.python_task import FunSearchPythonInterface
import evotoolkit

interface = FunSearchPythonInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=15
)
```
