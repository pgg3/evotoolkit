# EoH (Evolution of Heuristics)

::: evotoolkit.evo_method.eoh.EoH

---

## Configuration

::: evotoolkit.evo_method.eoh.EoHConfig

---

## Usage

```python
from evotoolkit.task.python_task import EoHPythonInterface
import evotoolkit

interface = EoHPythonInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    max_sample_nums=15
)
```
