# Try Different Algorithms

EvoToolkit supports multiple evolutionary algorithms. Try them all:

---

## EoH (Evolution of Heuristics)

```python
from evotoolkit.task.python_task import EoHPythonInterface

interface = EoHPythonInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

---

## FunSearch

```python
from evotoolkit.task.python_task import FunSearchPythonInterface

interface = FunSearchPythonInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

---

Next: [Next Steps](next-steps.md)
