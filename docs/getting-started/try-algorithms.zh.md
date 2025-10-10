# 尝试不同的算法

EvoToolkit 支持多种进化算法。试试所有算法：

---

## EoH（启发式进化）

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

下一步： [接下来做什么](next-steps.zh.md)
