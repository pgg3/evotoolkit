# Method

::: evotoolkit.core.Method

---

If you're implementing a custom evolutionary algorithm, extend `Method`:

```python
from evotoolkit.core import Method, BaseConfig

class MyCustomAlgorithm(Method):
    def run(self):
        for generation in range(self.config.max_generations):
            # Generate, evaluate, select
            pass
```
