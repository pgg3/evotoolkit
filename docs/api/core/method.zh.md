# Method（算法基类）

::: evotoolkit.core.Method

---

如果你要实现自定义进化算法，可继承 `Method`：

```python
from evotoolkit.core import Method, BaseConfig

class MyCustomAlgorithm(Method):
    def run(self):
        for generation in range(self.config.max_generations):
            # 生成、评估、选择
            pass
```
