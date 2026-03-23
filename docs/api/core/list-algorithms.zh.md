# evotoolkit.list_algorithms()

::: evotoolkit.list_algorithms

---

返回当前包显式提供的内置算法名称。

## 示例

```python
import evotoolkit

algorithms = evotoolkit.list_algorithms()
for algo in algorithms:
    print(f"- {algo}")
```
