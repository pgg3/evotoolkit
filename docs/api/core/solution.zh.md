# Solution（解）

::: evotoolkit.core.Solution

---

## 示例

```python
from evotoolkit.core import Solution, EvaluationResult

eval_res = EvaluationResult(valid=True, score=0.95, additional_info={"generation": 3})
solution = Solution(
    sol_string="def f(x): return x**2",
    evaluation_res=eval_res,
    other_info={"method": "mutation"}
)

print(solution.sol_string)
print(f"得分: {solution.evaluation_res.score}")
```
