# 快速开始

当前稳定 `1.0.2` 的工作流如下：

1. 定义一个 task。
2. 返回 `TaskSpec`。
3. 用 method interface 包装 task。
4. 显式实例化一个方法类。
5. 调用 `run()`。

```python
from evotoolkit import EvoEngineer
from evotoolkit.core import EvaluationResult, TaskSpec
from evotoolkit.task.python_task import EvoEngineerPythonInterface, PythonTask
from evotoolkit.tools import HttpsApi


class SquareTask(PythonTask):
    def build_python_spec(self, data) -> TaskSpec:
        return TaskSpec(
            name="square",
            prompt="Write a Python function `f(x)` that returns a numeric value.",
            modality="python",
        )

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        namespace = {}
        exec(candidate_code, namespace)  # noqa: S102
        if "f" not in namespace:
            return EvaluationResult(valid=False, score=float("-inf"), additional_info={"error": "Function `f` was not defined."})
        return EvaluationResult(valid=True, score=float(namespace["f"](3)), additional_info={})


task = SquareTask(data=None)
interface = EvoEngineerPythonInterface(task)
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key",
    model="gpt-4o",
)
algo = EvoEngineer(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=5,
)
best_solution = algo.run()
```

对于字符串问题，请使用 `StringTask`，并搭配 `EoHStringInterface`、`EvoEngineerStringInterface` 或 `FunSearchStringInterface`。

如果你想看一个可直接运行的完整示例，请参考仓库中的 `examples/custom_task/my_custom_task.py`。

## 断点续跑

EvoToolkit 在每次迭代结束后会自动保存 checkpoint。如果运行被中断，可以在调用 `run()` 前加载 checkpoint 来恢复进度：

```python
algo = EvoEngineer(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=20,
)

# 如果存在 checkpoint，则从断点恢复
if algo.checkpoint_exists():
    algo.load_checkpoint()

best_solution = algo.run()
```

每次运行会生成如下输出结构：

```
output_path/
├── checkpoint/
│   ├── state.pkl        # 完整算法状态（pickle 序列化）
│   └── manifest.json    # 元数据（算法类型、代数、运行状态等）
├── history/
│   └── gen_0.json, ...  # 每代的解记录
└── summary/
    └── best_per_generation.json, usage_history.json
```

checkpoint 包含完整的算法状态——种群、代数、解历史和 LLM 用量——因此续跑可以无缝衔接。
