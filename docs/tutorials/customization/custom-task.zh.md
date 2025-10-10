# 自定义任务教程

学习如何在 EvoToolkit 中创建自己的优化任务。

---

## 概述

本教程将向您展示如何：

- 扩展 `PythonTask` 基类
- 实现自定义评估逻辑
- 将自定义任务与进化算法一起使用

!!! tip "完整示例代码"
    本教程提供完整可运行的示例（点击查看/下载）：

    - [:material-download: my_custom_task.py](https://github.com/pgg3/evotoolkit/blob/master/examples/custom_task/my_custom_task.py) - 完整的自定义任务示例

    本地运行：
    ```bash
    cd examples/custom_task
    python my_custom_task.py
    ```

---

## 前置条件

- 完成 [科学符号回归教程](../built-in/scientific-regression.zh.md)
- 理解 Python 类和继承

---

## 创建自定义任务

### 步骤 1: 定义任务类

```python
from evotoolkit.task.python_task import PythonTask
from evotoolkit.core import Solution, EvaluationResult
import numpy as np

class MyOptimizationTask(PythonTask):
    """特定问题优化的自定义任务"""

    def __init__(self, data, target, timeout_seconds=30.0):
        """
        使用特定于问题的数据初始化任务

        Args:
            data: 输入数据（NumPy 数组）
            target: 目标输出值（NumPy 数组）
            timeout_seconds: 代码执行超时时间（秒）
        """
        self.target = target
        super().__init__(data, timeout_seconds)

    def _process_data(self, data):
        """处理输入数据并创建 task_info"""
        self.data = data
        self.task_info = {
            'data_size': len(data),
            'description': '函数近似任务'
        }

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """评估候选代码并返回评估结果"""
        # 1. 执行代码
        namespace = {'np': np}
        exec(candidate_code, namespace)

        # 2. 检查函数是否存在
        if 'my_function' not in namespace:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Function "my_function" not found'}
            )

        evolved_func = namespace['my_function']

        # 3. 计算适应度（score 越高越好）
        predictions = np.array([evolved_func(x) for x in self.data])
        mse = np.mean((predictions - self.target) ** 2)
        score = -mse  # 负 MSE，越高越好

        return EvaluationResult(
            valid=True,
            score=score,
            additional_info={'mse': mse}
        )

    def get_base_task_description(self) -> str:
        """获取任务描述供 prompt 生成使用"""
        return """你是函数近似专家。

任务：创建一个函数 my_function(x)，使其输出尽可能接近目标值。

要求：
- 定义函数 my_function(x: float) -> float
- 使用数学运算：+, -, *, /, **, np.exp, np.log, np.sin, np.cos 等
- 确保数值稳定性

示例代码：
    import numpy as np

    def my_function(x):
        return np.sin(x)
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """创建初始解"""
        initial_code = '''import numpy as np

def my_function(x):
    """简单线性函数作为基线"""
    return x
'''
        eval_res = self.evaluate_code(initial_code)
        return Solution(
            sol_string=initial_code,
            evaluation_res=eval_res
        )
```

**关键点：**

- 继承 `PythonTask` 而不是直接继承 `BaseTask`
- 实现 `_evaluate_code_impl()` 返回 `EvaluationResult` 对象
- 实现 `get_base_task_description()` 提供任务描述
- 实现 `make_init_sol_wo_other_info()` 创建初始解
- 使用 `_process_data()` 设置 `task_info`
- `score` 越高越好（使用负 MSE）

---

## 步骤 2: 使用自定义任务

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi
import numpy as np
import os

# 创建任务实例
data = np.linspace(0, 10, 50)
target = np.sin(data)  # 目标：近似正弦函数

task = MyOptimizationTask(data, target)

# 创建接口
interface = EvoEngineerPythonInterface(task)

# 设置 LLM
llm_api = HttpsApi(
    api_url=os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
    key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
    model="gpt-4o"
)

# 求解
result = evotoolkit.solve(
    interface=interface,
    output_path='./results/custom_task',
    running_llm=llm_api,
    max_generations=10
)

print(f"最佳得分: {result.evaluation_res.score:.4f}")
print(f"最佳 MSE: {result.evaluation_res.additional_info['mse']:.4f}")
```

---

## 示例：字符串匹配任务

```python
from evotoolkit.task.python_task import PythonTask
from evotoolkit.core import Solution, EvaluationResult

class StringMatchTask(PythonTask):
    """进化生成目标字符串的函数的任务"""

    def __init__(self, target_string, timeout_seconds=30.0):
        self.target = target_string
        super().__init__(data={'target': target_string}, timeout_seconds=timeout_seconds)

    def _process_data(self, data):
        """处理输入数据"""
        self.data = data
        self.task_info = {
            'target': self.target,
            'target_length': len(self.target)
        }

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """评估代码"""
        namespace = {}
        exec(candidate_code, namespace)

        if 'generate_string' not in namespace:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Function "generate_string" not found'}
            )

        try:
            generated = namespace['generate_string']()
            # 编辑距离越小越好，所以用负值作为 score
            distance = self.levenshtein_distance(generated, self.target)
            score = -distance  # 越高越好

            return EvaluationResult(
                valid=True,
                score=score,
                additional_info={'distance': distance, 'generated': generated}
            )
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': str(e)}
            )

    def levenshtein_distance(self, s1, s2):
        """计算 Levenshtein 编辑距离"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_base_task_description(self) -> str:
        """任务描述"""
        return f"""你是字符串生成专家。

任务：创建一个函数 generate_string()，生成目标字符串 "{self.target}"。

要求：
- 定义函数 generate_string() -> str
- 函数应返回与目标字符串尽可能接近的字符串

示例代码：
    def generate_string():
        return "Hello, World!"
"""

    def make_init_sol_wo_other_info(self) -> Solution:
        """创建初始解"""
        initial_code = f'''def generate_string():
    """初始简单实现"""
    return ""
'''
        eval_res = self.evaluate_code(initial_code)
        return Solution(
            sol_string=initial_code,
            evaluation_res=eval_res
        )
```

**用法:**

```python
task = StringMatchTask("Hello, EvoToolkit!")
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface, './results', llm_api)
print(f"生成的字符串: {result.evaluation_res.additional_info['generated']}")
```

---

## 最佳实践

### 1. 健壮的错误处理

```python
def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
    """在 _evaluate_code_impl 中实现健壮的错误处理"""
    try:
        # 执行和评估逻辑
        namespace = {}
        exec(candidate_code, namespace)
        # ... 评估逻辑 ...

        return EvaluationResult(
            valid=True,
            score=score,
            additional_info={}
        )
    except SyntaxError as e:
        return EvaluationResult(
            valid=False,
            score=float('-inf'),
            additional_info={'error': f'Syntax error: {str(e)}'}
        )
    except Exception as e:
        return EvaluationResult(
            valid=False,
            score=float('-inf'),
            additional_info={'error': f'Evaluation error: {str(e)}'}
        )
```

**注意：** PythonTask 的父类方法 `evaluate_code()` 已经提供了超时控制，在构造函数中设置 `timeout_seconds` 参数即可。

### 2. 验证解输出

```python
def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
    """验证函数输出的类型和范围"""
    namespace = {}
    exec(candidate_code, namespace)

    evolved_func = namespace['my_function']
    result = evolved_func(test_input)

    # 验证类型
    if not isinstance(result, (int, float, np.ndarray)):
        return EvaluationResult(
            valid=False,
            score=float('-inf'),
            additional_info={'error': 'Invalid output type'}
        )

    # 验证范围
    if isinstance(result, np.ndarray):
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': 'Output contains NaN or Inf'}
            )

    # 计算适应度
    score = -abs(result - expected)  # 负误差，越高越好
    return EvaluationResult(valid=True, score=score, additional_info={})
```

### 3. 使用 task_info 存储任务元数据

```python
def _process_data(self, data):
    """在 task_info 中存储重要的任务元数据"""
    self.data = data
    self.task_info = {
        'data_size': len(data),
        'input_dim': data.shape[1] if len(data.shape) > 1 else 1,
        'description': '自定义优化任务',
        'metric': 'MSE',
        # 其他有用的元数据...
    }
```

---

## 高级：自定义接口

如果您需要更精细的控制，可以为不同的进化方法自定义接口（Interface）。不同的方法（如 EvoEngineer、FunSearch、EoH）有各自的接口实现，它们控制着 prompt 生成、LLM 响应解析等行为。

如需了解如何自定义进化方法和接口，请参阅 [自定义进化方法教程](customizing-evolution.zh.md)。

---

## 完整示例

参见 `examples/custom_task/my_custom_task.py` 获取完整的可运行示例。

---

## 下一步

- 尝试 [CUDA 任务教程](../built-in/cuda-task.zh.md) 进行 GPU 优化
- 探索 [高级用法](../advanced-overview.zh.md) 了解低级 API
- 查看 [API 参考](../../api/tasks.md) 了解 Task 类详情
