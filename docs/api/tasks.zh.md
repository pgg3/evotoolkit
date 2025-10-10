# 任务 API

任务定义优化问题以及如何评估候选解。

---

## 概览

EvoToolkit 提供三类任务：

- **Python 任务** - 优化 Python 代码函数
- **String 任务** - 优化文本/字符串解（如提示词）
- **CUDA 任务** - 优化 GPU 内核代码

---

## Python 任务

### PythonTask

Python 代码优化任务的基类。

参见专门页面：[PythonTask](tasks/python/python-task.md)。

---

### ScientificRegressionTask

从数据中发现数学方程的科学符号回归。

参见专门页面：[ScientificRegressionTask](tasks/python/scientific-regression.md)。

---

### AdversarialAttackTask

为黑盒模型进化对抗攻击算法。

**用法：**

```python
from evotoolkit.task.python_task.adversarial_attack import AdversarialAttackTask

# 使用 mock 评估创建任务
task = AdversarialAttackTask(
    model=None,  # 可选：PyTorch 模型
    test_loader=None,  # 可选：测试数据加载器
    attack_steps=1000,
    n_test_samples=10,
    timeout_seconds=300.0,
    use_mock=True  # 使用 mock 评估进行测试
)

# 评估攻击代码
code = '''
def draw_proposals(x, num_proposals, step_size):
    # 生成对抗性提案样本
    proposals = ...
    return proposals
'''

result = task.evaluate_code(code)
print(f"得分: {result.score}")  # 负的 L2 距离（越高越好）
```

**参数：**

- `model` (`any`，可选)：要攻击的目标模型。如果为 None，则使用 mock 评估。
- `test_loader` (`any`，可选)：包含测试样本的 DataLoader。如果为 None，则使用 mock 评估。
- `attack_steps` (`int`)：每个样本的攻击迭代次数（默认：1000）
- `n_test_samples` (`int`)：要评估的测试样本数量（默认：10）
- `timeout_seconds` (`float`)：执行超时（默认：300.0）
- `use_mock` (`bool`)：使用 mock 评估而不是真实攻击（默认：False）

**方法：**

- `evaluate_code(code: str) -> EvaluationResult`：评估攻击算法代码

详见 [对抗攻击教程](../tutorials/built-in/adversarial-attack.zh.md)。

---

## String 任务

### StringTask

基于字符串的优化任务的基类（如提示词优化）。

**用法：**

```python
from evotoolkit.task.string_optimization.string_task import StringTask
from evotoolkit.core import EvaluationResult, Solution

class MyStringTask(StringTask):
    def _evaluate_string_impl(self, candidate_string: str) -> EvaluationResult:
        # 评估字符串解
        score = self.compute_score(candidate_string)
        return EvaluationResult(
            valid=True,
            score=score,
            additional_info={}
        )

    def get_base_task_description(self) -> str:
        return "优化一个字符串解..."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("初始字符串")
```

**构造函数：**

```python
def __init__(self, data, timeout_seconds: float = 30.0)
```

**抽象方法：**

- `_evaluate_string_impl(candidate_string: str) -> EvaluationResult`
- `get_base_task_description() -> str`
- `make_init_sol_wo_other_info() -> Solution`

---

### PromptOptimizationTask

优化 LLM 提示词模板以提高任务性能。

**用法：**

```python
from evotoolkit.task.string_optimization.prompt_optimization import PromptOptimizationTask

# 定义测试用例
test_cases = [
    {"question": "2+2等于多少？", "expected": "4"},
    {"question": "5*3等于多少？", "expected": "15"}
]

# 创建任务
task = PromptOptimizationTask(
    test_cases=test_cases,
    llm_api=my_llm_api,  # 如果 use_mock=True 则可选
    timeout_seconds=30.0,
    use_mock=True  # 使用 mock LLM 进行测试
)

# 评估提示词模板
prompt_template = "解答这道数学题：{question}\n只给出数字。"
result = task.evaluate_code(prompt_template)
print(f"准确率: {result.score}")  # 正确率（0.0 到 1.0）
```

**参数：**

- `test_cases` (`List[Dict[str, str]]`)：包含 'question' 和 'expected' 键的测试用例
- `llm_api` (可选)：用于测试提示词的 LLM API 实例（如果 `use_mock=False` 则必需）
- `timeout_seconds` (`float`)：评估超时（默认：30.0）
- `use_mock` (`bool`)：使用 mock LLM 响应进行测试（默认：False）

**模板格式：**

提示词模板必须包含 `{question}` 占位符：

```python
# 有效模板
"回答这个问题：{question}"
"问：{question}\n答："

# 无效 - 缺少占位符
"回答这个问题"  # 错误！
```

**方法：**

- `evaluate_code(prompt_template: str) -> EvaluationResult`：评估提示词模板

详见 [提示词工程教程](../tutorials/built-in/prompt-engineering.zh.md)。

---

## CUDA 任务

### CudaTask

CUDA 内核优化任务的基类。

**用法：**

```python
from evotoolkit.task.cuda_engineering import CudaTask, CudaTaskInfoMaker, Evaluator

# 创建评估器
evaluator = Evaluator(temp_path='./temp')

# 创建任务信息
task_info = CudaTaskInfoMaker.make_task_info(
    evaluator=evaluator,
    gpu_type="RTX 4090",
    cuda_version="12.4",
    org_py_code=original_python_code,
    func_py_code=function_python_code,
    cuda_code=baseline_cuda_code
)

# 创建任务
task = CudaTask(data=task_info, temp_path='./temp')

# 评估 CUDA 代码
eval_res = task.evaluate_code(candidate_cuda_code)
print(f"运行时间: {-eval_res.score:.4f}s")  # 得分为负运行时间
```

**构造函数：**

```python
def __init__(self, data, temp_path=None, fake_mode: bool = False)
```

**参数：**

- `data` (`dict`)：来自 `CudaTaskInfoMaker.make_task_info()` 的任务信息
- `temp_path` (`str`，可选)：CUDA 编译的临时路径
- `fake_mode` (`bool`)：跳过实际的 CUDA 评估（默认：False）

**方法：**

- `evaluate_code(code: str) -> EvaluationResult`：评估 CUDA 内核代码并返回结果，得分为负运行时间（得分越高 = 内核越快）

**注意：** CUDA 任务需要 `cuda_engineering` 额外依赖：

```bash
pip install evotoolkit[cuda_engineering]
```

详见 [CUDA 任务教程](../tutorials/built-in/cuda-task.zh.md)。

---

## 数据管理

首次访问时，数据集会自动从 GitHub releases 下载。

### Python API

```python
from evotoolkit.data import get_dataset_path, list_available_datasets

# 获取数据集路径（如果不存在则自动下载）
base_dir = get_dataset_path('scientific_regression')

# 访问特定数据集
bactgrow_path = base_dir / 'bactgrow'
train_csv = bactgrow_path / 'train.csv'

# 使用自定义目录
base_dir = get_dataset_path('scientific_regression', data_dir='./my_data')

# 列出类别中的所有数据集
datasets = list_available_datasets('scientific_regression')
print(datasets.keys())  # dict_keys(['bactgrow', 'oscillator1', 'oscillator2', 'stressstrain'])
```

**可用函数：**

- `get_dataset_path(category, data_dir=None)` - 获取数据集路径，如需要则自动下载
- `list_available_datasets(category)` - 列出类别中的所有数据集

**默认位置：** `~/.evotool/data/`

---

## 创建自定义任务

### Python 任务示例

```python
from evotoolkit.task.python_task import PythonTask
from evotoolkit.core import EvaluationResult, Solution

class MyOptimizationTask(PythonTask):
    """自定义优化任务"""

    def __init__(self, data, target):
        self.data = data
        self.target = target
        super().__init__(data={'data': data, 'target': target}, timeout_seconds=30.0)

    def _evaluate_code_impl(self, candidate_code: str) -> EvaluationResult:
        """评估解并返回结果（得分越高越好）"""
        # 1. 执行解代码
        namespace = {}
        try:
            exec(candidate_code, namespace)
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': str(e)}
            )

        # 2. 提取函数
        if 'my_function' not in namespace:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': '未找到函数 "my_function"'}
            )

        func = namespace['my_function']

        # 3. 计算适应度（负 MSE，使得越高越好）
        try:
            predictions = [func(x) for x in self.data]
            mse = sum((p - t)**2 for p, t in zip(predictions, self.target)) / len(self.data)
            score = -mse
            return EvaluationResult(
                valid=True,
                score=score,
                additional_info={'mse': mse}
            )
        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=float('-inf'),
                additional_info={'error': str(e)}
            )

    def get_base_task_description(self) -> str:
        return "优化一个函数以拟合数据..."

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("def my_function(x): return x")
```

详见 [自定义任务教程](../tutorials/customization/custom-task.zh.md)。

---

## 任务选择指南

| 任务类型 | 推荐类 | 用例 |
|----------|--------|------|
| 科学方程发现 | `ScientificRegressionTask` | 从数据中发现数学模型 |
| 对抗攻击 | `AdversarialAttackTask` | 进化攻击算法 |
| 提示词优化 | `PromptOptimizationTask` | 优化 LLM 提示词 |
| Python 代码 | `PythonTask` | 通用 Python 优化 |
| 字符串优化 | `StringTask` | 文本/配置优化 |
| GPU 内核 | `CudaTask` | CUDA 性能优化 |
| 自定义问题 | `BaseTask` | 任何其他优化问题 |

---

## 下一步

- 探索 [方法 API](methods.md) 了解进化算法
- 查看 [接口 API](interfaces.md) 了解任务-算法连接
- 尝试 [科学符号回归教程](../tutorials/built-in/scientific-regression.zh.md)
- 学习创建 [自定义任务](../tutorials/customization/custom-task.zh.md)
