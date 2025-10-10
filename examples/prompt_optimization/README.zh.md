# 提示词优化示例

本目录包含使用进化算法优化 LLM 提示词 **模板** 的示例。

## 概述

`PromptOptimizationTask` 是一个 **字符串优化任务**，演示如何使用 EvoToolkit 进化提示词模板以提升 LLM 性能。

### 与 Python 任务的关键区别

| 方面 | Python 任务 | String 任务（提示词优化） |
|------|-------------|---------------------------|
| **解决方案类型** | Python 代码 | 字符串模板 |
| **进化目标** | 函数/算法 | 提示词文本 |
| **评估方式** | 执行代码 | 用 LLM 测试模板 |
| **示例** | `def func(x): return x**2` | `"求解：{question}\n答案："` |

## 快速开始

### 基础示例

使用 mock LLM 运行基础示例（无需 API）：

```bash
python basic_example.py
```

**输出：**
```
初始提示词模板：
  "Answer this question: {question}"
初始得分：100.00%

自定义模板："Solve this math problem and give only the number: {question}"
得分：100.00%

与 Python 任务的关键区别：
- 解决方案是字符串模板（而非 Python 代码）
- 模板使用 {question} 占位符
- 进化直接优化提示词字符串
```

### 使用真实 LLM

1. 在 `basic_example.py` 中直接配置 LLM API 凭证：
   ```python
   llm_api = HttpsApi(
       api_url="api.openai.com",  # 你的 API URL
       key="your-api-key-here",   # 你的 API 密钥
       model="gpt-4o"
   )
   ```

2. 设置 `use_mock=False` 以使用真实 LLM

3. 运行示例或进化代码

## 模板格式

提示词模板是包含 `{question}` 占位符的字符串：

```python
# 正确示例
"回答这个问题：{question}"
"解答这道数学题：{question}\n只给出数字。"
"问题：{question}\n逐步思考并只提供最终答案。"

# 错误示例（缺少占位符）
"解决这个问题"  # ❌ 没有 {question} 占位符
"答案：42"      # ❌ 没有 {question} 占位符
```

## 任务结构

### 创建任务

```python
from evotoolkit.task import PromptOptimizationTask

test_cases = [
    {"question": "2+2等于多少？", "expected": "4"},
    {"question": "5*3等于多少？", "expected": "15"},
]

task = PromptOptimizationTask(
    test_cases=test_cases,
    use_mock=True  # 或使用 llm_api=your_api 配置真实 LLM
)
```

### 测试模板

```python
template = "求解：{question}\n只用数字回答："
result = task.evaluate_code(template)

print(f"得分：{result.score:.2%}")  # 准确率
print(f"正确数：{result.additional_info['correct']}/{result.additional_info['total']}")
```

### 运行进化

```python
import evotoolkit
from evotoolkit.task import EvoEngineerStringInterface

interface = EvoEngineerStringInterface(task)

result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)

# 找到的最佳模板
print(result.sol_string)
```

## 可用算法

```python
# EvoEngineer（推荐）
from evotoolkit.task import EvoEngineerStringInterface
interface = EvoEngineerStringInterface(task)

# EoH（启发式进化）
from evotoolkit.task import EoHStringInterface
interface = EoHStringInterface(task)

# FunSearch
from evotoolkit.task import FunSearchStringInterface
interface = FunSearchStringInterface(task)
```

## 使用场景

此任务可适配多种提示词优化场景：

1. **数学问题**：优化数学题求解提示词
2. **分类任务**：优化文本分类提示词
3. **翻译任务**：优化语言翻译提示词
4. **信息抽取**：优化信息提取提示词
5. **问答任务**：优化问答提示词

## 自定义

### 自定义评估逻辑

```python
class CustomPromptTask(PromptOptimizationTask):
    def _check_answer(self, response: str, expected: str) -> bool:
        # 自定义评估逻辑
        return your_check_function(response, expected)
```

### 不同的测试用例

```python
test_cases = [
    {"question": "分类：这部电影太棒了！", "expected": "正面"},
    {"question": "分类：这部电影太糟糕了！", "expected": "负面"},
]
```

## 架构

```
src/evotool/task/
├── python_task/          # Python 代码优化
├── cuda_engineering/     # CUDA 内核优化
└── string_optimization/  # 字符串优化（新功能！）
    ├── string_task.py
    ├── prompt_optimization/
    │   └── prompt_optimization_task.py
    └── method_interface/
        ├── evoengineer_interface.py
        ├── eoh_interface.py
        └── funsearch_interface.py
```

## 下一步

- 查看[教程](../../docs/tutorials/)了解更多高级用法
- 学习[自定义进化方法](../../docs/tutorials/customizing-evolution.md)
- 探索其他任务类型（Python、CUDA、科学回归）

## 注意事项

- **String 任务 vs Python 任务**：提示词优化是 STRING 任务，而非 Python 任务
- **模板语法**：必须包含 `{question}` 占位符
- **Mock 模式**：用于测试，无需 LLM API 成本
- **真实 LLM**：提供实际的提示词优化结果
