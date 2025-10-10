# 提示词工程教程

学习如何使用 LLM 驱动的进化算法来优化提示词模板，以提升下游任务性能。

!!! note "学术引用"
    如果您在研究中使用 EvoToolkit，请引用：

    ```bibtex
    @software{evotoolkit2025,
      title = {EvoToolkit: LLM-Driven Evolutionary Optimization},
      author = {Guo, Ping},
      year = {2025},
      url = {https://github.com/pgg3/evotoolkit}
    }
    ```

!!! tip "完整示例代码"
    本教程提供完整可运行的示例（点击查看/下载）：

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/prompt_optimization/basic_example.py) - 使用 mock LLM 的基础用法
    - [:material-file-document: README.zh.md](https://github.com/pgg3/evotoolkit/blob/master/examples/prompt_optimization/README.zh.md) - 示例文档和使用指南（中文版）

    本地运行：
    ```bash
    cd examples/prompt_optimization
    python basic_example.py
    ```

---

## 概述

本教程演示：

- 创建提示词优化任务
- 使用 LLM 驱动的进化改进提示词模板
- 在特定下游任务上测试提示词
- 自动进化高质量提示词

---

## 安装

安装 EvoToolkit：

```bash
pip install evotoolkit
```

**前置条件：**

- Python >= 3.11
- LLM API 访问权限（OpenAI、Claude 或其他兼容提供商）
- 提示词工程基础知识

---

## 理解提示词优化任务

### 什么是提示词优化任务？

提示词优化任务通过进化 **字符串模板** 来最大化下游任务的性能。与进化代码的 Python 任务不同，提示词任务直接进化提示词文本。

| 方面 | Python 任务 | 提示词任务 |
|------|-------------|-----------|
| **解决方案类型** | Python 代码 | 字符串模板 |
| **进化目标** | 函数/算法 | 提示词文本 |
| **评估方式** | 执行代码 | 用 LLM 测试模板 |
| **示例** | `def func(x): return x**2` | `"求解：{question}\n答案："` |

### 任务组件

一个提示词优化任务需要：

- **测试用例**：用于评估的问答对
- **模板语法**：包含 `{question}` 占位符的字符串
- **LLM API**：用于测试提示词模板（或使用 mock 模式）
- **评估指标**：在测试用例上的准确率

---

## 创建你的第一个提示词任务

### 步骤 1：定义测试用例

创建包含问题和预期答案的测试用例：

```python
test_cases = [
    {"question": "2+2等于多少？", "expected": "4"},
    {"question": "5*3等于多少？", "expected": "15"},
    {"question": "10-7等于多少？", "expected": "3"},
    {"question": "12/4等于多少？", "expected": "3"},
    {"question": "7+8等于多少？", "expected": "15"},
]
```

### 步骤 2：创建任务

```python
from evotoolkit.task import PromptOptimizationTask
from evotoolkit.tools.llm import HttpsApi

# 配置 LLM API
llm_api = HttpsApi(
    api_url="your_api_url",  # 例如: "ai.api.example.com"
    key="your_api_key",       # 你的 API 密钥
    model="gpt-4o"
)

task = PromptOptimizationTask(
    test_cases=test_cases,
    llm_api=llm_api,
    use_mock=False
)
```

### 步骤 3：测试初始模板

```python
# 获取初始解决方案
init_sol = task.make_init_sol_wo_other_info()

print(f"初始模板：{init_sol.sol_string}")
print(f"准确率：{init_sol.evaluation_res.score:.2%}")
print(f"正确数：{init_sol.evaluation_res.additional_info['correct']}/{init_sol.evaluation_res.additional_info['total']}")
```

**输出：**
```
初始模板："回答这个问题：{question}"
准确率：100.00%
正确数：5/5
```

### 步骤 4：测试自定义模板

```python
# 测试你自己的模板
custom_template = "解答这道数学题，只给出数字：{question}"
result = task.evaluate_code(custom_template)

print(f"自定义模板：{custom_template}")
print(f"准确率：{result.score:.2%}")
print(f"正确数：{result.additional_info['correct']}/{result.additional_info['total']}")
```

---

## 运行进化以优化提示词

### 步骤 1：创建接口

```python
import evotoolkit
from evotoolkit.task import EvoEngineerStringInterface

# 创建接口
interface = EvoEngineerStringInterface(task)
```

### 步骤 2：运行进化

```python
# 使用 LLM 运行进化
result = evotoolkit.solve(
    interface=interface,
    output_path='./prompt_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

print(f"找到的最佳模板：{result.sol_string}")
print(f"准确率：{result.evaluation_res.score:.2%}")
```

!!! tip "尝试不同的算法"
    EvoToolkit 支持多种提示词优化的进化算法：

    ```python
    # 使用 EoH
    from evotoolkit.task import EoHStringInterface
    interface = EoHStringInterface(task)

    # 使用 FunSearch
    from evotoolkit.task import FunSearchStringInterface
    interface = FunSearchStringInterface(task)

    # 使用 EvoEngineer（默认）
    from evotoolkit.task import EvoEngineerStringInterface
    interface = EvoEngineerStringInterface(task)
    ```

    然后使用相同的 `evotoolkit.solve()` 调用运行进化。不同的接口可能在不同的任务上表现更好。

---

## 理解模板格式

### 有效的模板

提示词模板必须包含 `{question}` 占位符：

```python
# ✅ 好的模板
"回答这个问题：{question}"
"解答这道数学题：{question}\n只给出数字。"
"问题：{question}\n逐步思考并只提供最终答案。"
"让我们来解决：{question}\n首先，分析问题..."

# ❌ 错误的模板（缺少占位符）
"解决这个问题"     # 没有 {question} 占位符
"答案：42"         # 没有 {question} 占位符
```

### 模板进化示例

在进化过程中，LLM 生成改进的模板：

```python
# 第 1 代
"答案：{question}"
# 准确率：60%

# 第 3 代
"解答这道数学题：{question}\n只提供数字答案。"
# 准确率：85%

# 第 7 代
"计算：{question}\n只显示最终数字，无需解释。"
# 准确率：100%
```

---

## 用例和应用

### 1. 数学问题求解

```python
test_cases = [
    {"question": "15 * 7 等于多少？", "expected": "105"},
    {"question": "144 / 12 等于多少？", "expected": "12"},
    # ...
]

task = PromptOptimizationTask(test_cases=test_cases, llm_api=llm_api)
```

### 2. 文本分类

```python
test_cases = [
    {"question": "这部电影太棒了！", "expected": "正面"},
    {"question": "这部电影太糟糕了！", "expected": "负面"},
    {"question": "我喜欢这部电影！", "expected": "正面"},
    # ...
]

task = PromptOptimizationTask(test_cases=test_cases, llm_api=llm_api)
```

### 3. 信息提取

```python
test_cases = [
    {"question": "提取日期：会议在 2024-03-15 举行", "expected": "2024-03-15"},
    {"question": "提取日期：我们将在 2024 年 3 月 20 日见面", "expected": "2024-03-20"},
    # ...
]

task = PromptOptimizationTask(test_cases=test_cases, llm_api=llm_api)
```

### 4. 翻译任务

```python
test_cases = [
    {"question": "翻译成英文：你好", "expected": "Hello"},
    {"question": "翻译成英文：谢谢", "expected": "Thank you"},
    # ...
]

task = PromptOptimizationTask(test_cases=test_cases, llm_api=llm_api)
```

---

## 自定义进化行为

进化提示词的质量主要由 **进化方法** 及其内部的 **提示设计** 控制。如果想提升结果：

- **调整提示**：继承现有的 Interface 类并自定义 LLM 提示
- **开发新算法**：创建全新的进化策略

!!! tip "了解更多"
    这些是适用于所有任务的通用技术。详细教程请参见：

    - **[自定义进化方法](../customization/customizing-evolution.zh.md)** - 如何修改提示和开发新算法
    - **[高级用法](../advanced-overview.zh.md)** - 更多高级配置选项

**快速示例 - 为提示词优化自定义提示：**

```python
from evotoolkit.task import EvoEngineerStringInterface

class CustomPromptInterface(EvoEngineerStringInterface):
    """优化提示词模板进化的接口。"""

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """自定义变异提示以强调清晰度和结构。"""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            individual = selected_individuals[0]

            prompt = f"""# 提示词模板优化

{task_description}

## 当前最佳模板
**准确率：** {current_best_sol.evaluation_res.score:.2%}
**模板：** {current_best_sol.sol_string}

## 待变异模板
**准确率：** {individual.evaluation_res.score:.2%}
**模板：** {individual.sol_string}

## 优化指南
通过以下方式改进模板：
- 添加清晰的指令
- 明确指定输出格式
- 包含相关的上下文或示例
- 使用恰当的语气和风格
- 确保保留 {{question}} 占位符

生成一个提高准确率的改进模板。

## 响应格式：
name: [描述性名称]
code:
[包含 {{question}} 占位符的改进模板]
thought: [修改理由]
"""
            return [{"role": "user", "content": prompt}]

        # 其他算子使用默认实现
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# 使用自定义接口
interface = CustomPromptInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./custom_results',
    running_llm=llm_api,
    max_generations=10
)
```

---

## 理解评估

### 评分机制

1. **模板测试**：每个模板在所有测试用例上进行测试
2. **LLM 响应**：LLM 使用模板生成答案
3. **答案检查**：响应与预期答案进行比较
4. **准确率计算**：得分 = (正确答案数) / (总测试用例数)

### 评估输出

```python
result = task.evaluate_code(template)

if result.valid:
    print(f"准确率：{result.score:.2%}")
    print(f"正确数：{result.additional_info['correct']}/{result.additional_info['total']}")
    print(f"详情：{result.additional_info['details']}")
else:
    print(f"错误：{result.additional_info['error_msg']}")
```

### 用于测试的 Mock 模式

使用 mock 模式测试而无需 LLM API 成本：

```python
# Mock 模式总是返回正确答案用于测试
task = PromptOptimizationTask(
    test_cases=test_cases,
    use_mock=True  # 不进行实际的 LLM 调用
)

# 适用于：
# - 测试任务设置
# - 调试模板格式
# - 理解工作流程
# - 开发自定义接口
```

---

## 自定义评估逻辑

对于专门的任务，你可以自定义答案检查：

```python
from evotoolkit.task import PromptOptimizationTask

class CustomPromptTask(PromptOptimizationTask):
    """具有专门答案检查的自定义任务。"""

    def _check_answer(self, response: str, expected: str) -> bool:
        """自定义评估逻辑。"""
        # 示例：不区分大小写的比较
        return response.strip().lower() == expected.strip().lower()

        # 示例：模糊匹配
        # from difflib import SequenceMatcher
        # similarity = SequenceMatcher(None, response, expected).ratio()
        # return similarity > 0.8

        # 示例：正则表达式匹配
        # import re
        # return bool(re.search(expected, response))

# 使用自定义任务
test_cases = [
    {"question": "法国的首都是？", "expected": "巴黎"},
    # "巴黎"、"PARIS"、"paris" 都被接受
]

task = CustomPromptTask(test_cases=test_cases, llm_api=llm_api)
```

---

## 完整示例

这是一个完整的工作示例：

```python
import evotoolkit
from evotoolkit.task import PromptOptimizationTask, EvoEngineerStringInterface
from evotoolkit.tools.llm import HttpsApi

# 1. 定义测试用例
test_cases = [
    {"question": "2+2等于多少？", "expected": "4"},
    {"question": "5*3等于多少？", "expected": "15"},
    {"question": "10-7等于多少？", "expected": "3"},
    {"question": "12/4等于多少？", "expected": "3"},
    {"question": "7+8等于多少？", "expected": "15"},
]

# 2. 配置 LLM API
llm_api = HttpsApi(
    api_url="your_api_url",  # 例如: "ai.api.example.com"
    key="your_api_key",       # 你的 API 密钥
    model="gpt-4o"
)

# 3. 创建任务
task = PromptOptimizationTask(
    test_cases=test_cases,
    llm_api=llm_api,
    use_mock=False
)

# 4. 创建接口
interface = EvoEngineerStringInterface(task)

# 5. 运行进化
result = evotoolkit.solve(
    interface=interface,
    output_path='./prompt_optimization_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

# 6. 显示结果
print(f"找到的最佳模板：")
print(f"  {result.sol_string}")
print(f"准确率：{result.evaluation_res.score:.2%}")
print(f"正确数：{result.evaluation_res.additional_info['correct']}/{result.evaluation_res.additional_info['total']}")
```

---

## 下一步

### 探索不同的优化策略

- 尝试不同的进化算法（EvoEngineer 变体、EoH、FunSearch）
- 比较不同接口的结果
- 实验不同的测试用例集
- 在各种下游任务上测试

### 自定义和改进进化过程

- 检查现有 Interface 类中的提示设计
- 继承并重写 Interface 以自定义提示
- 为不同的任务类型设计专门的提示
- 如有需要，开发全新的进化算法

### 了解更多

- [自定义进化方法](../customization/customizing-evolution.zh.md) - 深入了解提示自定义和算法开发
- [高级用法](../advanced-overview.zh.md) - 高级配置和技巧
- [API 参考](../../api/index.md) - 完整的 API 文档
- [开发文档](../../development/contributing.zh.md) - 贡献新方法和功能
