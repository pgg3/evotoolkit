# 接口 API

接口将优化任务连接到进化算法，处理特定于算法的适配。

---

## 什么是接口？

**接口**是**任务**（您想要优化什么）和**方法**（如何优化）之间的桥梁。

```
任务（问题） → 接口（适配器） → 方法（算法）
```

接口处理：
- 为 LLM 生成特定于算法的提示
- 特定于任务的算子（变异、交叉等）
- 解格式转换
- 评估编排

---

## Python 任务接口

### EvoEngineerPythonInterface

```python
class EvoEngineerPythonInterface(BaseMethodInterface):
    def __init__(self, task: PythonTask)
```

将 Python 任务连接到 EvoEngineer 算法。

**用法:**

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
import evotoolkit

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

---

### FunSearchPythonInterface

```python
class FunSearchPythonInterface(BaseMethodInterface):
    def __init__(self, task: PythonTask)
```

将 Python 任务连接到 FunSearch 算法。

**用法:**

```python
from evotoolkit.task.python_task import FunSearchPythonInterface
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
import evotoolkit

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = FunSearchPythonInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

---

### EoHPythonInterface

```python
class EoHPythonInterface(BaseMethodInterface):
    def __init__(self, task: PythonTask)
```

将 Python 任务连接到 EoH 算法。

**用法:**

```python
from evotoolkit.task.python_task import EoHPythonInterface
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
import evotoolkit

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EoHPythonInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

---

## CUDA 任务接口

CUDA 任务接口可用于 GPU 内核优化任务。

**可用接口:**

- `EvoEngineerFullInterface` - 完整的 CUDA 工程工作流
- `EvoEngineerFreeInterface` - 自由形式的 CUDA 优化
- `EvoEngineerInsightInterface` - 洞察引导的 CUDA 优化
- `FunSearchInterface` - 用于 CUDA 的函数搜索
- `EoHInterface` - 用于 CUDA 的启发式进化

**用法示例:**

```python
from evotoolkit.task.cuda_engineering.method_interface import EvoEngineerFullInterface
import evotoolkit

task = MyCudaTask()
interface = EvoEngineerFullInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

详见 [CUDA 任务教程](../tutorials/built-in/cuda-task.md)。

---

## 基础接口类

### BaseMethodInterface

所有方法接口的基类。详见参考页：
[BaseMethodInterface](interfaces/base-method-interface.md)。

---

## 接口选择指南

### 对于 Python 任务

| 任务类型 | 推荐接口 | 替代选择 |
|-----------|----------------------|-------------|
| 科学符号回归 | `EvoEngineerPythonInterface` | `FunSearchPythonInterface` |
| 通用优化 | `EvoEngineerPythonInterface` | `EoHPythonInterface` |
| 快速原型 | `EoHPythonInterface` | `EvoEngineerPythonInterface` |

### 对于 CUDA 任务

| 任务类型 | 推荐接口 |
|-----------|-----------------------|
| 内核优化 | `EvoEngineerCudaInterface` |
| GPU 算法发现 | `FunSearchCudaInterface` |

---

## 接口工作原理

### 1. 提示生成

接口为 LLM 创建特定于算法的提示：

```python
# EvoEngineer 提示示例
prompt = """
您正在进化一个 Python 函数来近似数据。

上一代最佳解:
{previous_best_code}

当前适应度: {fitness}

请改进此解或创建新解。
"""
```

### 2. 响应解析

接口从 LLM 响应中提取代码：

```python
response = llm_api.call(prompt)
solution = interface.parse_llm_response(response)
# solution.sol_string 现在包含提取的 Python/CUDA 代码
```

### 3. 算子应用

接口应用进化算子：

```python
# 变异
mutated = interface.mutate(solution)

# 交叉
offspring = interface.crossover(parent1, parent2)
```

---

## 高级：自定义接口

为专门的算法或任务创建自定义接口：

```python
from evotoolkit.core.method_interface import BaseMethodInterface
from evotoolkit.core import Solution

class MySpecializedInterface(BaseMethodInterface):
    def __init__(self, task):
        super().__init__(task)
        self.custom_config = self.load_custom_config()

    def generate_prompt(self, generation, population):
        # 具有领域特定指令的自定义提示
        best_sol = max(population, key=lambda s: s.evaluation_res.score if s.evaluation_res.valid else float('-inf'))

        prompt = f"""
        领域特定上下文: {self.custom_config['context']}

        进化一个改进以下内容的解:
        {best_sol.sol_string}

        当前最佳得分: {best_sol.evaluation_res.score}
        代数: {generation}
        """
        return prompt

    def parse_llm_response(self, response):
        # 自定义解析逻辑
        code = self.extract_code_with_custom_markers(response)
        return Solution(code=code)

    def load_custom_config(self):
        # 加载领域特定配置
        return {"context": "自定义领域知识"}
```

**用法:**

```python
task = MyCustomTask()
interface = MySpecializedInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

---

## 比较：接口 vs 直接方法调用

### 使用接口（高级 API）✅ 推荐

```python
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface=interface, ...)
```

**优点:**
- 简单明了
- 自动配置
- 从接口推断算法

### 直接方法调用（低级 API）⚙️ 高级

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(interface=interface, ...)
algorithm = EvoEngineer(config)
algorithm.run()
```

**优点:**
- 完全控制配置
- 访问内部状态
- 自定义后处理

---

## 下一步

- 参见 [任务 API](tasks.md) 了解可用的优化任务
- 查看 [方法 API](methods.md) 了解进化算法
- 尝试 [高级用法教程](../tutorials/advanced-overview.md) 了解低级 API

