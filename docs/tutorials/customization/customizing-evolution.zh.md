# 自定义进化方法

学习如何通过修改 prompt 或开发全新算法来自定义 EvoToolkit 中的进化行为。

---

## 概述

EvoToolkit 中进化优化的质量由以下因素控制：

1. **进化方法**：算法框架（EvoEngineer、EoH、FunSearch）
2. **Interface**：任务与方法之间的桥梁，包含 prompt 逻辑
3. **Prompts**：发送给 LLM 的指令，用于引导解的生成

本教程涵盖两个层次的自定义：

- **级别 1：自定义 prompt** - 继承现有 Interface 并修改 prompt（推荐）
- **级别 2：开发新算法** - 创建全新的进化策略（高级）

---

## 级别 1：自定义 Prompt

### 1.1 理解 Interface

每个进化方法使用一个 **Interface** 类，它负责：

- 定义操作符（init、mutation、crossover 等）
- 通过 `get_operator_prompt()` 为每个操作符生成 LLM prompt
- 将 LLM 响应解析为解决方案

**可用的 Interface：**

| Interface | 方法 | 描述 |
|-----------|------|------|
| `EvoEngineerPythonInterface` | EvoEngineer | Python 任务的主要 LLM 驱动算法 |
| `EoHPythonInterface` | EoH | Python 任务的启发式进化 |
| `FunSearchPythonInterface` | FunSearch | Python 任务的函数搜索 |
| `EvoEngineerCUDAInterface` | EvoEngineer | CUDA 代码进化 |

### 1.2 检查现有 Prompt

在自定义之前，先查看现有 Interface 如何生成 prompt：

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
import inspect

# 创建一个 interface
interface = EvoEngineerPythonInterface(task)

# 查看 prompt 生成方法的源码
print(inspect.getsource(interface.get_operator_prompt))
```

这可以让您看到：

- prompt 中包含了哪些信息
- prompt 的结构是怎样的
- LLM 需要遵循什么格式

### 1.3 创建自定义 Interface

要自定义 prompt，从现有 Interface 继承并重写 `get_operator_prompt()`：

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.core import Solution
from typing import List

class CustomInterface(EvoEngineerPythonInterface):
    """带有修改过的 prompt 的自定义 Interface"""

    def get_operator_prompt(self, operator_name: str,
                           selected_individuals: List[Solution],
                           current_best_sol: Solution,
                           random_thoughts: List[str],
                           **kwargs) -> List[dict]:
        """重写此方法来自定义任何操作符的 prompt"""

        # 获取基础任务描述
        task_description = self.task.get_base_task_description()

        if operator_name == "mutation":
            # 自定义变异 prompt
            prompt = f"""你是一个专家优化器。
当前最佳解得分: {current_best_sol.evaluation_res.score:.5f}

你的任务: {task_description}

当前代码:
{current_best_sol.sol_string}

通过应用变异生成改进的解决方案。
重点关注: [在此添加您的自定义要求]

格式:
- name: 描述性名称
- code: [完整代码]
- thought: [推理过程]
"""
            return [{"role": "user", "content": prompt}]

        elif operator_name == "crossover":
            # 自定义交叉 prompt
            parent1, parent2 = selected_individuals[0], selected_individuals[1]
            prompt = f"""结合这两个解决方案...
父代 1 (得分 {parent1.evaluation_res.score:.5f}):
{parent1.sol_string}

父代 2 (得分 {parent2.evaluation_res.score:.5f}):
{parent2.sol_string}

创建一个结合两者优势的后代...
"""
            return [{"role": "user", "content": prompt}]

        # 其他操作符使用默认实现
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# 使用您的自定义 Interface
custom_interface = CustomInterface(task)
result = evotoolkit.solve(
    interface=custom_interface,
    output_path='./custom_results',
    running_llm=llm_api,
    max_generations=10
)
```

### 1.4 Prompt 工程最佳实践

自定义 prompt 时：

#### 1.4.1 明确要求

```python
# 模糊
prompt = "改进这段代码"

# 明确
prompt = """通过以下方式改进代码:
1. 降低计算复杂度
2. 保持数值稳定性
3. 确保边界情况的正确性"""
```

#### 1.4.2 提供上下文和示例

```python
prompt = f"""任务: {task_description}

好的做法:
- 使用向量化的 NumPy 操作
- 尽可能避免循环
- 处理边界情况（空数组、零值）

不好的做法:
- 对大数组使用显式 Python 循环
- 不检查零值就进行除法

当前代码:
{current_best_sol.sol_string}

生成改进版本..."""
```

#### 1.4.3 融入领域知识

```python
# 对于科学回归
prompt = """基于已知的物理/生物学原理建立方程:
- Monod 方程用于底物限制: μ = μmax * S / (Ks + S)
- Arrhenius 方程用于温度: k = A * exp(-Ea / RT)
- Logistic 增长用于种群动力学
..."""

# 对于 CUDA 优化
prompt = """应用 GPU 优化技术:
- 合并内存访问
- 对频繁访问的数据使用共享内存
- 最小化分支分歧
..."""
```

#### 1.4.4 根据操作符类型自定义

不同的操作符受益于不同的 prompt：

```python
def get_operator_prompt(self, operator_name, ...):
    if operator_name == "init":
        # 初始探索 - 鼓励多样性
        prompt = "探索多样化的解决方案方法..."

    elif operator_name == "mutation":
        # 局部搜索 - 小的改进
        prompt = "对当前解决方案进行渐进式改进..."

    elif operator_name == "crossover":
        # 组合特征 - 重组
        prompt = "结合两个父代解决方案的优势..."
```

---

## 级别 2：开发新算法

!!! warning "高级主题"
    本节面向希望实现全新进化策略的用户。大多数用户应该从级别 1（自定义 prompt）开始，这通常已经足够。

### 2.1 何时开发新算法

在以下情况下考虑开发新算法：

- 现有算法（EvoEngineer、EoH、FunSearch）不适合您的问题结构
- 您有特定领域的进化策略
- 您想研究新的 LLM 驱动优化方法
- 您需要完全不同的进化流程或选择机制

### 2.2 算法架构

EvoToolkit 使用三层架构来实现新算法：

```
┌─────────────────────────────────────────┐
│  第 1 层：算法类 (Algorithm)            │
│  - 继承 Method 基类                     │
│  - 实现 run() 方法（进化主循环）        │
│  - 定义 Config 类（算法配置）           │
│  位置：evo_method/your_algorithm/       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  第 2 层：通用 Interface 基类           │
│  - 必需方法：parse_response()           │
│  - 其他方法：由算法需求决定             │
│  位置：core/method_interface/           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  第 3 层：任务专用 Interface            │
│  - 继承通用 Interface 基类              │
│  - 实现特定任务类型的具体逻辑           │
│  位置：task/*/method_interface/         │
└─────────────────────────────────────────┘
```

!!! important "Interface 设计的灵活性"
    **核心要求：** `BaseMethodInterface` 只强制要求一个方法：

    - `parse_response(response_str)` - 解析 LLM 响应

    **算法特定方法：** 其他所有方法都由您的算法需求决定：

    - **基于操作符**（如 EvoEngineer）：`get_init_operators()`, `get_offspring_operators()`, `get_operator_prompt()`
    - **迭代式**（如 FunSearch）：`generate_evolution_prompt()`
    - **您的设计**：定义任何您的算法需要的方法

**现有算法示例：**

- **EvoEngineer**：`evo_method/evoengineer/evoengineer.py` (第1层) → `core/method_interface/evoengineer_interface.py` (第2层) → `task/python_task/method_interface/evoengineer_interface.py` (第3层)
- **EoH**：`evo_method/eoh/` → `core/method_interface/eoh_interface.py` → `task/python_task/method_interface/eoh_interface.py`
- **FunSearch**：`evo_method/funsearch/` → `core/method_interface/funsearch_interface.py` → `task/python_task/method_interface/funsearch_interface.py`

### 2.3 创建新算法

#### 步骤 1：创建算法类 (第 1 层)

算法类负责实现进化主循环和种群管理：

```python
from evotoolkit.core import Method, BaseConfig, Solution
from evotoolkit.tools.llm import HttpsApi
from typing import List

class MyAlgorithmConfig(BaseConfig):
    """算法配置类"""
    def __init__(
        self,
        interface,  # Interface 实例
        output_path: str,
        running_llm: HttpsApi,
        max_generations: int = 10,
        pop_size: int = 5,
        offspring_per_generation: int = 3,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(interface, output_path, verbose=kwargs.get('verbose', True))
        self.running_llm = running_llm
        self.max_generations = max_generations
        self.pop_size = pop_size
        self.offspring_per_generation = offspring_per_generation
        self.temperature = temperature


class MyAlgorithm(Method):
    """我的自定义进化算法 - 简单的迭代式进化"""

    def __init__(self, config: MyAlgorithmConfig):
        super().__init__(config)
        self.config = config

    def run(self):
        """主进化循环 - 实现您的进化策略"""
        self.verbose_title("MY ALGORITHM STARTED")

        # 1. 初始化：创建初始解
        if len(self.run_state_dict.sol_history) == 0:
            init_sol = self._get_init_sol()
            self.run_state_dict.sol_history.append(init_sol)
            self.run_state_dict.population.append(init_sol)
            self._save_run_state_dict()

        # 2. 主进化循环
        while self.run_state_dict.generation < self.config.max_generations:
            self.verbose_info(f"Generation {self.run_state_dict.generation}")

            current_best = max(self.run_state_dict.population,
                             key=lambda s: s.evaluation_res.score)

            # 生成新的候选解
            for i in range(self.config.offspring_per_generation):
                # 生成 prompt
                messages = self.config.interface.generate_evolution_prompt(
                    current_best=current_best,
                    population=self.run_state_dict.population,
                    generation=self.run_state_dict.generation
                )

                # 调用 LLM
                response, usage = self.config.running_llm.get_response(messages)

                # 解析响应
                new_sol = self.config.interface.parse_response(response)

                # 添加到历史和种群
                if new_sol.evaluation_res.valid:
                    self.run_state_dict.sol_history.append(new_sol)
                    self.run_state_dict.population.append(new_sol)

            # 种群管理：保留最优的个体
            self.run_state_dict.population.sort(
                key=lambda s: s.evaluation_res.score, reverse=True
            )
            self.run_state_dict.population = \
                self.run_state_dict.population[:self.config.pop_size]

            self.run_state_dict.generation += 1
            self._save_run_state_dict()

        # 3. 标记完成
        self.run_state_dict.is_done = True
        self._save_run_state_dict()
```

**文件位置：** 您可以将算法文件放在任何位置。如果要贡献到 EvoToolkit 库，建议放在 `src/evotool/evo_method/my_algorithm/` 目录下。

#### 步骤 2：创建通用 Interface 基类 (第 2 层)

通用 Interface 定义算法需要的核心方法。**重要：** `BaseMethodInterface` 只要求实现一个核心方法：

- `parse_response(response_str: str)` - 解析 LLM 响应

**其他方法完全由您的算法需求决定。** 不同算法有不同的结构：

- **基于操作符的算法**（如 EvoEngineer）：需要 `get_init_operators()`, `get_offspring_operators()`, `get_operator_prompt()`
- **迭代式算法**（如 FunSearch）：可能只需要 `generate_evolution_prompt()`
- **您的算法**：定义任何您需要的方法

**示例：简单的迭代式 Interface**

```python
from abc import abstractmethod
from typing import List
from evotoolkit.core import Solution, BaseTask
from evotoolkit.core.method_interface import BaseMethodInterface

class MyAlgorithmInterface(BaseMethodInterface):
    """我的算法的通用 Interface 基类"""

    def __init__(self, task: BaseTask):
        super().__init__(task)

    @abstractmethod
    def parse_response(self, response_str: str) -> Solution:
        """解析 LLM 响应（必需）"""
        pass

    @abstractmethod
    def generate_evolution_prompt(
        self,
        current_best: Solution,
        population: List[Solution],
        generation: int
    ) -> List[dict]:
        """
        为当前代生成进化 prompt（算法特定方法）

        Args:
            current_best: 当前最优解
            population: 当前种群
            generation: 当前代数

        Returns:
            List[dict]: LLM 消息列表
        """
        pass
```

**文件位置：** 您可以将 Interface 文件放在任何位置。如果要贡献到 EvoToolkit 库，建议放在 `src/evotool/core/method_interface/my_algorithm_interface.py`。

!!! note "设计原则"
    - **必需方法**：只有 `parse_response()` 是必需的
    - **自定义方法**：根据算法的进化策略添加任何需要的方法
    - **灵活性**：不要受限于现有算法的结构，设计最适合您问题的接口

#### 步骤 3：创建任务专用 Interface (第 3 层)

为特定任务类型实现具体逻辑：

```python
from evotoolkit.core import MyAlgorithmInterface, Solution
from evotoolkit.task.python_task import PythonTask
from typing import List
import re

class MyAlgorithmPythonInterface(MyAlgorithmInterface):
    """针对 Python 任务的 Interface 实现"""

    def __init__(self, task: PythonTask):
        super().__init__(task)

    def parse_response(self, response_str: str) -> Solution:
        """从 LLM 响应中解析 Python 代码"""
        # 提取 Python 代码块
        code_match = re.search(r'```python\n(.*?)\n```', response_str, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response_str

        # 评估代码
        eval_res = self.task.evaluate_code(code)

        return Solution(
            sol_string=code,
            evaluation_res=eval_res,
            other_info={'raw_response': response_str}
        )

    def generate_evolution_prompt(
        self,
        current_best: Solution,
        population: List[Solution],
        generation: int
    ) -> List[dict]:
        """为当前代生成进化 prompt"""

        task_description = self.task.get_base_task_description()

        # 构建 prompt
        prompt = f"""你是进化优化专家。

任务描述：
{task_description}

当前最优解 (得分: {current_best.evaluation_res.score:.5f}):
{current_best.sol_string}

当前代数: {generation}

请改进这个解决方案，生成一个性能更好的版本。关注：
- 算法效率和准确性
- 数值稳定性
- 边界情况处理

请提供改进的 Python 代码。"""

        return [{"role": "user", "content": prompt}]
```

**文件位置：** 您可以将任务专用 Interface 文件放在任何位置。如果要贡献到 EvoToolkit 库，建议放在 `src/evotool/task/python_task/method_interface/my_algorithm_interface.py`。

!!! tip "Prompt 中的代码块"
    在 prompt 字符串中引用代码时，直接插入代码文本即可，不要使用 markdown 代码块标记。LLM 能够理解代码结构，而 markdown 标记可能导致混淆。

#### 步骤 4：使用您的新算法

```python
import evotoolkit
from evotoolkit.task.python_task import MyAlgorithmPythonInterface
from evotoolkit.tools.llm import HttpsApi
import os

# 创建任务
task = MyTask(...)

# 创建任务专用 Interface
interface = MyAlgorithmPythonInterface(task)

# 配置 LLM
llm_api = HttpsApi(
    api_url=os.environ.get("LLM_API_URL"),
    key=os.environ.get("LLM_API_KEY"),
    model="gpt-4o"
)

# 使用 evotoolkit.solve() 会自动找到对应的算法类
# 或者手动创建：
from evotoolkit.evo_method.my_algorithm import MyAlgorithm, MyAlgorithmConfig

config = MyAlgorithmConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=20,
    temperature=0.8  # 您的自定义参数
)

algorithm = MyAlgorithm(config)
algorithm.run()

# 获取最佳解
best_sol = algorithm.run_state_dict.get_best_solution()
print(f"Best score: {best_sol.evaluation_res.score}")
```

#### 步骤 5：注册算法（可选）

如果希望算法能被 `evotoolkit.solve()` 自动识别，需要注册：

```python
from evotoolkit.registry import register_algorithm

@register_algorithm("my_algorithm", config=MyAlgorithmConfig)
class MyAlgorithm(Method):
    # ...
```

然后可以这样使用：

```python
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    algorithm="my_algorithm",  # 指定算法名称
    max_generations=20
)
```

### 2.4 学习现有算法实现

在实现您自己的算法之前，建议研究现有算法的源代码：

**推荐阅读顺序：**

1. **EvoEngineer 算法**（最完整的示例）
   - 算法类：`src/evotool/evo_method/evoengineer/evoengineer.py`
   - Config：`src/evotool/evo_method/evoengineer/run_config.py`
   - 通用 Interface：`src/evotool/core/method_interface/evoengineer_interface.py`
   - Python Interface：`src/evotool/task/python_task/method_interface/evoengineer_interface.py`

2. **EoH 算法**（更简单的示例）
   - 算法类：`src/evotool/evo_method/eoh/`
   - 通用 Interface：`src/evotool/core/method_interface/eoh_interface.py`
   - Python Interface：`src/evotool/task/python_task/method_interface/eoh_interface.py`

3. **FunSearch 算法**（不同的进化策略）
   - 算法类：`src/evotool/evo_method/funsearch/`
   - 通用 Interface：`src/evotool/core/method_interface/funsearch_interface.py`

**关键要点：**

- 算法类的 `run()` 方法定义主进化循环
- Interface 的 `get_operator_prompt()` 控制 LLM 交互
- Config 类管理算法超参数
- `_apply_operators_parallel()` 实现并行评估
- `_manage_population_size()` 管理种群大小

---

## 特定任务的自定义示例

### 3.1 科学回归

```python
class ScientificInterface(EvoEngineerPythonInterface):
    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        if operator_name == "mutation":
            prompt = f"""你是一个物理学家/生物学家，正在发现方程。

当前方程 (MSE: {current_best_sol.evaluation_res.score:.6f}):
{current_best_sol.sol_string}

使用已建立的原理:
- Monod: μ = μmax * S / (Ks + S)
- Arrhenius: k = A * exp(-Ea / RT)
- Michaelis-Menten 动力学
- Logistic 增长

约束:
- 确保量纲一致性
- 避免数值不稳定性
- 保持模型简洁

生成改进的方程..."""
            return [{"role": "user", "content": prompt}]
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)
```

### 3.2 CUDA 优化

```python
class CUDAInterface(EvoEngineerCUDAInterface):
    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        if operator_name == "mutation":
            prompt = f"""你是一个 GPU 优化专家。

当前 CUDA kernel (时间: {current_best_sol.evaluation_res.score:.3f}ms):
{current_best_sol.sol_string}

应用优化:
- 合并内存访问模式
- 对临时数据使用共享内存
- 减少 bank 冲突
- 最小化线程分歧
- 优化 block/grid 维度

生成优化的 kernel..."""
            return [{"role": "user", "content": prompt}]
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)
```

### 3.3 提示词工程

```python
class PromptOptimizationInterface(EvoEngineerPythonInterface):
    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        if operator_name == "mutation":
            prompt = f"""你是 LLM 提示词工程专家。

当前 prompt (得分: {current_best_sol.evaluation_res.score:.3f}):
{current_best_sol.sol_string}

改进策略:
- 添加清晰的指令和结构
- 提供相关示例
- 指定输出格式
- 包含约束和指导方针
- 使用适当的语气和风格

生成改进的 prompt..."""
            return [{"role": "user", "content": prompt}]
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)
```

---

## 测试和调试

### 4.1 记录 Prompt

要查看发送给 LLM 的 prompt：

```python
class DebugInterface(EvoEngineerPythonInterface):
    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):

        prompts = super().get_operator_prompt(operator_name, selected_individuals,
                                             current_best_sol, random_thoughts, **kwargs)

        # 记录 prompt 以进行调试
        print(f"\n{'='*60}")
        print(f"操作符: {operator_name}")
        print(f"PROMPT:\n{prompts[0]['content']}")
        print(f"{'='*60}\n")

        return prompts
```

### 4.2 验证自定义 Interface

在运行完整进化之前，测试您的 Interface：

```python
# 创建 interface
interface = CustomInterface(task)

# 获取初始解
init_sol = task.make_init_sol_wo_other_info()

# 测试每个操作符的 prompt 生成
for op in interface.get_offspring_operators():
    prompts = interface.get_operator_prompt(
        operator_name=op.name,
        selected_individuals=[init_sol],
        current_best_sol=init_sol,
        random_thoughts=[]
    )
    print(f"操作符 {op.name}:")
    print(prompts[0]['content'][:200] + "...")
    print()
```

---

## 下一步

- **实验**：尝试不同的 prompt 风格，看看哪种效果最好
- **分析**：比较不同自定义方案的结果
- **分享**：考虑将成功的自定义方案贡献给项目

**相关文档：**

- [科学回归教程](../built-in/scientific-regression.zh.md) - 应用示例
- [CUDA 任务教程](../built-in/cuda-task.zh.md) - GPU 代码优化
- [高级用法](../advanced-overview.zh.md) - 更多配置选项
- [API 参考](../../api/index.md) - 完整的 Interface API 文档
- [贡献指南](../../development/contributing.zh.md) - 分享您的自定义方法
