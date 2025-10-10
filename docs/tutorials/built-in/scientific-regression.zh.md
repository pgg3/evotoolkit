# 科学符号回归教程

学习如何使用 LLM 驱动的进化从真实科学数据集中发现数学方程。

!!! note "学术引用"
    科学符号回归任务和数据集基于 CoEvo 研究。如果您在学术工作中使用此功能，请引用：

    ```bibtex
    @misc{guo2024coevocontinualevolutionsymbolic,
        title={CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models},
        author={Ping Guo and Qingfu Zhang and Xi Lin},
        year={2024},
        eprint={2412.18890},
        archivePrefix={arXiv},
        primaryClass={cs.AI},
        url={https://arxiv.org/abs/2412.18890}
    }
    ```

!!! tip "完整示例代码"
    本教程提供了完整的可运行示例（点击查看/下载）：

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/scientific_regression/basic_example.py) - 基础示例
    - [:material-download: custom_prompt.py](https://github.com/pgg3/evotoolkit/blob/master/examples/scientific_regression/custom_prompt.py) - 自定义 prompt 示例
    - [:material-download: compare_algorithms.py](https://github.com/pgg3/evotoolkit/blob/master/examples/scientific_regression/compare_algorithms.py) - 对比不同算法
    - [:material-file-document: README.zh.md](https://github.com/pgg3/evotoolkit/blob/master/examples/scientific_regression/README.zh.md) - 示例说明和运行指南

    本地运行：
    ```bash
    cd examples/scientific_regression
    python basic_example.py
    ```

---

## 概述

本教程演示：

- 加载科学数据集用于符号回归
- 从数据中发现数学方程
- 自动优化方程参数
- 进化复杂的科学模型

---

## 安装

安装科学符号回归依赖：

```bash
pip install evotoolkit[scientific_regression]
```

这会安装：

- SciPy（用于参数优化）
- Pandas（用于数据加载）

**前置知识：**

- 基本了解符号回归概念
- 熟悉 NumPy 和 SciPy 使用

---

## 准备数据集

EvoToolkit 支持 **懒下载** - 首次使用时自动下载数据集到默认位置。

**可用数据集：**

- **bactgrow**: 大肠杆菌细菌生长率预测（4输入：种群、底物、温度、pH）
- **oscillator1**: 阻尼非线性振荡器加速度（2输入：位置、速度）
- **oscillator2**: 阻尼非线性振荡器变体2（2输入：位置、速度）
- **stressstrain**: 铝棒应力预测（2输入：应变、温度）

**自定义数据目录：**

```python
# 在任务中指定数据目录（推荐）
task = ScientificRegressionTask(
    dataset_name="bactgrow",
    data_dir='./my_data'  # 首次运行时自动下载到此目录
)
```

---

## 示例：细菌生长建模

### 步骤 1: 创建任务

```python
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask

# 为细菌生长数据集创建任务
task = ScientificRegressionTask(
    dataset_name="bactgrow",
    max_params=10,          # 可优化参数数量
    timeout_seconds=60.0    # 每次评估超时时间
)

print(f"数据集: {task.dataset_name}")
print(f"训练集大小: {task.task_info['train_size']}")
print(f"测试集大小: {task.task_info['test_size']}")
```

**输出:**
```
数据集: bactgrow
训练集大小: 7500
测试集大小: 2500
输入数量: 4
```

### 步骤 2: 理解任务

科学符号回归任务的目标是 **从数据中发现数学方程** 。对于细菌生长数据集，我们需要找到一个函数来预测生长率。

**函数签名：** `equation(b, s, temp, pH, params) -> growth_rate`

**输入变量：**

- `b`: 种群密度
- `s`: 底物浓度
- `temp`: 温度
- `pH`: pH 值
- `params`: 可优化常数数组 (params[0] 到 params[9])

**评估流程：**

1. 您提供方程的结构（如 `params[0] * s / (params[1] + s)`）
2. 框架使用 `scipy.optimize.minimize` 自动优化参数值
3. 在测试集上计算 MSE（均方误差）作为适应度（越低越好）

### 步骤 3: 使用初始解测试

```python
# 获取初始解（简单线性模型）
init_sol = task.make_init_sol_wo_other_info()

print("初始解代码:")
print(init_sol.sol_string)

# 评估它
result = task.evaluate_code(init_sol.sol_string)
print(f"得分: {result.score:.6f}")
print(f"测试 MSE: {result.additional_info['test_mse']:.6f}")
```

**输出:**
```python
初始解代码:
import numpy as np

def equation(b, s, temp, pH, params):
    """线性基准模型。"""
    return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]

得分: 0.017200
测试 MSE: 0.017200
```

### 步骤 4: 尝试自定义初始解

您可以提供自定义的初始方程作为进化的起点。例如，这是一个基于生物学机制的复杂模型：

```python
custom_code = '''import numpy as np

def equation(b, s, temp, pH, params):
    """具有生物学机制的非线性细菌生长模型。"""

    # Monod 方程用于底物限制
    growth_rate = params[0] * s / (params[1] + s)

    # 高斯温度效应
    optimal_temp = params[4]
    temp_effect = params[2] * np.exp(-params[3] * (temp - optimal_temp)**2)

    # 高斯 pH 效应
    optimal_pH = params[7]
    pH_effect = params[5] * np.exp(-params[6] * (pH - optimal_pH)**2)

    # 带环境容量的 logistic 生长
    carrying_capacity = params[9]
    density_limit = params[8] * (1 - b / carrying_capacity)

    return growth_rate * temp_effect * pH_effect * density_limit
'''

result = task.evaluate_code(custom_code)
print(f"自定义模型得分: {result.score:.6f}")
print(f"测试 MSE: {result.additional_info['test_mse']:.6f}")
```

**输出:**
```
自定义模型得分: 0.021515
测试 MSE: 0.021515
```

!!! note "关于初始解"
    注意：在这里编写的任何自定义方程只是作为 **初始化解**。进化算法将使用 LLM 从这个起点开始生成和改进方程。最终的进化结果取决于所选的进化方法及其内部 prompt 设计。

### 步骤 5: 使用 EvoEngineer 运行进化

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi
import os

# 为 EvoEngineer 创建接口
interface = EvoEngineerPythonInterface(task)

# 配置 LLM API
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key-here",
    model="gpt-4o"
)

# 运行进化
result = evotoolkit.solve(
    interface=interface,
    output_path='./scientific_regression_results',
    running_llm=llm_api,
    max_generations=5,
    pop_size=10
)

print(f"找到最佳解！")
print(f"得分: {result['best_solution'].evaluation_res.score:.6f}")
print(f"代码:\n{result['best_solution'].sol_string}")
```

!!! tip "尝试其他算法"
    EvoToolkit 支持多种进化算法。只需更换 Interface 即可：

    ```python
    # 使用 EoH
    from evotoolkit.task.python_task import EoHPythonInterface
    interface = EoHPythonInterface(task)

    # 使用 FunSearch
    from evotoolkit.task.python_task import FunSearchPythonInterface
    interface = FunSearchPythonInterface(task)
    ```

    然后使用相同的 `evotoolkit.solve()` 调用运行进化。不同算法可能在不同任务上表现不同，建议多尝试对比。

---

## 自定义进化行为

进化过程的质量主要由 **进化方法** 及其内部 **prompt 设计** 控制。如果您想改进结果：

- **调整 prompt**: 继承现有 Interface 类并自定义 LLM prompt
- **开发新算法**: 创建全新的进化策略和操作符

!!! tip "深入学习"
    这些是适用于所有任务的通用技术。详细教程请参阅：

    - **[自定义进化方法](../customization/customizing-evolution.zh.md)** - 如何修改 prompt 和开发新算法
    - **[高级用法](../advanced-overview.zh.md)** - 更多高级配置选项

**快速示例 - 为科学回归自定义 prompt:**

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface

class ScientificRegressionInterface(EvoEngineerPythonInterface):
    """针对科学方程发现优化的 Interface，为 mutation 算子自定义 prompt"""

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """自定义 mutation 算子的 prompt，强调物理/生物学原理"""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            prompt = f"""你是一个科学方程发现专家。

任务: {task_description}

当前最佳方程 (得分: {current_best_sol.evaluation_res.score:.5f}):
{current_best_sol.sol_string}

要求: 生成改进的方程，必须基于已知物理/生物学原理（如Monod方程、Arrhenius方程等）。
确保数值稳定性和模型简洁性。

输出格式:
- name: 方程名称
- code: Python代码
- thought: 改进思路
"""
            return [{"role": "user", "content": prompt}]

        # init 和 crossover 算子使用父类的默认 prompt
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# 使用自定义 Interface
interface = ScientificRegressionInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

!!! note "关于 EvoEngineer 的算子"
    EvoEngineer 使用三个算子：**init**（初始化）、**mutation**（变异）、**crossover**（交叉）。
    父类 `EvoEngineerPythonInterface` 已经定义了这些算子和默认 prompt。
    你只需重写 `get_operator_prompt()` 来自定义特定算子的 prompt，其他算子会自动使用默认实现。

完整的自定义教程和更多示例，请参阅 [自定义进化方法](../customization/customizing-evolution.zh.md)。

---

## 理解评估

### 评分工作原理

1. **参数优化**: 通过使用 BFGS 方法的 `scipy.optimize.minimize` 优化参数来评估方程结构
2. **MSE 计算**: 预测值与真实值之间的均方误差
3. **适应度**: 负 MSE（越高越好，因此越低的 MSE = 越高的适应度）

### 评估输出

```python
result = task.evaluate_code(code)

if result.valid:
    print(f"得分: {result.score}")                           # 越高越好
    print(f"训练 MSE: {result.additional_info['train_mse']}")  # 训练数据上
    print(f"测试 MSE: {result.additional_info['test_mse']}")    # 测试数据上（用于适应度）
else:
    print(f"错误: {result.additional_info['error']}")
```

---

## 下一步

### 探索不同的任务和方法

- 尝试不同的数据集（oscillator1、oscillator2、stressstrain）
- 比较不同进化方法（EvoEngineer、EoH、FunSearch）的结果
- 可视化预测与真实值

### 自定义和改进进化过程

- 检查现有 Interface 类的 prompt 设计
- 继承并重写 Interface 来自定义 prompt
- 为不同操作符（init/mutation/crossover）设计专门的 prompt
- 如有需要，开发全新的进化算法

### 更多学习资源

- [自定义进化方法](../customization/customizing-evolution.zh.md) - 深入学习 prompt 自定义和算法开发
- [高级用法](../advanced-overview.zh.md) - 进阶配置和技巧
- [API 参考](../../api/index.md) - 完整的 API 文档
- [开发文档](../../development/contributing.zh.md) - 贡献新方法和特性
