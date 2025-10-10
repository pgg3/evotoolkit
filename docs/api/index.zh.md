# API 参考

欢迎来到 EvoToolkit API 参考文档。本节提供所有公共 API、类和函数的详细信息。

---

## 概述

EvoToolkit 组织为几个主要模块：

- **[核心 API](core.md)**: 核心功能，包括 `evotoolkit.solve()`、`Solution`、`Task` 和基类
- **[任务](tasks.md)**: 内置优化任务（Python 和 CUDA）
- **[方法](methods.md)**: 进化算法（EoH、EvoEngineer、FunSearch）
- **[接口](interfaces.md)**: 连接任务和算法的方法接口
- **[工具](tools.md)**: 实用工具和 LLM API 客户端

---

## 快速 API 参考

### 高级 API

使用 EvoToolkit 的最简单方式：

```python
import evotoolkit

result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

详见 [核心 API: evotoolkit.solve()](core/solve.md)。

### 核心类

| 类 | 描述 | 文档 |
|-------|-------------|---------------|
| `Solution` | 表示候选解 | [核心 API](core/solution.md) |
| `Task` | 优化任务基类 | [核心 API](core/base-task.md) |
| `MethodInterface` | 算法接口基类 | [接口](interfaces.md) |

### 内置任务

| 任务 | 描述 | 文档 |
|------|-------------|---------------|
| `ScientificRegressionTask` | 科学符号回归任务 | [任务](tasks.md#scientificregressiontask) |
| `PythonTask` | 通用 Python 任务 | [任务](tasks.md#pythontask) |
| `CudaTask` | GPU 内核优化任务 | [任务](tasks.md#cudatask) |

### 进化算法

| 算法 | 描述 | 文档 |
|-----------|-------------|---------------|
| `EvoEngineer` | 主要的 LLM 驱动进化算法 | [方法](methods.md#evoengineer) |
| `FunSearch` | 函数搜索优化 | [方法](methods.md#funsearch) |
| `EoH` | 启发式进化 | [方法](methods.md#eoh) |

---

## API 设计理念

EvoToolkit 提供两个级别的 API：

### 1. 高级 API（推荐）

通过 `evotoolkit.solve()` 的高级 API 自动处理大部分复杂性：

```python
# 创建任务和接口
task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)

# 求解
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

**优点**:
- 简单明了
- 自动配置
- 适合大多数用例

### 2. 低级 API（高级）

低级 API 提供细粒度控制：

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

# 创建自定义配置
config = EvoEngineerConfig(
    task=task,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    pop_size=10,
    # ... 更多自定义设置
)

# 创建并运行算法
algorithm = EvoEngineer(config)
algorithm.run()

# 获取最佳解
best_solution = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)
```

**优点**:
- 完全自定义
- 访问内部状态
- 高级调试

详见 [高级用法教程](../tutorials/advanced-overview.md)。

---

## 模块组织

```
evotool/
├── __init__.py              # 高级 API (solve 函数)
├── core/                    # 核心抽象
│   ├── base_task.py        # Task 基类
│   ├── solution.py         # Solution 类
│   ├── base_method.py      # 算法基类
│   ├── base_config.py      # 配置基类
│   └── method_interface/   # 算法接口
├── evo_method/             # 进化算法
│   ├── eoh/               # EoH 实现
│   ├── evoengineer/       # EvoEngineer 实现
│   └── funsearch/         # FunSearch 实现
├── task/                   # 任务实现
│   ├── python_task/       # Python 任务框架
│   ├── cuda_engineering/  # CUDA 任务框架
│   └── string_optimization/ # 字符串优化任务
├── tools/                  # 工具
│   └── llm.py             # LLM API 客户端 (HttpsApi)
└── data/                   # 数据管理工具
```

---

## 常见模式

### 模式 1: 基本优化

```python
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface, './results', llm_api, max_generations=5)
```

### 模式 2: 自定义任务

```python
from evotoolkit.core import BaseTask, Solution

class MyTask(BaseTask):
    def evaluate(self, solution: Solution) -> float:
        # 您的评估逻辑
        return fitness_value

task = MyTask()
interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface, './results', llm_api)
```

### 模式 3: 算法比较

```python
algorithms = [
    ('EoH', EoHPythonInterface(task)),
    ('EvoEngineer', EvoEngineerPythonInterface(task)),
    ('FunSearch', FunSearchPythonInterface(task))
]

for name, interface in algorithms:
    result = evotoolkit.solve(interface, f'./results/{name}', llm_api)
    print(f"{name}: {result.fitness}")
```

---

## API 版本控制

EvoToolkit 遵循[语义化版本](https://semver.org/lang/zh-CN/)：

- **主版本** (1.x.x): 破坏性 API 更改
- **次版本** (x.1.x): 新功能，向后兼容
- **修订版本** (x.x.1): Bug 修复，向后兼容

检查当前版本：

```python
import evotoolkit
print(evotoolkit.__version__)  # 例如 "1.0.0"
```

---

## 类型提示

EvoToolkit 在整个代码库中使用类型提示。使用 `mypy` 等类型检查器进行静态分析：

```bash
pip install mypy
mypy your_script.py
```

---

## 下一步

- 浏览 [核心 API](core.md) 文档
- 探索 [任务 API](tasks.md) 了解内置任务
- 查看 [方法 API](methods.md) 了解进化算法
- 了解 [接口 API](interfaces.md) 的算法集成
