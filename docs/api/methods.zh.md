# 方法 API

进化方法是驱动优化过程的核心算法。

---

## 可用算法

EvoToolkit 提供三种主要的进化算法：

| 算法 | 最适合 | 特征 |
|-----------|----------|-----------------|
| **EvoEngineer** | 通用优化 | 多功能、稳健、良好的默认选择 |
| **FunSearch** | 函数发现 | 专门用于函数近似 |
| **EoH** | 启发式优化 | 快速、高效，适合简单问题 |

---

## EvoEngineer

```python
class EvoEngineer(Method):
    def __init__(self, config: EvoEngineerConfig)
    def run(self)
```

EvoEngineer 是 EvoToolkit 中主要的 LLM 驱动进化算法。

### 高级 API 用法

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface
import evotoolkit

interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8
)
```

### 低级 API 用法

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8,
    elite_ratio=0.2,
    mutation_rate=0.3,
    crossover_rate=0.7
)

algorithm = EvoEngineer(config)
algorithm.run()
best = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)
```

### 配置参数

**进化参数:**
- `max_generations` (`int`): 最大进化代数
- `pop_size` (`int`): 种群大小
- `max_sample_nums` (`int`): 每代最大 LLM 采样数

**选择参数:**
- `elite_ratio` (`float`): 保留为精英的比例（默认: 0.2）
- `tournament_size` (`int`): 锦标赛选择大小（默认: 3）

**变异参数:**
- `mutation_rate` (`float`): 变异概率（默认: 0.3）
- `crossover_rate` (`float`): 交叉概率（默认: 0.7）

---

## FunSearch

```python
class FunSearch(Method):
    def __init__(self, config: FunSearchConfig)
    def run(self)
```

FunSearch 是一种专门用于函数发现的优化方法。

### 用法

```python
from evotoolkit.task.python_task import FunSearchPythonInterface
import evotoolkit

interface = FunSearchPythonInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=20,
    num_islands=4
)
```

### 低级 API

```python
from evotoolkit.evo_method.funsearch import FunSearch, FunSearchConfig

config = FunSearchConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=20,
    num_islands=4,
    island_size=8,
    migration_rate=0.1,
    migration_interval=5
)

algorithm = FunSearch(config)
algorithm.run()
```

### 配置参数

**岛模型参数:**
- `num_islands` (`int`): 并行进化岛数量（默认: 4）
- `island_size` (`int`): 每个岛的种群大小（默认: 8）
- `migration_rate` (`float`): 岛间迁移率（默认: 0.1）
- `migration_interval` (`int`): 迁移间隔代数（默认: 5）

---

## EoH

```python
class EoH(Method):
    def __init__(self, config: EoHConfig)
    def run(self)
```

EoH（启发式进化）是一种快速高效的进化方法。

### 用法

```python
from evotoolkit.task.python_task import EoHPythonInterface
import evotoolkit

interface = EoHPythonInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10
)
```

### 低级 API

```python
from evotoolkit.evo_method.eoh import EoH, EoHConfig

config = EoHConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    max_sample_nums=20,
    elite_ratio=0.3,
    heuristic_pool_size=50
)

algorithm = EoH(config)
algorithm.run()
```

### 配置参数

**启发式参数:**
- `heuristic_pool_size` (`int`): 启发式池大小（默认: 50）
- `elite_ratio` (`float`): 精英比例（默认: 0.3）

---

## 算法比较

### 性能特征

| 特征 | EvoEngineer | FunSearch | EoH |
|---------|-------------|-----------|-----|
| 收敛速度 | 中等 | 慢 | 快 |
| 解质量 | 高 | 很高 | 中等 |
| 计算成本 | 中等 | 高 | 低 |
| 适用范围 | 广泛 | 函数发现 | 启发式优化 |

### 何时使用

**EvoEngineer:**
- 通用优化问题
- 平衡性能和质量
- 不确定使用哪个时的默认选择

**FunSearch:**
- 需要高质量解
- 有足够计算预算
- 函数近似和发现任务

**EoH:**
- 快速原型设计
- 简单优化问题
- 有限的计算资源

---

## 自定义算法

实现您自己的进化算法：

```python
from evotoolkit.core import BaseMethod, BaseConfig

class MyAlgorithm(BaseMethod):
    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def run(self):
        # 初始化
        population = self.initialize_population()

        for gen in range(self.config.max_generations):
            # 生成新解
            new_solutions = self.generate_with_llm()

            # 评估
            for sol in new_solutions:
                eval_res = self.config.interface.task.evaluate_code(sol.sol_string)
                sol.evaluation_res = eval_res

            # 选择
            population = self.select(population + new_solutions)

            # 记录
            best = max(population, key=lambda s: s.evaluation_res.score if s.evaluation_res.valid else float('-inf'))
            print(f"第 {gen} 代: {best.evaluation_res.score}")
```

---

## 下一步

- 探索 [接口 API](interfaces.md) 了解算法-任务集成
- 查看 [核心 API](core.md) 了解基类
- 尝试 [高级用法教程](../tutorials/advanced-overview.md)
- 学习 [自定义任务](../tutorials/customization/custom-task.md) 创建

