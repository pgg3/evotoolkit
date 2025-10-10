# 算法配置

掌握每个进化算法的详细配置选项。

---

## 概述

EvoToolkit 中的每个进化算法都有自己的配置类和特定参数。本教程涵盖所有配置选项以及如何根据您的用例进行调优。

---

## EvoEngineer 配置

EvoEngineer 是主要的 LLM 驱动进化算法。

```python
from evotoolkit.evo_method.evoengineer import EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,

    # 进化参数
    max_generations=20,      # 最大代数
    pop_size=10,             # 种群大小
    max_sample_nums=15,      # 每代最大采样数

    # 并行控制
    num_samplers=4,          # 并行采样器数量
    num_evaluators=4,        # 并行评估器数量

    # 日志
    verbose=True             # 显示详细日志
)
```

### 关键参数

**进化参数：**
- `max_generations` - 要运行的进化代数
- `pop_size` - 种群中维护的解数量
- `max_sample_nums` - 每代采样的最大新解数量

**并行执行：**
- `num_samplers` - 并行 LLM 采样工作器
- `num_evaluators` - 并行评估工作器

**日志：**
- `verbose` - 启用详细进度日志

### 重要说明

**LLM 温度和其他采样参数在创建 `HttpsApi` 时设置，而不是在算法配置中。**

```python
from evotoolkit.tools.llm import HttpsApi

# LLM 配置在这里
llm_api = HttpsApi(
    api_key="your-key",
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,  # LLM 温度
    max_tokens=4096
)

# 算法配置不包含温度
config = EvoEngineerConfig(
    running_llm=llm_api,  # 传递配置好的 LLM
    # ... 其他参数
)
```

---

## FunSearch 配置

FunSearch 使用岛屿模型进行持续进化。

```python
from evotoolkit.evo_method.funsearch import FunSearchConfig

config = FunSearchConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,

    # 采样参数
    max_sample_nums=30,           # 最大采样数
    programs_per_prompt=2,        # 每个提示生成的程序数

    # 岛屿模型
    num_islands=4,                # 并行进化岛屿数量
    max_population_size=1000,     # 每个岛屿的最大种群大小

    # 并行控制
    num_samplers=5,               # 并行采样器数量
    num_evaluators=5,             # 并行评估器数量

    # 日志
    verbose=True
)
```

### 关键参数

**采样：**
- `max_sample_nums` - 要生成的总采样数
- `programs_per_prompt` - 每次 LLM 调用的解数量

**岛屿模型：**
- `num_islands` - 独立的进化岛屿（增加多样性）
- `max_population_size` - 每个岛屿的最大解数量

**注意：** FunSearch **不**使用 `max_generations`。它基于岛屿模型持续进化，直到达到 `max_sample_nums`。

### 何时使用 FunSearch

- 当您想要持续进化而不是固定代数时
- 用于探索多样化的解空间
- 当您有计算资源支持大规模种群时

---

## EoH 配置

EoH（启发式进化）提供对遗传算子的显式控制。

```python
from evotoolkit.evo_method.eoh import EoHConfig

config = EoHConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,

    # 进化参数
    max_generations=10,       # 最大代数
    pop_size=5,               # 种群大小
    max_sample_nums=20,       # 每代最大采样数
    selection_num=2,          # 交叉的父代数量

    # 算子控制
    use_e2_operator=True,     # 使用 E2 算子（交叉）
    use_m1_operator=True,     # 使用 M1 算子（变异）
    use_m2_operator=True,     # 使用 M2 算子（第二种变异）

    # 并行控制
    num_samplers=5,           # 并行采样器数量
    num_evaluators=5,         # 并行评估器数量

    # 日志
    verbose=True
)
```

### 关键参数

**进化：**
- `max_generations` - 代数
- `pop_size` - 种群大小（通常小于 EvoEngineer）
- `max_sample_nums` - 每代采样数
- `selection_num` - 为交叉选择的父代数

**遗传算子：**
- `use_e2_operator` - 启用/禁用交叉算子
- `use_m1_operator` - 启用/禁用第一种变异算子
- `use_m2_operator` - 启用/禁用第二种变异算子

### 何时使用 EoH

- 当您想要显式控制遗传算子时
- 用于研究比较不同的算子组合
- 当传统进化算法概念很重要时

---

## 调优指南

### 种群大小

**小（5-10）：**
- 优点：代数更快，成本更低
- 缺点：多样性较少，可能过快收敛
- 最适合：简单问题，资源有限

**中等（10-20）：**
- 优点：速度和多样性的良好平衡
- 缺点：没有主要缺点
- 最适合：大多数问题（推荐默认值）

**大（20+）：**
- 优点：最大多样性，彻底探索
- 缺点：较慢，成本更高
- 最适合：复杂问题，研究

---

### 并行执行

```python
config = EvoEngineerConfig(
    # ... 其他参数
    num_samplers=4,      # 并行 LLM 调用
    num_evaluators=4,    # 并行评估
)
```

**指南：**
- `num_samplers`：根据 LLM API 速率限制设置
- `num_evaluators`：根据 CPU/GPU 可用性设置
- 保守地开始（2-4），如果资源允许则增加

**示例配置：**

```python
# 保守（低资源）
num_samplers=2
num_evaluators=2

# 平衡（中等资源）
num_samplers=4
num_evaluators=4

# 激进（高资源）
num_samplers=8
num_evaluators=8
```

---

### 代数

**少代数（5-10）：**
- 快速实验
- 简单问题
- 快速原型

**中等代数（10-20）：**
- 大多数问题
- 平衡探索
- 推荐默认值

**多代数（20+）：**
- 复杂问题
- 研究
- 最终优化运行

---

## 配置预设

### 快速实验

```python
config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    pop_size=5,
    max_sample_nums=10,
    num_samplers=2,
    num_evaluators=2,
    verbose=True
)
```

**用于：** 测试、调试、快速迭代

---

### 平衡性能

```python
config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=15,
    pop_size=10,
    max_sample_nums=20,
    num_samplers=4,
    num_evaluators=4,
    verbose=True
)
```

**用于：** 大多数生产用例

---

### 彻底搜索

```python
config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=30,
    pop_size=15,
    max_sample_nums=30,
    num_samplers=6,
    num_evaluators=6,
    verbose=True
)
```

**用于：** 研究、基准测试、最终运行

---

## 下一步

- 学习 [算法内部](internals.zh.md) 分析进化行为
- 查看 [调试与性能分析](debugging.zh.md) 进行性能优化
- 查阅 [API 参考](../../api/methods.md) 获取完整的参数详情

---

## 资源

- [EvoEngineer 论文](https://arxiv.org/abs/...) - 算法详情
- [FunSearch 论文](https://www.nature.com/articles/...) - 岛屿模型理论
- [EoH 论文](https://arxiv.org/abs/...) - 启发式进化
