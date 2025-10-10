# 低级 API

了解何时以及如何使用低级 API 来最大化控制进化优化。

---

## 高级 API vs 低级 API

### 高级 API（推荐大多数用户）

```python
import evotoolkit

# 简单明了
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10
)
```

**优点：**
- 简单明了
- 自动配置
- 内置最佳实践
- 更少的代码维护

**缺点：**
- 对内部控制较少
- 自定义能力有限
- 固定的工作流结构

**最适合：**
- 大多数优化任务
- 快速原型开发
- 标准工作流
- 快速入门

---

### 低级 API（高级用户）

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

# 完全控制配置
config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8,
    max_sample_nums=12,
    num_samplers=4,  # 并行采样器数量
    num_evaluators=4,  # 并行评估器数量
    verbose=True
)

# 创建并运行算法
algorithm = EvoEngineer(config)
algorithm.run()

# 访问内部状态
best_solution = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)
all_solutions = algorithm.run_state_dict.sol_history
```

**优点：**
- 完全控制参数
- 访问内部状态
- 自定义工作流集成
- 高级调试能力

**缺点：**
- 代码更复杂
- 需要算法知识
- 更多维护负担
- 容易配置错误

**最适合：**
- 研究和实验
- 自定义工作流集成
- 性能优化
- 算法开发

---

## 使用不同的算法

### EvoEngineer

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8,
    max_sample_nums=12,
    num_samplers=4,
    num_evaluators=4,
    verbose=True
)

algorithm = EvoEngineer(config)
algorithm.run()
```

---

### FunSearch

```python
from evotoolkit.evo_method.funsearch import FunSearch, FunSearchConfig

config = FunSearchConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_sample_nums=30,
    programs_per_prompt=2,
    num_islands=4,
    max_population_size=1000,
    num_samplers=5,
    num_evaluators=5,
    verbose=True
)

algorithm = FunSearch(config)
algorithm.run()
```

**注意：** FunSearch 不使用 `max_generations`。它基于岛屿模型持续进化。

---

### EoH (启发式进化)

```python
from evotoolkit.evo_method.eoh import EoH, EoHConfig

config = EoHConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20,
    selection_num=2,
    use_e2_operator=True,  # 交叉
    use_m1_operator=True,  # 变异 1
    use_m2_operator=True,  # 变异 2
    num_samplers=5,
    num_evaluators=5,
    verbose=True
)

algorithm = EoH(config)
algorithm.run()
```

---

## 访问结果

### 获取最佳解

```python
algorithm.run()

# 方法 1：使用内置辅助函数
best_solution = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)

# 方法 2：手动搜索
all_solutions = algorithm.run_state_dict.sol_history
valid_solutions = [s for s in all_solutions if s.evaluation_res.valid]
best_solution = max(valid_solutions, key=lambda s: s.evaluation_res.score)

print(f"最佳分数: {best_solution.evaluation_res.score}")
print(f"最佳代码:\n{best_solution.sol_string}")
```

---

### 访问进化历史

```python
# 获取运行状态
run_state = algorithm.run_state_dict

# 所有生成过的解
all_solutions = run_state.sol_history

# 当前种群
current_population = run_state.population

# 分数进展
scores = [
    sol.evaluation_res.score
    for sol in all_solutions
    if sol.evaluation_res.valid
]

print(f"总解数: {len(all_solutions)}")
print(f"有效解数: {len(scores)}")
print(f"最佳分数: {max(scores)}")
print(f"平均分数: {sum(scores) / len(scores)}")
```

---

## 自定义工作流集成

### 检查点和恢复

```python
import pickle

# 运行几代
algorithm = EvoEngineer(config)
for gen in range(5):
    algorithm.run_one_generation()

    # 保存检查点
    with open(f'checkpoint_gen{gen}.pkl', 'wb') as f:
        pickle.dump(algorithm.run_state_dict, f)

# 稍后：从检查点恢复
with open('checkpoint_gen4.pkl', 'rb') as f:
    saved_state = pickle.load(f)

algorithm.run_state_dict = saved_state
algorithm.run()  # 从上次中断处继续
```

---

### 自定义停止准则

```python
class CustomEvoEngineer(EvoEngineer):
    def should_stop(self):
        # 如果找到分数 > 0.95 的解就停止
        best = self._get_best_sol(self.run_state_dict.sol_history)
        if best and best.evaluation_res.score > 0.95:
            print("找到优秀解！提前停止。")
            return True

        # 否则使用默认停止准则
        return super().should_stop()

algorithm = CustomEvoEngineer(config)
algorithm.run()
```

---

### 混合算法

```python
# 从 EvoEngineer 开始进行探索
config1 = EvoEngineerConfig(
    interface=interface,
    output_path='./results/phase1',
    running_llm=llm_api,
    max_generations=5,
    pop_size=10
)

algo1 = EvoEngineer(config1)
algo1.run()

# 获取第一阶段的最佳解
best_from_phase1 = sorted(
    algo1.run_state_dict.sol_history,
    key=lambda s: s.evaluation_res.score if s.evaluation_res.valid else float('-inf'),
    reverse=True
)[:3]

# 使用 FunSearch 进行精化
config2 = FunSearchConfig(
    interface=interface,
    output_path='./results/phase2',
    running_llm=llm_api,
    max_sample_nums=50
)

algo2 = FunSearch(config2)
# 用第一阶段的解初始化
algo2.run_state_dict.population = best_from_phase1
algo2.run()
```

---

## 下一步

- 学习 [算法配置](configuration.zh.md) 进行详细的参数调优
- 探索 [算法内部](internals.zh.md) 分析进化行为
- 查看 [调试与性能分析](debugging.zh.md) 获取故障排除技巧

---

## 资源

- [API 参考](../../api/methods.md) - 完整的 API 文档
- [架构指南](../../development/architecture.md) - 理解内部原理
