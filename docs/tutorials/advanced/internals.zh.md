# 算法内部

访问和分析进化算法的内部状态。

---

## 概述

EvoToolkit 的低级 API 提供对算法内部的完全访问，允许您：
- 检查进化历史
- 访问解种群
- 提取指标和统计数据
- 绘制进化进程

---

## 访问运行状态

所有算法将其内部状态存储在 `run_state_dict` 中：

```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

algorithm = EvoEngineer(config)
algorithm.run()

# 访问运行状态
run_state = algorithm.run_state_dict

# 获取所有解历史
all_solutions = run_state.sol_history

# 获取当前种群
current_population = run_state.population
```

---

## 检查进化历史

### 获取所有解

```python
# 所有生成过的解（包括无效的）
all_solutions = algorithm.run_state_dict.sol_history

print(f"总共生成的解: {len(all_solutions)}")
```

---

### 过滤有效解

```python
# 仅有效解
valid_solutions = [
    sol for sol in all_solutions
    if sol.evaluation_res.valid
]

print(f"有效解: {len(valid_solutions)}")
print(f"成功率: {len(valid_solutions) / len(all_solutions) * 100:.1f}%")
```

---

### 获取分数历史

```python
# 提取分数（越高越好）
score_history = [
    sol.evaluation_res.score
    for sol in all_solutions
    if sol.evaluation_res.valid
]

print(f"最佳分数: {max(score_history)}")
print(f"平均分数: {sum(score_history) / len(score_history):.4f}")
print(f"分数提升: {max(score_history) - score_history[0]:.4f}")
```

---

### 获取最佳解

```python
# 方法 1：使用内置辅助函数
best_solution = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)

# 方法 2：手动搜索
best_solution = max(
    all_solutions,
    key=lambda s: s.evaluation_res.score if s.evaluation_res.valid else float('-inf')
)

print(f"最佳分数: {best_solution.evaluation_res.score}")
print(f"最佳代码:\n{best_solution.sol_string}")
```

---

## 解对象结构

每个解包含详细信息：

```python
solution = all_solutions[0]

# 核心属性
solution.sol_string          # 实际的代码/解字符串
solution.evaluation_res      # 评估结果对象
solution.other_info         # 附加元数据字典

# 评估结果
eval_res = solution.evaluation_res
eval_res.valid              # 布尔值：解是否有效？
eval_res.score              # 浮点数：适应度分数（越高越好）
eval_res.error_message      # 字符串：如果无效则为错误信息
eval_res.metadata           # 字典：附加评估信息

# 示例：打印解的详细信息
for i, sol in enumerate(all_solutions[:5]):
    print(f"\n解 {i+1}:")
    print(f"  有效: {sol.evaluation_res.valid}")
    print(f"  分数: {sol.evaluation_res.score:.4f}")
    if not sol.evaluation_res.valid:
        print(f"  错误: {sol.evaluation_res.error_message}")
```

---

## 绘制进化进程

### 分数随时间变化

```python
import matplotlib.pyplot as plt

# 按顺序获取有效分数
scores = [
    sol.evaluation_res.score
    for sol in all_solutions
    if sol.evaluation_res.valid
]

plt.figure(figsize=(10, 6))
plt.plot(scores, marker='o', alpha=0.6, linewidth=1, markersize=4)
plt.xlabel('解索引', fontsize=12)
plt.ylabel('分数（越高越好）', fontsize=12)
plt.title('进化进程', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results/evolution_progress.png', dpi=300)
plt.show()
```

---

### 按代数的最佳分数

```python
import matplotlib.pyplot as plt
import numpy as np

# 按代数分组解
generations = {}
for sol in all_solutions:
    if sol.evaluation_res.valid:
        gen = sol.other_info.get('generation', 0)
        if gen not in generations:
            generations[gen] = []
        generations[gen].append(sol.evaluation_res.score)

# 获取每代的最佳分数
gen_numbers = sorted(generations.keys())
best_scores = [max(generations[gen]) for gen in gen_numbers]
avg_scores = [np.mean(generations[gen]) for gen in gen_numbers]

plt.figure(figsize=(10, 6))
plt.plot(gen_numbers, best_scores, 'g-o', label='最佳分数', linewidth=2)
plt.plot(gen_numbers, avg_scores, 'b--s', label='平均分数', linewidth=2)
plt.xlabel('代数', fontsize=12)
plt.ylabel('分数', fontsize=12)
plt.title('按代数的分数', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results/score_by_generation.png', dpi=300)
plt.show()
```

---

### 成功率分析

```python
import matplotlib.pyplot as plt

# 按代数计算成功率
success_rates = []
for gen in gen_numbers:
    total = len([s for s in all_solutions if s.other_info.get('generation') == gen])
    valid = len(generations.get(gen, []))
    success_rates.append(valid / total * 100 if total > 0 else 0)

plt.figure(figsize=(10, 6))
plt.bar(gen_numbers, success_rates, alpha=0.7, color='steelblue')
plt.xlabel('代数', fontsize=12)
plt.ylabel('成功率 (%)', fontsize=12)
plt.title('按代数的解有效性', fontsize=14)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./results/success_rate.png', dpi=300)
plt.show()
```

---

## 分析解的多样性

### 代码长度分布

```python
import matplotlib.pyplot as plt

# 获取代码长度
code_lengths = [
    len(sol.sol_string)
    for sol in all_solutions
    if sol.evaluation_res.valid
]

plt.figure(figsize=(10, 6))
plt.hist(code_lengths, bins=20, alpha=0.7, color='coral', edgecolor='black')
plt.xlabel('代码长度（字符）', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.title('解代码长度分布', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./results/code_length_dist.png', dpi=300)
plt.show()
```

---

### 分数分布

```python
import matplotlib.pyplot as plt

scores = [
    sol.evaluation_res.score
    for sol in all_solutions
    if sol.evaluation_res.valid
]

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('分数', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.title('分数分布', fontsize=14)
plt.axvline(max(scores), color='r', linestyle='--', linewidth=2, label=f'最佳: {max(scores):.4f}')
plt.axvline(np.mean(scores), color='b', linestyle='--', linewidth=2, label=f'平均: {np.mean(scores):.4f}')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('./results/score_distribution.png', dpi=300)
plt.show()
```

---

## 提取指标

### 综合统计

```python
import numpy as np

def compute_statistics(all_solutions):
    """计算综合进化统计数据"""

    valid_solutions = [s for s in all_solutions if s.evaluation_res.valid]
    scores = [s.evaluation_res.score for s in valid_solutions]

    stats = {
        'total_solutions': len(all_solutions),
        'valid_solutions': len(valid_solutions),
        'success_rate': len(valid_solutions) / len(all_solutions) * 100,
        'best_score': max(scores) if scores else None,
        'worst_score': min(scores) if scores else None,
        'mean_score': np.mean(scores) if scores else None,
        'median_score': np.median(scores) if scores else None,
        'std_score': np.std(scores) if scores else None,
        'score_range': max(scores) - min(scores) if scores else None,
    }

    return stats

stats = compute_statistics(all_solutions)

print("进化统计:")
print(f"  总解数: {stats['total_solutions']}")
print(f"  有效解数: {stats['valid_solutions']}")
print(f"  成功率: {stats['success_rate']:.1f}%")
print(f"\n分数统计:")
print(f"  最佳: {stats['best_score']:.4f}")
print(f"  最差: {stats['worst_score']:.4f}")
print(f"  平均: {stats['mean_score']:.4f}")
print(f"  中位数: {stats['median_score']:.4f}")
print(f"  标准差: {stats['std_score']:.4f}")
print(f"  范围: {stats['score_range']:.4f}")
```

---

### 导出到 DataFrame

```python
import pandas as pd

# 将解转换为 DataFrame 进行分析
data = []
for i, sol in enumerate(all_solutions):
    data.append({
        'index': i,
        'valid': sol.evaluation_res.valid,
        'score': sol.evaluation_res.score if sol.evaluation_res.valid else None,
        'generation': sol.other_info.get('generation', -1),
        'code_length': len(sol.sol_string),
        'error': sol.evaluation_res.error_message if not sol.evaluation_res.valid else None
    })

df = pd.DataFrame(data)

# 保存到 CSV
df.to_csv('./results/evolution_data.csv', index=False)

# 快速分析
print(df.describe())
print("\n按代数的分数:")
print(df.groupby('generation')['score'].agg(['mean', 'max', 'count']))
```

---

## 下一步

- 学习 [调试与性能分析](debugging.zh.md) 来排查问题
- 查看 [低级 API](low-level-api.zh.md) 获取更多控制选项
- 查看 [配置](configuration.zh.md) 进行参数调优

---

## 资源

- [Matplotlib 文档](https://matplotlib.org/) - 绘图库
- [Pandas 文档](https://pandas.pydata.org/) - 数据分析
- [NumPy 文档](https://numpy.org/) - 数值计算
