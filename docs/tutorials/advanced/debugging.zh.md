# 调试与性能分析

调试问题并优化进化工作流的性能。

---

## 启用详细日志

### 基本详细模式

```python
from evotoolkit.evo_method.evoengineer import EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    verbose=True  # 启用详细日志
)

algorithm = EvoEngineer(config)
algorithm.run()
```

**输出示例：**
```
第 1/10 代:
  - 生成了 12 个解
  - 有效解: 8
  - 最佳分数: 0.245
  - 平均分数: 0.512
  - 保留精英: 2

第 2/10 代:
  - 生成了 12 个解
  - 有效解: 10
  - 最佳分数: 0.189
  - 平均分数: 0.431
  - 保留精英: 2
...
```

---

## 保存中间解

### 保存所有代

```python
config = EvoEngineerConfig(
    # ... 其他参数
    save_all_generations=True  # 保存每代的解
)
```

**目录结构：**
```
results/
├── generation_1/
│   ├── solution_1.py
│   ├── solution_2.py
│   └── ...
├── generation_2/
│   ├── solution_1.py
│   └── ...
├── ...
└── best_solution.py
```

这允许您：
- 检查失败的解
- 调试评估错误
- 分析解的进化
- 从崩溃中恢复

---

## 检查 LLM 交互

### 启用 LLM 日志

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./results/llm_debug.log'),
        logging.StreamHandler()
    ]
)

# 启用 LLM 日志记录器
logger = logging.getLogger('evotoolkit.llm')
logger.setLevel(logging.DEBUG)

# 现在运行算法 - 所有 LLM 交互都将被记录
algorithm.run()
```

**日志输出示例：**
```
2024-01-15 10:23:45 - evotoolkit.llm - DEBUG - 发送提示到 LLM:
  --- 提示开始 ---
  You are an expert Python programmer...
  --- 提示结束 ---

2024-01-15 10:23:52 - evotoolkit.llm - DEBUG - 收到 LLM 响应:
  --- 响应开始 ---
  def target_function(x):
      return x ** 2 + 2 * x + 1
  --- 响应结束 ---

2024-01-15 10:23:52 - evotoolkit.llm - DEBUG - Token 使用: 245 输入, 67 输出
```

---

### 保存提示和响应

```python
class DebugInterface(EvoEngineerPythonInterface):
    def __init__(self, task):
        super().__init__(task)
        self.prompt_history = []

    def query_llm(self, prompt):
        response = super().query_llm(prompt)

        # 保存以供调试
        self.prompt_history.append({
            'prompt': prompt,
            'response': response,
            'timestamp': time.time()
        })

        # 保存到文件
        with open('./results/llm_history.json', 'w') as f:
            json.dump(self.prompt_history, f, indent=2)

        return response
```

---

## 性能分析

### 时间分析

```python
import time

start_time = time.time()

algorithm = EvoEngineer(config)
algorithm.run()

elapsed = time.time() - start_time
print(f"\n总优化时间: {elapsed:.2f} 秒")
print(f"每代时间: {elapsed / config.max_generations:.2f} 秒")

# 详细计时（如果可用）
if hasattr(algorithm.run_state_dict, 'metadata'):
    gen_times = algorithm.run_state_dict.metadata.get('generation_times', [])
    for i, t in enumerate(gen_times):
        print(f"第 {i+1} 代: {t:.2f}秒")
```

---

### 使用 cProfile 进行详细分析

```python
import cProfile
import pstats
from pstats import SortKey

# 分析运行
profiler = cProfile.Profile()
profiler.enable()

algorithm = EvoEngineer(config)
algorithm.run()

profiler.disable()

# 保存并分析结果
stats = pstats.Stats(profiler)
stats.sort_stats(SortKey.CUMULATIVE)

# 打印前 20 个最耗时的函数
print("\n前 20 个耗时函数:")
stats.print_stats(20)

# 保存到文件
with open('./results/profile_stats.txt', 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats()
```

---

### 内存分析

```python
import tracemalloc

# 开始内存跟踪
tracemalloc.start()

algorithm = EvoEngineer(config)
algorithm.run()

# 获取内存使用
current, peak = tracemalloc.get_traced_memory()
print(f"\n当前内存使用: {current / 1024 / 1024:.2f} MB")
print(f"峰值内存使用: {peak / 1024 / 1024:.2f} MB")

# 获取最大内存分配
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("\n前 10 个内存分配:")
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
```

---

## 常见调试场景

### 问题：解没有改进

**诊断：**
```python
# 检查分数进展
scores = [s.evaluation_res.score for s in all_solutions if s.evaluation_res.valid]
print(f"前 5 个分数: {scores[:5]}")
print(f"后 5 个分数: {scores[-5:]}")

# 检查是否陷入局部最优
if len(set(scores[-10:])) == 1:
    print("警告: 分数没有变化 - 可能陷入困境！")
```

**解决方案：**
- 增加种群多样性（更大的 `pop_size`）
- 增加采样（更大的 `max_sample_nums`）
- 调整 LLM 温度（更高以获得更多探索）
- 检查任务评估是否正常工作

---

### 问题：许多无效解

**诊断：**
```python
valid_count = sum(1 for s in all_solutions if s.evaluation_res.valid)
total_count = len(all_solutions)
success_rate = valid_count / total_count * 100

print(f"成功率: {success_rate:.1f}%")

# 检查错误消息
errors = [s.evaluation_res.error_message for s in all_solutions if not s.evaluation_res.valid]
from collections import Counter
print("\n最常见的错误:")
for error, count in Counter(errors).most_common(5):
    print(f"  {count}次: {error[:100]}...")
```

**解决方案：**
- 改进提示的清晰度
- 在提示中添加更多示例
- 放宽任务约束
- 检查任务评估逻辑

---

### 问题：性能慢

**诊断：**
```python
import time

# 计时每个组件
start = time.time()
algorithm = EvoEngineer(config)
init_time = time.time() - start

start = time.time()
algorithm.run()
run_time = time.time() - start

print(f"初始化: {init_time:.2f}秒")
print(f"执行: {run_time:.2f}秒")
print(f"每代: {run_time / config.max_generations:.2f}秒")
```

**解决方案：**
- 增加并行度（`num_samplers`、`num_evaluators`）
- 减少 `max_sample_nums`
- 使用更快的 LLM 模型
- 优化任务评估代码

---

## 自定义算法实现

对于想要实现自己进化算法的高级用户：

```python
from evotoolkit.core import BaseMethod, BaseConfig, Solution

class MyCustomAlgorithm(BaseMethod):
    """自定义进化算法实现"""

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.population = []

    def run(self):
        """主进化循环"""
        # 初始化种群
        self.population = self.initialize_population()

        for generation in range(self.config.max_generations):
            print(f"\n第 {generation + 1}/{self.config.max_generations} 代")

            # 使用 LLM 生成新解
            new_solutions = self.generate_solutions()

            # 评估解
            for solution in new_solutions:
                eval_res = self.config.interface.task.evaluate_code(solution.sol_string)
                solution.evaluation_res = eval_res

            # 更新种群
            self.population = self.select(self.population + new_solutions)

            # 记录进度
            valid_pop = [s for s in self.population if s.evaluation_res.valid]
            if valid_pop:
                best = max(valid_pop, key=lambda s: s.evaluation_res.score)
                print(f"  最佳分数: {best.evaluation_res.score:.4f}")

        # 保存结果
        self.save_results()

    def initialize_population(self):
        """生成初始种群"""
        initial_solutions = []

        # 使用 LLM 生成初始解
        for i in range(self.config.pop_size):
            prompt = self.config.interface.get_init_prompt()
            response = self.config.running_llm.query(prompt)

            solution = Solution(sol_string=response)
            eval_res = self.config.interface.task.evaluate_code(response)
            solution.evaluation_res = eval_res

            initial_solutions.append(solution)

        return initial_solutions

    def generate_solutions(self):
        """为当前代生成新解"""
        new_solutions = []

        # 示例：变异
        for parent in self.population[:3]:  # 取前 3 名
            prompt = self.config.interface.get_mutation_prompt(parent)
            response = self.config.running_llm.query(prompt)

            solution = Solution(sol_string=response)
            new_solutions.append(solution)

        return new_solutions

    def select(self, solutions):
        """为下一代选择最佳解"""
        # 过滤有效解
        valid = [s for s in solutions if s.evaluation_res.valid]

        # 按分数排序（越高越好）
        valid.sort(key=lambda s: s.evaluation_res.score, reverse=True)

        # 保留前 pop_size 个解
        return valid[:self.config.pop_size]

    def save_results(self):
        """保存最终结果"""
        best = max(self.population, key=lambda s: s.evaluation_res.score)

        with open(f'{self.config.output_path}/best_solution.py', 'w') as f:
            f.write(best.sol_string)

        print(f"\n优化完成！")
        print(f"最佳分数: {best.evaluation_res.score:.4f}")
```

**使用：**
```python
algorithm = MyCustomAlgorithm(config)
algorithm.run()
```

---

## 调试检查清单

当出现问题时，检查：

- [ ] 任务评估函数正常工作
- [ ] LLM API 正在响应（检查日志）
- [ ] 提示清晰且包含示例
- [ ] 正在生成解（检查 `sol_history`）
- [ ] 分数计算正确
- [ ] 配置参数合理
- [ ] 资源充足（API 速率限制、内存）
- [ ] 输出目录可写

---

## 下一步

- 查看 [算法内部](internals.zh.md) 了解分析技术
- 查看 [配置](configuration.zh.md) 进行参数调优
- 探索 [低级 API](low-level-api.zh.md) 获得更多控制

---

## 资源

- [Python 日志文档](https://docs.python.org/3/library/logging.html)
- [cProfile 文档](https://docs.python.org/3/library/profile.html)
- [tracemalloc 文档](https://docs.python.org/3/library/tracemalloc.html)
- [性能分析指南](https://docs.python.org/3/library/debug.html)
