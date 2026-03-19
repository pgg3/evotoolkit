# 控制任务 (Box2D / Lunar Lander) 教程

学习如何使用 LLM 驱动的进化算法为 Gymnasium LunarLander-v3 环境发现可解释的控制策略。

!!! tip "完整示例代码"
    本教程提供完整的可运行示例（点击查看/下载）：

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/lunar_lander/basic_example.py) - 基本用法示例
    - [:material-file-document: README.md](https://github.com/pgg3/evotoolkit/blob/master/examples/lunar_lander/README.md) - 示例文档和使用指南

    本地运行：
    ```bash
    cd examples/lunar_lander
    python basic_example.py
    ```

---

## 概述

本教程演示：

- 创建 LunarLander 控制任务
- 使用 LLM 驱动的进化算法进化可解释的 Python 控制策略
- 理解 `policy(state) -> action` 函数接口
- 在 Gymnasium 环境中评估策略
- 运行进化算法发现有效的着陆策略

与神经网络控制器不同，EvoToolkit 进化的是 **人类可读的 Python 代码**，使得策略可以被检查、理解和形式化验证。

---

## 安装

```bash
pip install evotoolkit[control_box2d]
```

这将安装：

- `gymnasium[box2d]` - 带 Box2D 物理引擎的 Gymnasium 环境
- `box2d-py` - Box2D 物理引擎

**前置要求：**

- Python >= 3.10
- LLM API 访问权限（OpenAI、Claude 或其他兼容提供商）

---

## 理解 LunarLander 任务

### 任务进化什么？

该任务进化一个 `policy` 函数，将 8 维状态观测映射到 4 个离散动作之一：

| 状态索引 | 含义 |
|---------|------|
| 0 | X 位置 |
| 1 | Y 位置 |
| 2 | X 速度 |
| 3 | Y 速度 |
| 4 | 倾斜角度 |
| 5 | 角速度 |
| 6 | 左腿接触（布尔值）|
| 7 | 右腿接触（布尔值）|

| 动作 | 含义 |
|------|------|
| 0 | 什么都不做 |
| 1 | 点燃左引擎 |
| 2 | 点燃主引擎（向上推力）|
| 3 | 点燃右引擎 |

### 评估

每个策略在多个回合中评估，得分为每回合的平均奖励：

- **完美着陆**：约 200 分
- **坠毁**：-100 分
- **燃料效率**：每帧引擎使用 -0.3 分

---

## 快速开始

### 步骤 1：创建任务

```python
from evotoolkit.task.python_task.control_box2d import LunarLanderTask

task = LunarLanderTask(
    num_episodes=5,       # 每次评估的回合数
    max_steps=1000,       # 每回合最大步数
    render_mode=None,     # 设置为 "human" 可视化观看
    seed=42,              # 可重现性的随机种子
    timeout_seconds=60.0,
)

print(f"环境: {task.task_info['env_name']}")
print(f"状态维度: {task.task_info['state_dim']}")
print(f"动作维度: {task.task_info['action_dim']}")
```

### 步骤 2：测试基线策略

```python
# 获取内置基线策略
init_sol = task.make_init_sol_wo_other_info()
result = task.evaluate_code(init_sol.sol_string)

print(f"基线得分: {result.score:.2f}")
if result.valid:
    print(f"平均奖励: {result.additional_info['avg_reward']:.2f}")
    print(f"成功率: {result.additional_info['success_rate']:.1%}")
```

### 步骤 3：运行进化

```python
import evotoolkit
from evotoolkit.task.python_task.control_box2d import EvoEngineerControlInterface
from evotoolkit.tools.llm import HttpsApi

# 创建控制专用接口
interface = EvoEngineerControlInterface(task)

# 配置 LLM
llm_api = HttpsApi(
    api_url="api.openai.com",
    key="your-api-key-here",
    model="gpt-4o"
)

# 运行进化
result = evotoolkit.solve(
    interface=interface,
    output_path='./lunar_lander_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

print(f"发现的最优策略：")
print(result.sol_string)
print(f"得分: {result.evaluation_res.score:.2f}")
```

---

## 策略函数接口

进化的函数必须具有以下确切的签名：

```python
def policy(state: list) -> int:
    """
    LunarLander-v3 的控制策略。

    参数：
        state: 8 维观测值：
            [x_pos, y_pos, x_vel, y_vel, angle, angular_vel,
             left_contact, right_contact]

    返回：
        action: {0, 1, 2, 3} 中的整数
            0 = 什么都不做
            1 = 点燃左引擎
            2 = 点燃主引擎（向上推力）
            3 = 点燃右引擎
    """
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
    # 你的控制逻辑
    return 0
```

### 示例：简单启发式策略

```python
def policy(state):
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state

    # 下落太快时点燃主引擎
    if vy < -1.0:
        return 2

    # 修正角度
    if angle > 0.2:
        return 3  # 点燃右引擎向左倾
    elif angle < -0.2:
        return 1  # 点燃左引擎向右倾

    # 缓慢悬停下降
    if vy < -0.5:
        return 2

    return 0
```

---

## 可用接口

| 接口 | 算法 | 描述 |
|------|------|------|
| `EvoEngineerControlInterface` | EvoEngineer | 推荐 — 使用控制专用提示 |
| `EvoEngineerPythonInterface` | EvoEngineer | 通用 Python 接口 |
| `EoHPythonInterface` | EoH | 启发式进化 |
| `FunSearchPythonInterface` | FunSearch | 函数搜索 |

```python
# 导入选项
from evotoolkit.task.python_task.control_box2d import EvoEngineerControlInterface
from evotoolkit.task.python_task import EvoEngineerPythonInterface, EoHPythonInterface
```

---

## `LunarLanderTask` API

```python
class LunarLanderTask(PythonTask):
    def __init__(
        self,
        num_episodes: int = 10,           # 每次评估的回合数
        max_steps: int = 1000,            # 每回合最大步数
        render_mode: str | None = None,   # "human" 可视化
        use_mock: bool = False,           # 返回随机得分（用于测试）
        seed: int | None = None,          # 随机种子
        timeout_seconds: float = 60.0,   # 执行超时
    )
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `evaluate_code(code_str)` | 评估策略字符串，返回 `EvaluationResult` |
| `make_init_sol_wo_other_info()` | 获取内置基线策略作为 Solution |

**`EvaluationResult.additional_info` 的键：**

| 键 | 描述 |
|----|------|
| `avg_reward` | 回合平均奖励 |
| `std_reward` | 奖励标准差 |
| `success_rate` | 奖励 > 200 的回合比例 |
| `min_reward` | 最低回合奖励 |
| `max_reward` | 最高回合奖励 |

---

## 获得更好结果的技巧

1. **使用更多回合** 以稳定评估：`num_episodes=10` 或更多
2. **设置随机种子** 以提高开发阶段的可重现性
3. **使用 `EvoEngineerControlInterface`** — 它在提示中包含控制专用的领域知识
4. **增加 `max_generations`** 应对更难的任务

---

## 下一步

- [自定义进化方法](../customization/customizing-evolution.zh.md) — 修改提示以获得更好的控制策略
- [高级用法](../advanced-overview.zh.md) — 低级 API 和配置
- [API 参考](../../api/index.md) — 完整 API 文档
