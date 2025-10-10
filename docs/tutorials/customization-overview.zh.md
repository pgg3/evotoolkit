# 自定义

学习如何扩展 EvoToolkit 以解决您的特定优化问题。

## 自定义选项

### 自定义任务
**[→ 开始教程](customization/custom-task.zh.md)**

为特定领域问题创建您自己的优化任务。

**您将学到:**
- 扩展 `Task` 基类
- 实现自定义评估逻辑
- 定义解空间
- 与进化算法集成

**前置条件:** 基础 EvoToolkit 知识（科学符号回归教程）

---

### 自定义进化方法
**[→ 开始教程](customization/customizing-evolution.zh.md)**

学习如何通过修改 prompt 或开发新算法来自定义进化行为。

**您将学到:**
- 理解 Interface 架构
- 自定义 LLM prompt 以改进结果
- 为特定任务设计专门的 prompt
- 开发全新的进化算法（高级）
- 实现温度退火等自定义策略

**前置条件:** 基础 EvoToolkit 知识（科学符号回归教程）

---

## 何时自定义

### 创建自定义任务的时机：
- 您的问题领域未被内置任务覆盖
- 您需要特定的评估指标
- 您想优化特定领域的代码或结构
- 您需要自定义约束或验证

### 自定义进化方法的时机：
- 默认 prompt 对您的任务效果不佳
- 您想将领域知识融入进化过程
- 您需要特殊的变异或交叉策略
- 您想尝试新颖的进化方法

---

## 快速开始示例

### 自定义任务示例
```python
from evotoolkit.core import BaseTask, Solution

class MyCustomTask(BaseTask):
    def evaluate(self, solution: Solution) -> float:
        # 您的评估逻辑
        return fitness_score
```

### 自定义接口示例
```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface

class MyCustomInterface(EvoEngineerPythonInterface):
    def get_prompt_components(self):
        # 为您的任务自定义 prompt
        return custom_prompts
```

---

## 最佳实践

1. **从简单开始：** 在创建全新任务之前，先尝试修改现有任务或 prompt
2. **充分测试：** 使用已知解验证您的自定义评估逻辑
3. **良好文档：** 清楚地记录您的任务需求和约束
4. **分享成果：** 考虑将您的自定义任务贡献给社区

---

## 下一步

掌握自定义后：
- 探索[高级用法](advanced-overview.zh.md)了解低级 API 控制
- 查看 [API 参考](../api/index.md)获取详细文档
- 在 [GitHub 讨论](https://github.com/pgg3/evotoolkit/discussions)中分享您的自定义任务