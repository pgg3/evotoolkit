# 高级用法

掌握低级 API 以实现最大控制和自定义。

---

## 概述

这些高级教程涵盖：

- **低级 API** - 直接控制算法和配置
- **算法配置** - 微调进化参数
- **算法内部** - 访问和分析内部状态
- **调试和性能分析** - 故障排除和性能优化

---

## 前置条件

- 完成 [科学符号回归](built-in/scientific-regression.zh.md) 教程
- 完成 [自定义任务](customization/custom-task.zh.md) 教程
- 理解进化算法

---

## 教程

目前，所有高级教程内容都整合在一个完整的教程文档中：

**[→ 查看完整的高级用法教程](advanced-overview.zh.md)**

该教程涵盖以下主题：

### 1. 低级 API
了解高级和低级 API 之间的区别，以及何时使用它们。

- 高级 vs 低级 API 对比
- 直接实例化算法
- 访问内部状态
- 自定义工作流控制

---

### 2. 算法配置
掌握每个进化算法的详细配置选项。

- EvoEngineer 配置参数
- FunSearch 岛屿模型设置
- EoH 算子控制
- 并行执行调优

---

### 3. 算法内部
访问和分析进化算法的内部状态。

- 检查进化历史
- 访问解种群
- 绘制进化进程
- 提取指标和统计数据

---

### 4. 调试和性能分析
调试问题并优化进化工作流的性能。

- 启用详细日志
- 保存中间解
- 检查 LLM 提示/响应
- 时间和内存分析
- 实现自定义算法

---

## 何时使用高级功能

### 使用低级 API 的时机：
- 需要对进化过程进行细粒度控制
- 默认配置不满足需求
- 想要实现自定义停止准则
- 需要访问中间结果

### 使用自定义配置的时机：
- 默认参数对您的任务效果不佳
- 想要优化速度或质量
- 需要调整并行执行
- 正在尝试算法变体

### 使用调试工具的时机：
- 进化没有按预期收敛
- 想要理解算法行为
- 需要优化资源使用
- 正在开发自定义算法

---

## 快速参考

### 基础低级模式
```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=8,
    verbose=True
)

algorithm = EvoEngineer(config)
algorithm.run()

# 访问结果
best = algorithm._get_best_sol(algorithm.run_state_dict.sol_history)
```

---

## 下一步

掌握高级用法后：

- 探索 [API 参考](../api/index.md) 获取完整文档
- 阅读 [架构文档](../development/architecture.md) 了解内部原理
- 通过 [贡献指南](../development/contributing.md) 贡献您的改进