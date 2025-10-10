# 教程

欢迎来到 EvoToolkit 教程！这些分步指南将帮助您掌握使用 LLM 进行进化优化。

---

## 入门指南

初次使用 EvoToolkit？从这里开始：

1. **[安装](../installation.md)** - 设置您的环境
2. **[快速开始](../getting-started.md)** - 5 分钟内运行您的第一个优化任务
3. **[科学符号回归教程](built-in/scientific-regression.zh.md)** - 深入了解完整示例

---

## 教程分类

### 内置任务

学习如何使用 EvoToolkit 的预构建优化任务：

- **[科学符号回归](built-in/scientific-regression.zh.md)** - 从数据中发现数学方程
- **[提示词工程](built-in/prompt-engineering.zh.md)** - 优化 LLM 提示词以提升性能
- **[对抗攻击](built-in/adversarial-attack.zh.md)** - 生成对抗样本
- **[CUDA 任务](built-in/cuda-task.zh.md)** - 优化 GPU 内核性能

### 自定义

扩展 EvoToolkit 以满足您的特定需求：

- **[自定义任务](customization/custom-task.zh.md)** - 创建您自己的优化问题
- **[自定义进化方法](customization/customizing-evolution.zh.md)** - 修改 prompt 和算法

### 高级

掌握低级 API：

- **[高级用法](advanced-overview.zh.md)** - 精细控制和调试

---

## 教程概览

| 教程 | 难度 | 时间 | 涵盖主题 |
|------|------|------|----------|
| [科学符号回归](built-in/scientific-regression.zh.md) | 初级 | 20 分钟 | 高级 API、真实数据集、方程进化 |
| [提示词工程](built-in/prompt-engineering.zh.md) | 初级-中级 | 20 分钟 | LLM prompt 优化、任务性能提升 |
| [对抗攻击](built-in/adversarial-attack.zh.md) | 中级 | 25 分钟 | 进化对抗样本、攻击算法设计 |
| [CUDA 任务](built-in/cuda-task.zh.md) | 高级 | 30 分钟 | GPU 优化、CUDA 内核、性能 |
| [自定义任务](customization/custom-task.zh.md) | 中级 | 20 分钟 | 创建任务、评估、自定义适应度 |
| [自定义进化方法](customization/customizing-evolution.zh.md) | 中级-高级 | 30 分钟 | Prompt 工程、自定义算法、Interface 开发 |
| [高级用法](advanced-overview.zh.md) | 高级 | 25 分钟 | 低级 API、自定义配置、调试 |

---

## 快速参考

### 常见工作流模式

#### 模式 1: 基础优化
```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface

interface = EvoEngineerPythonInterface(task)
result = evotoolkit.solve(interface, './results', llm_api)
```

#### 模式 2: 算法比较
```python
algorithms = [
    ('EoH', EoHPythonInterface(task)),
    ('EvoEngineer', EvoEngineerPythonInterface(task)),
    ('FunSearch', FunSearchPythonInterface(task))
]

for name, interface in algorithms:
    result = evotoolkit.solve(interface, f'./results/{name}', llm_api)
```

#### 模式 3: 自定义配置
```python
from evotoolkit.evo_method.evoengineer import EvoEngineer, EvoEngineerConfig

config = EvoEngineerConfig(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=20,
    pop_size=10
)

algorithm = EvoEngineer(config)
algorithm.run()
```

---

## 可下载示例

所有教程代码都可以作为 `examples/` 目录中的独立 Python 脚本使用：

- `examples/scientific_regression/` - 科学方程发现
- `examples/custom_task/my_custom_task.py` - 自定义任务实现
- `examples/cuda_task/kernel_optimization.py` - CUDA 内核优化
- `examples/advanced/low_level_api.py` - 低级 API 用法

克隆仓库开始：

```bash
git clone https://github.com/pgg3/evotoolkitkit.git
cd evotool/examples
```

---

## 需要帮助？

### 文档和资源

- **[API 参考](../api/index.md)** - 详细的 API 文档
- **[开发指南](../development/contributing.md)** - 贡献代码指南
- **[高级示例](https://github.com/pgg3/evotoolkit/tree/master/examples)** - 复杂用例参考

### 社区支持

- **[GitHub 讨论](https://github.com/pgg3/evotoolkit/discussions)** - 提问和分享项目
- **[GitHub Issues](https://github.com/pgg3/evotoolkit/issues)** - 报告问题和建议功能
- **[示例库](https://github.com/pgg3/evotoolkit/wiki/Examples)** - 社区贡献的示例
- **[博客](https://github.com/pgg3/evotoolkit/wiki/Blog)** - 文章和案例研究

### 直接联系

- **邮件**: pguo6680@gmail.com

---

## 视频教程

即将推出！订阅我们的 [YouTube 频道](#) 获取视频教程。
