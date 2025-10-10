# EvoToolkit

**LLM驱动的解进化优化工具包**

EvoToolkit 是一个 Python 库，利用大语言模型（LLM）来进化优化问题的解决方案。它结合了进化算法的强大能力与基于 LLM 的解生成和改进，支持代码、文本及其他可评估的表示形式。

---

## ✨ 主要特性

- **🤖 LLM 驱动进化**: 使用最先进的语言模型生成和进化解决方案
- **🔬 多种算法**: 支持 EoH、EvoEngineer 和 FunSearch 进化方法
- **🌍 任务无关**: 支持任何可评估的优化任务（代码、文本、数学表达式等）
- **🎯 可扩展框架**: 易于扩展的任务系统，支持自定义优化问题
- **🔌 简单 API**: 高级 `evotoolkit.solve()` 函数，快速原型开发
- **🛠️ 高级定制**: 低级 API，提供精细化控制

### 内置任务类型

| 任务类型 | 描述 | 详情 |
|---------|------|------|
| **🔬 科学符号回归** | 在真实科学数据集上进行符号回归 | [科学回归教程](tutorials/built-in/scientific-regression.zh.md) |
| **💬 提示词工程** | 优化 LLM prompts 以提升下游任务性能 | [提示词工程教程](tutorials/built-in/prompt-engineering.zh.md) |
| **🛡️ 对抗攻击** | 进化对抗攻击算法 | [对抗攻击教程](tutorials/built-in/adversarial-attack.zh.md) |
| **⚡ CUDA 代码进化** | 进化和优化 CUDA kernels | [CUDA 任务教程](tutorials/built-in/cuda-task.zh.md) |

---

## 🚀 快速开始

### 安装

```bash
pip install evotoolkit

# 或安装全部依赖
pip install evotoolkit[all]
```

详细安装说明请参阅[安装指南](installation.md)。

### 第一个优化任务

```python
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

# 1. 创建任务
task = ScientificRegressionTask(dataset_name="bactgrow")

# 2. 创建接口
interface = EvoEngineerPythonInterface(task)

# 3. 使用 LLM 求解
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key-here",
    model="gpt-4o"
)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5
)
```

就是这么简单！EvoToolkit 将使用 LLM 进化数学方程来拟合您的科学数据。

完整的演示请查看[快速开始指南](getting-started.md)。

---

## 📚 可用算法

| 算法 | 描述 |
|------|------|
| **EvoEngineer** | 主要的 LLM 驱动进化算法 |
| **FunSearch** | 函数搜索优化方法 |
| **EoH** | 启发式进化 |

查看[教程](tutorials/index.md)了解更多使用示例。

---

## 📖 文档

- **[安装](installation.md)**: 安装说明和设置
- **[快速开始](getting-started.md)**: 快速入门指南和基本用法
- **[教程](tutorials/index.md)**: 常见任务的分步教程
- **[API 参考](api/index.md)**: 详细的 API 文档
- **[开发](development/contributing.md)**: 贡献指南和架构

---

## 🔗 链接

- **GitHub**: [https://github.com/pgg3/evotoolkit](https://github.com/pgg3/evotoolkit)
- **PyPI**: [https://pypi.org/project/evotoolkit/](https://pypi.org/project/evotoolkit/)
- **论文**: arXiv（已提交）

---

## 📄 许可证

EvoToolkit 采用双重许可：

- **学术与开源使用**: 免费用于学术研究、教育和开源项目。学术出版物中 **需要引用**。
- **商业使用**: 需要单独的商业许可证。请联系 pguo6680@gmail.com 获取许可。

详细条款请参阅 [LICENSE](https://github.com/pgg3/evotoolkit/blob/master/LICENSE)。

---

## 🙏 引用

如果您在研究中使用 EvoToolkit，请引用：

```bibtex
@article{guo2025evotoolkit,
  title={evotoolkit: A Unified LLM-Driven Evolutionary Framework for Generalized Solution Search},
  author={Guo, Ping and Zhang, Qingfu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  note={Submitted to arXiv}
}
```

---

## 💬 获取帮助

- **问题**: [GitHub Issues](https://github.com/pgg3/evotoolkit/issues)
- **讨论**: [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions)
- **邮箱**: pguo6680@gmail.com
