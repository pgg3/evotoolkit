# EvoToolkit

**面向可执行解的 LLM 驱动进化优化框架**

EvoToolkit 是一个 Python 框架，用大语言模型来进化代码、符号表达式、提示词以及其他可执行文本。它把系统拆成可复用的 `Method -> Interface -> Task` 三层结构，让算法和应用领域能够自由组合，而不是每个项目都从零重写基础设施。

---

## 核心特性

- **LLM 驱动进化**：用语言模型生成、修改并筛选候选解
- **多算法支持**：内置 EoH、EvoEngineer、FunSearch
- **统一架构**：覆盖 Python、字符串、CUDA、控制、CANN 等任务
- **简洁顶层 API**：通过 `evotoolkit.solve(...)` 直接运行
- **双语文档**：提供教程、安装说明和 API 参考

### 内置任务族

| 任务类型 | 说明 | 入口 |
|----------|------|------|
| 科学回归 | 在真实数据集上做符号回归 | [科学回归教程](tutorials/built-in/scientific-regression.zh.md) |
| Prompt 优化 | 优化提示模板 | [Prompt 工程教程](tutorials/built-in/prompt-engineering.zh.md) |
| 对抗攻击 | 进化黑盒攻击算法 | [对抗攻击教程](tutorials/built-in/adversarial-attack.zh.md) |
| CUDA 工程 | 进化 CUDA 内核 | [CUDA 教程](tutorials/built-in/cuda-task.zh.md) |
| 控制任务 | 进化可解释控制策略 | [Control Box2D 教程](tutorials/built-in/control-box2d.zh.md) |
| CANN Init | 生成 Ascend C 算子内核 | [CANN Init 教程](tutorials/built-in/cann-init.zh.md) |

---

## 快速开始

### 安装

```bash
pip install evotoolkit

# 可选任务依赖
pip install "evotoolkit[scientific_regression]"
pip install "evotoolkit[prompt_engineering]"
pip install "evotoolkit[adversarial_attack]"
pip install "evotoolkit[cuda_engineering]"
pip install "evotoolkit[control_box2d]"
pip install "evotoolkit[cann_init]"
pip install "evotoolkit[all_tasks]"
```

公开发布的 Python 包当前在 Python 3.10-3.12 上测试通过。CUDA 与 CANN 工作流除了这些 Python 依赖外，还需要对应的硬件和厂商工具链。

### 第一个优化任务

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.tools import HttpsApi

task = ScientificRegressionTask(dataset_name="bactgrow")
interface = EvoEngineerPythonInterface(task)

llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key-here",
    model="gpt-4o",
)

result = evotoolkit.solve(
    interface=interface,
    output_path="./results",
    running_llm=llm_api,
    max_generations=5,
)
```

完整上手流程见 [快速开始](getting-started.zh.md)。

---

## 算法

| 算法 | 说明 |
|------|------|
| **EvoEngineer** | 结构化算子驱动的 LLM 进化搜索 |
| **FunSearch** | 带 island database 的程序搜索 |
| **EoH** | 以启发式进化为核心的多算子方法 |

---

## 文档导航

- **[安装](installation.md)**：环境与依赖说明
- **[快速开始](getting-started.zh.md)**：第一个可运行示例
- **[教程](tutorials/index.md)**：端到端任务教程
- **[API 参考](api/index.md)**：公开接口与任务说明
- **[开发文档](development/contributing.zh.md)**：贡献与维护流程

---

## 链接

- **GitHub**: [https://github.com/pgg3/evotoolkit](https://github.com/pgg3/evotoolkit)
- **PyPI**: [https://pypi.org/project/evotoolkit/](https://pypi.org/project/evotoolkit/)
- **更新日志**: [CHANGELOG.md](https://github.com/pgg3/evotoolkit/blob/master/CHANGELOG.md)
- **相对既有工作的软件改进**: [SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md](https://github.com/pgg3/evotoolkit/blob/master/SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md)

---

## 许可证

EvoToolkit 核心框架使用 MIT License。部分面向硬件的可选工作流依赖外部厂商工具链，相关外部要求请查看各任务的安装说明。

---

## 引用

如果你在研究中使用 EvoToolkit，建议引用你实际使用的软件版本或仓库快照。一个可用的仓库引用条目如下：

```bibtex
@software{guo2026evotoolkit,
  author = {Guo, Ping and Zhang, Qingfu},
  title = {evotoolkit},
  year = {2026},
  url = {https://github.com/pgg3/evotoolkit},
  version = {1.0.0rc6}
}
```

---

## 获取帮助

- **Issues**: [GitHub Issues](https://github.com/pgg3/evotoolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions)
- **Email**: pguo6680@gmail.com
