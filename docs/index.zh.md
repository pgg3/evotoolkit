# EvoToolkit

**面向可执行解的 LLM 驱动进化优化框架**

EvoToolkit 是一个 Python 框架，用大语言模型来进化代码、符号表达式、提示词以及其他可执行文本。它把系统拆成可复用的 `Method -> Interface -> Task` 三层结构，让算法、适配层与执行基础设施能够自由组合，而不是每个项目都从零重写。

---

## 核心特性

- **LLM 驱动进化**：用语言模型生成、修改并筛选候选解
- **多算法支持**：内置 EoH、EvoEngineer、FunSearch
- **统一架构**：以算法与参考任务适配层为中心
- **简洁顶层 API**：通过 `evotoolkit.solve(...)` 直接运行
- **双语文档**：提供教程、安装说明和 API 参考

### 参考任务族

| 任务类型 | 角色 | 入口 |
|----------|------|------|
| 科学回归 | CPU 可审查的符号回归参考适配层 | [科学回归教程](tutorials/built-in/scientific-regression.zh.md) |
| Prompt 优化 | CPU 可审查的字符串优化参考适配层 | [Prompt 工程教程](tutorials/built-in/prompt-engineering.zh.md) |
| 对抗攻击 | CPU 可审查的算法进化参考适配层 | [对抗攻击教程](tutorials/built-in/adversarial-attack.zh.md) |
| CUDA 工程 | 可选的硬件相关参考任务族；主审查面聚焦任务壳层与接口层 | [CUDA 教程](tutorials/built-in/cuda-task.zh.md) |
| 控制任务 | CPU 可审查的控制策略参考适配层 | [Control Box2D 教程](tutorials/built-in/control-box2d.zh.md) |
| CANN Init | 实验性相邻工作流，不属于主 reviewed surface | [CANN Init 教程](tutorials/built-in/cann-init.zh.md) |

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
- **[教程](tutorials/index.md)**：参考任务与示例脚本的端到端教程
- **[API 参考](api/index.md)**：核心框架与参考适配层说明
- **[开发文档](development/contributing.zh.md)**：贡献与维护流程

### Reviewer 文档

- **[相对既有工作的软件改进](https://github.com/pgg3/evotoolkit/blob/master/SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md)**：框架相对既有 domain artifact 的软件增量
- **[Reviewed Surface 定义](https://github.com/pgg3/evotoolkit/blob/master/REVIEWED_SURFACE.md)**：MLOSS 主审查边界
- **[测试与覆盖率](https://github.com/pgg3/evotoolkit/blob/master/TESTING.md)**：验证命令与覆盖率口径说明

---

## 链接

- **GitHub**: [https://github.com/pgg3/evotoolkit](https://github.com/pgg3/evotoolkit)
- **PyPI**: [https://pypi.org/project/evotoolkit/](https://pypi.org/project/evotoolkit/)
- **更新日志**: [CHANGELOG.md](https://github.com/pgg3/evotoolkit/blob/master/CHANGELOG.md)
- **相对既有工作的软件改进**: [SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md](https://github.com/pgg3/evotoolkit/blob/master/SOFTWARE_IMPROVEMENTS_FROM_PRIOR_WORK.md)

---

## 许可证

EvoToolkit 核心框架使用 MIT License。部分硬件相关的可选工作流依赖外部厂商工具链，相关外部要求请查看各任务的安装说明。

---

## 引用

如果你在研究中使用 EvoToolkit，建议引用你实际使用的软件版本或仓库快照。一个可用的仓库引用条目如下：

```bibtex
@software{guo2026evotoolkit,
  author = {Guo, Ping and Zhang, Qingfu},
  title = {evotoolkit},
  year = {2026},
  url = {https://github.com/pgg3/evotoolkit},
  version = {1.0.0}
}
```

---

## 获取帮助

- **Issues**: [GitHub Issues](https://github.com/pgg3/evotoolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pgg3/evotoolkit/discussions)
- **Email**: pguo6680@gmail.com
