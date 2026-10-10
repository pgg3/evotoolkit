<div align="center">

# EvoToolkit

**LLM-driven program search, reduced to five composable parts.**

[![CI](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![Python](https://img.shields.io/pypi/pyversions/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![Docs](https://readthedocs.org/projects/evotoolkit/badge/?version=latest)](https://evotoolkit.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](#english) · [简体中文](#简体中文)

</div>

---

## English

EvoToolkit is a small Python harness for evolving executable artifacts with language models: Python modules, CUDA kernels, prompts, or any set of text files. You write one **Evaluator**; the toolkit supplies the search loop, the strategies, the LLM and coding-agent adapters, and a replayable event log.

### Why 2.0

| | |
|---|---|
| **One loop, many strategies** | `EoHStrategy`, `EvoEngineerStrategy` and `FunSearchStrategy` share the same `Runner`, budget accounting and log format, so results are comparable. |
| **Chat models or coding agents** | `ChatProposer` wraps any chat-completions endpoint. `AgentProposer` drives Claude Code, Codex and friends through [headless-cli](https://github.com/RobertTLange/headless-cli). |
| **Artifacts are file sets** | A solution is `dict[path, content]` with a content-hash identity. Single-file tasks set `main_file` and never notice. |
| **Replayable, no pickle** | Every solution, failed call and RNG snapshot goes to `events.jsonl`; full prompts and transcripts go to `traces/`. `Runner.resume()` rebuilds state from the log. |
| **Honest budgets** | `budget` counts evaluated proposals. Parse errors and empty responses are failed calls that do not spend it; `max_calls` caps the total. |
| **Generational or steady** | Batch-and-observe, or continuously refill proposer workers and observe each result as it lands. |

### Install

```bash
pip install evotoolkit
```

Python 3.10+. The only runtime dependency is NumPy.

### Quick start

```python
from evotoolkit import EvaluationResult, Runner, TaskSpec
from evotoolkit.proposers import ChatProposer
from evotoolkit.strategies import EoHStrategy
from evotoolkit.tools import HttpsApi


class Evaluator:
    spec = TaskSpec("greeting", "Write exactly hello in answer.txt.", "string", "answer.txt")

    def evaluate(self, files):
        text = files.get("answer.txt", "").strip()
        return EvaluationResult(True, float(text == "hello"), "")


evaluator = Evaluator()
llm = HttpsApi(api_url="https://api.openai.com/v1/chat/completions", key="your-key", model="gpt-4o")
runner = Runner(
    evaluator=evaluator,
    proposer=ChatProposer(llm),
    strategy=EoHStrategy(evaluator.spec),
    output_dir="results/greeting",
    budget=20,
    seed=0,
)
best = runner.run()
print(best.files if best else "No valid result")
```

Swap `EoHStrategy` for `EvoEngineerStrategy` or `FunSearchStrategy` without touching the rest. To try it without an API key, run the bundled offline example:

```bash
git clone https://github.com/pgg3/evotoolkit && cd evotoolkit
uv sync --group dev
uv run python examples/custom_task/my_custom_task.py --offline --output-dir results/demo
```

### How it fits together

```
Runner ──► Strategy ──► Archive        which parents, which instruction?
  ├──────► Proposer                    new artifacts from parents + instruction
  ├──────► Evaluator                   how good is this artifact?
  └──────► EventLog                    append-only JSONL, replayed on resume
```

Components are protocols, not base classes: any object with the right methods works. See [Extensions and API](docs/extensions.md) for the contracts and built-ins.

### Resume

```python
runner = Runner.resume("results/greeting", evaluator=evaluator,
                       proposer=ChatProposer(llm), strategy=EoHStrategy(evaluator.spec))
best = runner.run()
```

### Documentation

[evotoolkit.readthedocs.io](https://evotoolkit.readthedocs.io/) · [Quick start](docs/quickstart.md) · [Extensions and API](docs/extensions.md) · [Design](docs/design.md)

### Upgrading from 1.x

2.0 is a clean break. `Method`, `Task`, `Interface`, the per-method state classes and pickle checkpoints are gone, with no shims. Pin `evotoolkit<2` to stay on 1.x.

### Notes

Child processes and temporary workspaces are not security sandboxes; isolate untrusted code yourself. `AgentProposer` needs a separately installed `headless` CLI and an authenticated backend.

---

## 简体中文

EvoToolkit 是一个用语言模型演化可执行产物的轻量 Python 框架：Python 模块、CUDA 内核、提示词，或任意一组文本文件。你只需写一个 **Evaluator**，其余的搜索循环、策略、LLM 与编码 agent 适配器、可回放的事件日志都由工具包提供。

### 2.0 的特点

| | |
|---|---|
| **一个循环，多种策略** | `EoHStrategy`、`EvoEngineerStrategy`、`FunSearchStrategy` 共用同一个 `Runner`、同一套预算计数和日志格式，结果可直接比较。 |
| **聊天模型或编码 agent** | `ChatProposer` 适配任意 chat-completions 接口；`AgentProposer` 通过 [headless-cli](https://github.com/RobertTLange/headless-cli) 驱动 Claude Code、Codex 等。 |
| **产物是文件集合** | 一个解就是 `dict[路径, 内容]`，以内容哈希作为身份。单文件任务只需设置 `main_file`。 |
| **可回放，不用 pickle** | 每个解、每次失败调用和 RNG 快照都写入 `events.jsonl`，完整提示与对话记录写入 `traces/`。`Runner.resume()` 从日志重建状态。 |
| **诚实的预算** | `budget` 只计已评估的提案。解析失败或空响应算作失败调用，不消耗预算；`max_calls` 限制总调用数。 |
| **代际或稳态** | 按批次选择并观察，或持续填满 proposer 工作线程、逐个观察结果。 |

### 安装

```bash
pip install evotoolkit
```

需要 Python 3.10+，运行时仅依赖 NumPy。

### 快速开始

```python
from evotoolkit import EvaluationResult, Runner, TaskSpec
from evotoolkit.proposers import ChatProposer
from evotoolkit.strategies import EoHStrategy
from evotoolkit.tools import HttpsApi


class Evaluator:
    spec = TaskSpec("greeting", "Write exactly hello in answer.txt.", "string", "answer.txt")

    def evaluate(self, files):
        text = files.get("answer.txt", "").strip()
        return EvaluationResult(True, float(text == "hello"), "")


evaluator = Evaluator()
llm = HttpsApi(api_url="https://api.openai.com/v1/chat/completions", key="your-key", model="gpt-4o")
runner = Runner(
    evaluator=evaluator,
    proposer=ChatProposer(llm),
    strategy=EoHStrategy(evaluator.spec),
    output_dir="results/greeting",
    budget=20,
    seed=0,
)
best = runner.run()
print(best.files if best else "No valid result")
```

把 `EoHStrategy` 换成 `EvoEngineerStrategy` 或 `FunSearchStrategy`，其余代码不变。不想配 API key，可以直接跑自带的离线示例：

```bash
git clone https://github.com/pgg3/evotoolkit && cd evotoolkit
uv sync --group dev
uv run python examples/custom_task/my_custom_task.py --offline --output-dir results/demo
```

### 组件关系

```
Runner ──► Strategy ──► Archive        选哪些父代、用哪条指令？
  ├──────► Proposer                    由父代 + 指令生成新产物
  ├──────► Evaluator                   这个产物有多好？
  └──────► EventLog                    追加写 JSONL，resume 时回放
```

组件是 protocol 而非基类：任何有对应方法的对象都能用。契约与内置实现见[扩展与 API](docs/extensions.zh.md)。

### 断点续跑

```python
runner = Runner.resume("results/greeting", evaluator=evaluator,
                       proposer=ChatProposer(llm), strategy=EoHStrategy(evaluator.spec))
best = runner.run()
```

### 文档

[evotoolkit.readthedocs.io](https://evotoolkit.readthedocs.io/zh/latest/) · [快速开始](docs/quickstart.zh.md) · [扩展与 API](docs/extensions.zh.md) · [设计](docs/design.zh.md)

### 从 1.x 升级

2.0 是一次彻底切换。`Method`、`Task`、`Interface`、各方法的 state 类和 pickle 检查点全部移除，没有兼容层。需要留在 1.x 请固定 `evotoolkit<2`。

### 说明

子进程和临时工作目录不是安全沙箱，运行不可信代码请自行隔离。`AgentProposer` 需要单独安装 `headless` CLI 并完成后端认证。

---

<div align="center">

[MIT License](LICENSE) · © Ping Guo

</div>
