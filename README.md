<div align="center">

# EvoToolkit

**LLM-driven program search, reduced to five composable parts.**

[![CI](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg3/evotoolkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![Python](https://img.shields.io/pypi/pyversions/evotoolkit)](https://pypi.org/project/evotoolkit/)
[![Docs](https://readthedocs.org/projects/evotoolkit/badge/?version=latest)](https://evotoolkit.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

English · [简体中文](https://github.com/pgg3/evotoolkit/blob/master/docs/README.zh.md)

</div>

---

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

<div align="center">

[MIT License](LICENSE) · © Ping Guo

</div>
