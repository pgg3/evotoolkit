# 快速开始

在仓库中执行：

```bash
uv sync --group dev
uv run python examples/custom_task/my_custom_task.py --offline --output-dir results/demo
uv run python examples/custom_task/my_custom_task.py --offline --mode steady --output-dir results/steady
```

离线示例无需密钥：生成 12 个 Python 函数提案，在子进程中计算与 sin(x) 的误差，返回最佳文件。真实模型调用需设置 `LLM_API_KEY`，可选 `LLM_API_URL`、`LLM_MODEL`，并去掉 `--offline`。

## 组装搜索

```python
from evotoolkit import EvaluationResult, Runner, TaskSpec
from evotoolkit.proposers import ChatProposer
from evotoolkit.strategies import EvoEngineerStrategy

class GreetingEvaluator:
    spec = TaskSpec("greeting", "在 answer.txt 中写 hello。", "string", "answer.txt")

    def evaluate(self, files):
        return EvaluationResult(True, float(files.get("answer.txt", "").strip() == "hello"), "")

# llm.get_response(messages) 返回 (响应文本, usage 字典)。
evaluator = GreetingEvaluator()
runner = Runner(evaluator=evaluator, proposer=ChatProposer(llm),
                strategy=EvoEngineerStrategy(evaluator.spec), output_dir="results/greeting",
                budget=20, seed=42, propose_workers=4, eval_workers=4,
                mode="generational")
best = runner.run()
```

可直接替换为 `EoHStrategy(spec)` 或 `FunSearchStrategy(spec)`。分数越大越好；失败返回 `EvaluationResult(False, None, feedback)`。

## 预算与并发

`budget` 统计实际提案，包含评估无效和重复文件。失败、空返回或解析失败的调用记录为 `call_failed`，不扣提案预算。`max_calls` 限制全部调用，默认 `max(1, 3 * budget)`；崩溃时未完成的调用也计入。Runner 预留在途提案额度，避免并发超预算。返回数量超过请求数量时拒绝整个调用。

`generational` 一批全部完成后更新档案并选择下一批；`steady` 持续填补空闲 proposer worker，每个评估完成后立即更新档案。单个调用完成即可进入独立评估线程池，无需等待其他调用。`step()` 执行一个有限批次，`run()` 才持续调度 steady 模式。

`runner.samples`、`calls`、`failed_calls`、`steps`、`status`、`best_solution` 提供进展。状态分别为预算耗尽 `completed`、调用上限 `max_calls`、策略停止或无任务 `stopped`。`run()` 自动关闭日志；手动逐步执行时使用上下文管理器或 `close()`。

## 恢复

```python
runner = Runner.resume("results/greeting", evaluator=evaluator,
                       proposer=ChatProposer(llm), strategy=EvoEngineerStrategy(evaluator.spec))
best = runner.run()
```

传入同配置的新策略，档案必须为空。任务规格及组件名称需一致。预算、线程数、调度模式和随机数状态从日志恢复；已落盘结果重建档案，未完成调用只占调用额度。外部模型状态和线程完成顺序无法恢复，因此不承诺继续运行的结果逐位一致。

`events.jsonl` 保存完整文件、评估、父代、计数器和随机数状态。`traces/<call_id>.json` 保存完整提示、响应或 agent 转录。事件中的反馈限长 4 KiB，trace 不截断。usage 按调用计算，汇总 token 时应按 `Solution.call_id` 去重。
