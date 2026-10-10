# 扩展与 API

## 组件契约

不要求继承基类；实现相应成员即可。类型从 `evotoolkit.core` 导入。

| 组件 | 成员 |
|---|---|
| Evaluator | `spec: TaskSpec`；`evaluate(files) -> EvaluationResult` |
| Proposer | `name`；`propose(ctx) -> list[Proposal]` |
| Archive | `add(solution)`；`sample(n, *, rng)`；`best()`；`len(archive)` |
| Strategy | `name`、`archive`；`next_batch(step, max_n, remaining_budget, rng)`；`observe(solutions)`；`should_stop(step)` |

`Files` 为相对路径到文本内容的字典，路径用 `/` 分隔。TaskSpec 包含 name、prompt、modality、main_file、extras。Instruction 包含 name、num_parents、text、output_format。模板可替换 `{modality}`、`{main_file}`；其他字面花括号需写作 `{{`、`}}`。

ProposalContext 包含任务、父代、指令、最佳解、提示和请求数量。每个 Proposal 必须原样回填 `ctx.call_id`（由 Runner 在分发前生成，携带其他 ID 的提案会被拒绝）；同一调用的提案共享该 ID 和 usage；返回数量不得超过请求。Solution 添加内容哈希、父代 ID、指令与 proposer 名、评估和步号；新解使用 `Solution.create(...)`，序列化使用 `to_dict/from_dict`，读取时忽略未知字段。

策略在主线程串行执行，提案器和评估器会并发调用。选择操作使用传入的 RNG；需恢复的状态放入 archive，通过 observe 更新。评估器自行处理超时和隔离，失败返回无效结果，Runner 另有异常兜底。有效分数必须有限且越大越好，多目标信息放入 metrics。

## 内置组件

- `RankedPopulation(max_size=5)`：只保存有效、有限分数、内容唯一的解；排名加权有放回采样。
- `IslandArchive(num_islands=5, reset_interval=30000, ...)`：分数聚类、温度采样、簇内偏好短产物。首个有效解播种所有岛；后续按内容哈希分配，并按已接受的唯一解数量重置弱岛。它替代旧版的采样岛分配和墙钟重置，以保证日志重放。
- `EvoEngineerStrategy(spec, pop_size=5)`：init、crossover、mutation，附带最佳解和思路提示。
- `EoHStrategy(spec, pop_size=5, selection_num=2)`：i1、e1、e2、m1、m2。
- `FunSearchStrategy(spec, num_islands=5, num_parents=2, reset_interval=30000)`：岛屿父代采样和改进指令。

内置策略由 Runner 的预算或调用上限停止；自定义停止规则实现 should_stop。

## ChatProposer

`ChatProposer(llm, templates=None, *, eventlog=None)` 调用 `llm.get_response(messages)`，返回 `(text, usage)`。templates 是指令名到 Instruction 的覆盖表；默认模板位于 `evotoolkit.proposers.templates.INSTRUCTIONS`。

- code_block：一个 fenced block，写入 main_file，可在前面写 `{思路}`。
- name_code_thought：代码块前写 `name:`，后写 `thought:`。
- files：每个文件一个代码块，开头标签直接写相对路径，例如 src/main.py。

同调用的多个候选用独占一行的 `===PROPOSAL===` 分隔，数量必须等于请求数量。解析失败记录为失败调用，并保留完整 trace。Runner 自动为内置 proposer 绑定 EventLog。

## AgentProposer

`AgentProposer(backend, workdir_factory=None, *, timeout=600, eventlog=None)` 使用单独安装的 headless-cli：

```bash
npm install -g @roberttlange/headless
```

后端认证需自行配置。实际命令是 `headless <backend> --prompt-file TASK.md --work-dir DIR --json --usage --timeout N`，见[上游文档](https://github.com/RobertTLange/headless-cli/blob/main/docs/usage.md)。

每次调用一个独立工作目录，产生一个文件集；num_proposals 必须为 1。最后一个父代的文件写入目录，全部父代及指令写入 TASK.md。`workdir_factory(ctx)` 可返回产出专用目录的上下文管理器，其中可预置 evaluate.py 和支持文件；或在 `spec.extras['evaluation_command']` 提供 argv 列表以生成 evaluate.py。都未配置时，仅在生成后由 Runner 评估。

TASK.md、evaluate.py 及工厂支持文件属于输入，不会作为产物返回。默认只返回 main_file 和已有父代文件；增删文件时由 agent 在 ARTIFACTS.json 中写出相对输出路径的 JSON 数组，清单本身不是产物。这能避免把工作目录里的无关输出误收集为候选。

headless 使用后端自动批准模式；临时目录和子进程不是安全沙箱，运行不可信代码时应由调用方提供 OS 或容器隔离。适配器不安装后端、不管理认证。

## 子进程评估

```python
from evotoolkit.evaluators import subprocess_evaluate
result = subprocess_evaluate({"main.py": "print(42)"}, "main.py", timeout=5)
assert result.returncode == 0
```

entry 字符串表示相对 Python 入口文件，argv 列表表示显式命令。返回 CompletedProcess，含 stdout、stderr、returncode；超时码为 124，准备失败为 1。文件路径经过校验。超时终止直接子进程，不负责完整进程树隔离。

## 日志与恢复

事件类型包括 run_start、call_started、solution、call_failed、step_end、run_end。进程被打断留下的末尾半行，读取时跳过，下次以写模式打开日志时删除；其他格式错误的行会报错。写入使用 flush，没有 fsync；同一目录只允许一个运行写入。

恢复检查任务规格、组件名称以及策略的 `config`（构造超参数）；调用方仍需提供原 proposer 配置。已落盘结果可重建状态，未落盘结果无法恢复。RNG 快照保留选择进度，但并发完成顺序和外部模型响应仍可能变化。
