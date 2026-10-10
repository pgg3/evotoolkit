# EvoToolkit 2.0 设计

状态：已实现并作为 2.0.0 发布。实现与原草案不一致的地方，都记在它所属的小节里。使用方式见 `extensions.zh.md`。

2.0 与 1.x 彻底切断：没有兼容 shim，没有弃用路径。
英文版见 `design.md`，以英文版为准。

## 1. 范围

### EvoToolkit 是什么

一个最小化的 LLM 驱动程序搜索 harness。它服务的问题形状是：

- 一个**可执行的产物**，以文件集的形式存储（一个 Python 模块、一个 CUDA 内核加它的 launcher、一个 prompt 文件……），
- 一个**评估器**，运行产物并返回分数，
- 一个**预算**（提案次数，外加 proposer 调用次数上限；见不变量 6）。

工具做的全部事情就是：从 archive 里挑父代，让 proposer 产出新产物，评估它们，记录它们，重复。

### 谁用它

| 用户 | 需要什么 |
|---|---|
| 对比搜索策略的研究者 | FunSearch、EoH、EvoEngineer 和新策略跑在同一个循环、同一套预算记账、同一种事件日志上。 |
| 对比 proposer 的研究者 | API 模型和 coding agent（Claude Code、Codex……）在同一任务上对比，成本有记录。 |
| 优化内核或启发式的实践者 | 写一个 `Evaluator`，选一个策略，跑。 |
| 分析者 | 读 JSONL 事件日志，重建谱系、分数轨迹、停滞情况。 |

### 使用场景

1. 组合优化的启发式设计。评估器在实例集上取平均 gap。
2. 内核或算子优化（EvoEngineer）。评估器负责编译、正确性检查、profiling。
3. 数学构造。评估器是 checker。
4. prompt 或配置搜索。评估器是下游任务分数。
5. 以后：对指令模板本身做元层面搜索。现在不做，但设计不能堵死它。

## 2. 结构

### 组件

五个组件，每个只回答一个问题。箭头表示允许的依赖方向。

```
Runner  -->  Strategy  -->  Archive
  |             |
  |             +--->  (指令模板：是数据，不是代码)
  +--->  Proposer
  +--->  Evaluator
  +--->  EventLog
```

| 组件 | 回答的问题 | 知道什么 | 不许知道什么 |
|---|---|---|---|
| Evaluator | 这个产物多好？ | 产物的文件 | proposer、archive、线程 |
| Proposer | 给定父代和指令，能产出哪些新产物？ | ProposalContext | archive、打分、线程 |
| Archive | 存什么、怎么采样父代？ | Solution 和分数 | LLM、prompt、磁盘文件 |
| Strategy | 这一步选哪些父代、下什么指令？ | Archive、模板 | LLM 调用、线程、磁盘文件 |
| Runner | 循环、并发、预算、持久化。 | 以上全部 | prompt 文本、选择策略 |

### 数据对象

```python
Files = dict[str, str]          # 相对路径 -> 内容

@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt: str                 # 给 proposer 看的任务描述
    modality: str               # "python" | "cuda" | "string" | ...
    main_file: str              # 单文件 proposer 写的那个文件，如 "solution.py"
    extras: dict[str, Any]

@dataclass
class EvaluationResult:
    valid: bool
    score: float | None         # 越大越好
    feedback: str               # 人和 LLM 可读：报错、profiler 输出……
    metrics: dict[str, Any]     # 结构化的附加指标，包括任何目标向量
    cost_seconds: float

@dataclass(frozen=True)
class Instruction:
    name: str                   # "i1", "e1", "funsearch", "diff_refine", ...
    num_parents: int
    text: str                   # 与模态无关；可用 {modality}、{main_file} 占位
    output_format: str          # "code_block" | "name_code_thought" | "files"

@dataclass
class ProposalContext:
    spec: TaskSpec
    parents: list["Solution"]
    instruction: Instruction
    best: "Solution | None"
    hints: list[str]            # 自由文本，例如其他解的描述
    num_proposals: int          # Strategy 这次想要几个提案
    call_id: str                # Runner 分发前分配，proposer 原样回填

@dataclass
class Proposal:
    call_id: str                # 同一次 proposer 调用产出的提案共享
    files: Files
    name: str
    rationale: str
    usage: dict[str, Any]       # 按调用记，不按提案记；用 call_id 去重

@dataclass
class Solution:
    id: str                     # 文件内容哈希，取 16 位十六进制
    files: Files
    name: str
    rationale: str
    parent_ids: list[str]
    instruction: str            # Instruction.name
    proposer: str               # Proposer.name
    call_id: str
    usage: dict[str, Any]
    evaluation: EvaluationResult | None
    created_step: int
```

`Solution.id = sha256("\0".join(f"{path}\0{content}" for path in sorted(files)))[:16]`。

数据模型里不再有 GA 的标签。crossover 就是 `num_parents=2` 的指令，mutation 就是 `num_parents=1` 的指令。EoH 的五个算子和 FunSearch 的一个算子都变成模板表里的行。

### 不变量

1. **先写日志再进 archive。** Solution 先追加进事件日志，再插入 archive。日志只追加不修改。
2. **状态是日志的函数。** `Runner.resume()` 通过重放日志重建 archive 和计数器。不用 pickle。
3. **Strategy 是纯的。** `Strategy.next_batch()` 读 archive、返回 context。不做 I/O，不调 LLM，不开线程。并行度完全归 Runner：Runner 通过 `max_n` 告诉 Strategy 现在有几个空位，并且串行调用 `observe`，所以 Strategy 内部永远不需要锁。
4. **Proposer 跨调用无状态。** 任何记忆都放在 archive 或日志里。
5. **内容寻址的身份。** 文件完全相同的两个解 `id` 相同。archive 可以拒绝重复，但日志仍然记录这次尝试。
6. **预算只在 Runner 里计数，按提案数计。** Strategy 看到的 `remaining_budget` 是一个数字，它从不扣减。失败的 proposer 调用产出零个提案，不消耗提案预算；另有 `max_calls` 上限约束总调用次数。
7. **提案之间唯一的共享通道是 archive。** 任何不经过 archive 的依赖都是某个 Proposer 内部的事，记录在它的 trace 里。跨评估的依赖由 Strategy 通过 `observe` 和下一次 `next_batch` 表达；多个 proposer 串联则组合成一个 Proposer。核心不建模提案之间的依赖图。
8. **Evaluator 永不抛异常。** 失败就是 `valid=False` 并填写 `feedback`。隔离和超时是 Evaluator 的职责；Runner 只测 `cost_seconds`，并作为最后兜底捕获异常。

## 3. 接口契约

所有契约都是 `typing.Protocol`。实现方不继承。

```python
class Evaluator(Protocol):
    spec: TaskSpec
    def evaluate(self, files: Files) -> EvaluationResult: ...

class Proposer(Protocol):
    name: str
    def propose(self, ctx: ProposalContext) -> list[Proposal]: ...   # 可以抛异常

class Archive(Protocol):
    def add(self, solution: Solution) -> None: ...
    def sample(self, n: int, *, rng: np.random.Generator) -> list[Solution]: ...
    def best(self) -> Solution | None: ...
    def __len__(self) -> int: ...

class Strategy(Protocol):
    name: str
    archive: Archive
    def next_batch(self, step: int, max_n: int, remaining_budget: int, rng: np.random.Generator) -> list[ProposalContext]: ...
    def observe(self, solutions: list[Solution]) -> None: ...   # 加进 archive、裁剪等
    def should_stop(self, step: int) -> bool: ...
```

Runner 是具体类，不是 protocol：

```python
class Runner:
    def __init__(self, *, evaluator, proposer, strategy, output_dir, budget,
                 max_calls=None, propose_workers=4, eval_workers=4,
                 mode="generational", seed=0, verbose=True): ...
    def run(self) -> Solution | None: ...
    def step(self) -> None: ...
    @classmethod
    def resume(cls, output_dir, *, evaluator, proposer, strategy) -> "Runner": ...
```

`Runner.step()` 只做这些事：`ctxs = strategy.next_batch(...)`，把它们送进两级流水线，构造 Solution，逐个追加到日志，`strategy.observe(solutions)`，更新计数器，写 checkpoint 标记。

两级流水线，并行度各自独立：

1. **提案级**：`propose_workers` 个线程并发调用 `proposer.propose(ctx)`，每次调用返回一个列表。抛异常则记一条 `call_failed` 事件。
2. **评估级**：每个返回的列表立即拆开，逐个提交给 `eval_workers` 个线程跑 `evaluator.evaluate`，不等同批其他调用。

两种调度模式，同一个 Strategy 都能跑：

- `generational`：发一批，等全部回来，再要下一批。EoH 和 EvoEngineer 的行为，golden 基线按它录。
- `steady`：哪个 worker 空了就立刻 `next_batch(max_n=1)` 再要一个，不等别人。FunSearch 原版和 AlphaEvolve 的行为。LLM 延迟方差大或评估很快时吞吐更高。

### 内置 Proposer

- `ChatProposer(llm, templates)`：用指令的 `text` 和 `output_format` 把 `ctx` 格式化成 messages，调 LLM，按 `output_format` 解析成 `Files`。`code_block` 和 `name_code_thought` 写入 `spec.main_file`；`files` 解析多个带路径标记的围栏块。把现在的六个 interface 文件吸收成一张模板表。
- `AgentProposer(backend, workdir_factory)`：物化一个工作区（`TASK.md` 描述全部父代，落盘最后一个父代的文件，`evaluate.py` 仅在 `workdir_factory` 或 `evaluation_command` 提供时生成），通过 `headless-cli` 跑 headless coding agent，读回主文件、父代路径以及 `ARTIFACTS.json` 声明的文件。agent 轨迹就是 trace。

### 内置 Evaluator 辅助

- `subprocess_evaluate(files, entry, timeout)`：把文件写进临时目录，在子进程里带超时运行 `entry`，返回 stdout、stderr 和退出码。任务族自行选用；核心不强制隔离。

### 内置 Archive

- `RankedPopulation(max_size)`：1.x 的 `PopulationState` 加基于排名的采样。偏离：无效解一律不存；1.x 在种群有空位时会用无效解填充。
- `IslandArchive(num_islands, ...)`：1.x 的 `ProgramsDatabase`，为了让 `resume` 只靠日志就能重建，做了两处有意的改动。新解按内容哈希分配到岛屿，而不是随机抽；弱岛在每累计 `reset_interval` 个被接受的唯一解后重置，而不是按墙钟定时。

### 内置 Strategy

- `FunSearchStrategy`：一个指令，`num_parents` 2，IslandArchive。
- `EoHStrategy`：指令 i1、e1、e2、m1、m2，保留算子语义、统一改写到共享模板（模板还会加入分数、反馈和当前最优），RankedPopulation。偏离：1.x 每代把每个算子在整个种群上跑一遍；2.0 把算子轮流分配到 Runner 给出的空位上，一次 `next_batch(max_n=4)` 得到 e1、e2、m1、m2 各一个。
- `EvoEngineerStrategy`：init、crossover、mutation，带当前最优和 hints，RankedPopulation。

每个都远少于 150 行。

## 4. 持久化布局（v1）

```
<output_dir>/
  events.jsonl            只追加，每行一个 JSON 对象
  traces/<call_id>.json   完整的 prompt messages 和原始回复，或 agent 轨迹
```

事件类型：

```json
{"v": 1, "type": "run_start", "ts": "...", "task": {TaskSpec}, "strategy": "eoh", "proposer": "chat:gpt-4o", "seed": 0, "budget": 45, "config": {"max_calls": 135, "propose_workers": 4, "eval_workers": 4, "mode": "generational", "verbose": true}, "env": {"python": "3.12.1", "platform": "linux", "evotoolkit": "2.0.0"}}
{"v": 1, "type": "call_started", "ts": "...", "step": 3, "call_id": "...", "instruction": "e1", "rng_state": {...}}
{"v": 1, "type": "solution", "ts": "...", "step": 3, "solution": {Solution}}
{"v": 1, "type": "call_failed", "ts": "...", "step": 3, "call_id": "...", "instruction": "e1", "error": "..."}
{"v": 1, "type": "step_end", "ts": "...", "step": 3, "archive_size": 5, "best_id": "ab12...", "best_score": 0.91, "proposals": 4, "valid": 3, "calls": 2, "failed_calls": 0, "rng_state": {...}}
{"v": 1, "type": "run_end", "ts": "...", "status": "completed", "steps": 10, "samples": 45}
```

`call_started` 事件和 `config`、`rng_state` 字段是实现期间加的。`call_started` 让 `resume` 能数到那些没产出结果的调用，`max_calls` 在崩溃后仍然有效。`rng_state` 是 numpy 位生成器在每次调用前和每步结束时的状态，恢复后父代选择沿着原来的随机流继续，而不是从头开始。

规则：任何破坏性字段变更都要升 `v`。读取方必须忽略未知字段。`Solution.files` 在日志里完整保存；`feedback` 由 Runner 在解进入 archive 之前截断到 4 kB，因此运行中的 archive、据此生成的 prompt 和恢复后的 archive 完全一致（日志写入方保留 `feedback_truncated: true` 兜底）。`run_start.config.strategy` 记录策略超参数，`resume` 拒绝 `config` 不一致的策略。每次调用的 `call_id` 由 Runner 生成一次，`call_started`、`call_failed`、所有产出的 `solution` 和 trace 文件共用。trace 永不截断。进程被打断留下的末尾半行，读取方跳过，下一次以写模式打开日志时删除；其他格式错误的行一律报错。

## 5. 行为基线

`tests/golden/` 用脚本化 LLM 和固定种子，为每个内置策略记录：每一步的 `(主文件内容, valid, score, 父代 id)` 排序多重集、下发的指令、最优分数、失败调用数；以及最终的样本数、调用数和状态。

原计划是用 1.x 录制、要求 2.0 复现。这在构造上做不到：不变量 6 规定无法解析的回复是一次失败调用、不扣预算，而 1.x 把它算作一个无效样本，所以同样的脚本化回复下，步骤边界和样本数必然不同。因此 1.x 快照已退役（重写前的 git 历史里还有），`*-v2.json` 改为用 2.0 录制。人工核对过的较弱证据：三个策略在 1.x 和 2.0 下产出的有效程序集合和最终最优分完全一致。

v2 文件现在防的是 2.0 自身的意外漂移，不是相对 1.x 的漂移。只有在有意改变行为时才重录，并在提交信息里说明。

已知上限：线程完成顺序不确定，所以基线记录的是每步的多重集而不是序列。父代选择发生在主线程、分发之前，且 generational 批次按 id 排序后再交给 `observe`，所以 archive 状态和选择都能由种子复现（加入父代 id 列后，才发现岛屿播种此前依赖线程完成顺序）。

## 6. 推迟的决定

考虑过、暂不做、等到有真实需求再回头看的东西。每条写清楚触发条件。

- **提案依赖图。** 见不变量 7。触发条件：接入 agent 后，出现一种依赖形态既不在一次调用内部、又不经过 archive。
- **steady 模式的基线。** 结果依赖时序，不能用 golden 对比，只能验证不变量（日志解数等于预算、每个解的父代 id 都在它之前出现）。
- **多目标选择。** `score` 保持单个浮点数、越大越好；目标向量放 `metrics`。多目标是一个按 `metrics` 采样的新 Archive 实现，核心不用改。触发条件：某个任务族和某个策略都需要它。
- **元层面搜索。** 指令模板是数据，能序列化成文件，所以一张模板表可以当 Solution。它的 Evaluator 用小预算跑一个内层 `Runner`，给内层最优打分。核心已有的东西够用。触发条件：场景 5 有人真的要做。

## 7. 决策

**Python，不是 Node 或 Rust。** 评估器活在 Python 生态里（numpy、torch、triton、运筹实例库）。瓶颈是 LLM 延迟和评估器运行时间，不是 harness。放弃：Rust 核心加 Python 绑定，因为换不来任何可测量的收益，还让构建面翻倍。

**产物是文件集，chat proposer 也一样。** 内核是内核加 launcher；agent 改的是文件。单字符串以后会逼出第二套身份方案。单文件任务设好 `main_file` 就感觉不到差别。

**JSONL 事件日志加 trace 文件，不是 pickle。** pickle 把 checkpoint 绑死在类布局上，分析工具也读不了。完整 prompt 放 `traces/`，日志保持可 grep 的体积。放弃：SQLite，因为单写者的追加文件更简单、可 diff。

**失败的调用不消耗提案预算。** 预算是研究变量，一个不稳定的 API 不该悄悄把它吃掉。`max_calls` 防死循环。

**隔离是 Evaluator 的职责。** 核心提供 `subprocess_evaluate` 但不强制。任务族自己知道自己的信任要求。

**Protocol，不是 ABC。** 用户应该能传任何有正确方法的对象。放弃：ABC 继承，因为它迫使用户导入我们的基类，并绑定到我们的构造函数签名。

**用 headless-cli 子进程接 agent，不写逐个 agent 的适配器。** 一个依赖已经统一了 Claude Code、Codex、Cursor、Gemini、OpenCode。放弃：自己写适配器，那是没有研究价值的维护工作。

**指令是数据，且与模态无关。** prompt 模板是行，不是子类；围栏和解析放在 `output_format`。六个文件收成一张表。这让 EoH 的算子和 FunSearch 的 prompt 可比，也让元层面（场景 5）变成「进化这些行」的问题。

**`output_format` 是字符串枚举。** 解析器在 `ChatProposer` 里按名字查表。放弃：把解析函数挂在 `Instruction` 上，因为模板必须能序列化成 JSON 或 YAML。

**不兼容 1.x。** 一次发版彻底切断。放弃：shim 和弃用周期，因为用户基数小，而且旧表面正是要被替换的东西。

**除了 evolution 之外不用生物学词汇。** Proposer、Proposal、Archive、Strategy、Evaluator、Instruction。放弃：gene、capsule、organism、crossover、mutation 作为核心名字。
