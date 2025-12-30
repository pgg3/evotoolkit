"""
版本: debug_loop_v0.5 (Dynamic & Atomic & Robust Refactor)
位置: evotoolkit/evo_method/cann_initer/phases/debug_loop.py

================================================================================
[已实现 - 核心架构重构]
1. 原子能力解耦 (Atomic Capabilities):
   将调试过程中的通用能力拆解为独立原子方法，供所有 Agent 复用：
   - _inject_logic_safeguard: 逻辑死锁/超时防御 (针对 FIX_LOGIC)。
   - _record_diagnosis_to_history: 统一的历史记录管理。
   - _save_debug_prompt: 统一的 Prompt 落盘机制 (支持重试计数)。
   - _apply_code_cleaning: 动态代码清洗 (去毒药头文件 + 去中文幻觉)。
   - _check_circuit_breaker: 差异化的熔断检测 (Kernel/Tiling/Pybind 不同阈值)。

2. 通用重试引擎 (Generic Retry Engine):
   实现了 _fix_with_retry 模板方法，统一了 Kernel、Tiling、Pybind 的修复流程。
   消除了不同 Agent 之间的实现不对称性。

================================================================================
[已实现 - 关键特性增强]
1. In-Loop Retry (原地重试机制):
   当熔断机制触发（代码严重坍塌/截断）时，不再退出到外层编译循环，
   而是直接在内部生成严厉警告 Prompt 进行重试。
   -> 收益：避免了重新编译旧代码造成的几十秒算力浪费。

2. 逻辑死锁防御 (Anti-Deadlock):
   当 Analysis Agent 诊断出逻辑错误 (FIX_LOGIC) 时，自动注入防死循环/防越界提示。

3. 动态清洗与正则外科手术:
   基于专家建议 (suggestion_type)，精准剔除 Host 端头文件和标准库依赖。

4. 知识库 自动化分析（实现中）
================================================================================
[待完善 / 未来计划]
1. Tiling 专用分析器:
   目前 _fix_tiling 尚未接入 ErrorAnalyzer (use_analyzer=False)。
   -> 后续需在 analysis_agent 中增加针对 Tiling/Host 代码的诊断规则。

2. M/Hard 模式下的多变量协同:
   目前 Kernel 和 Tiling 是串行修复。在全量生成模式下，两者可能存在强耦合（改了 Tiling 结构体导致 Kernel 编译失败）。
   -> 后续可能需要引入 "Joint Fix" 或更复杂的协同机制。
"""
"""
evotoolkit/examples/cann_init/agent/
├── knowledge_base.json          <-- [核心] 你的最终知识库 (由 MD 转换 + Learner 自动追加)
├── output/
│   └── knowledge_candidates/    <-- [中间态] DebugLoop 甩出来的成功样本
├── _config.py
├── 8_iter_debug.py
├── 11_knowledge_learner.py
└── ...
"""
import time
from ..run_config import CANNIniterConfig
from .analysis_agent import ErrorAnalyzer
from evotoolkit.core import Solution
from typing import TYPE_CHECKING, Callable, Any
import json
import re
from ..parsers import parse_code, parse_json
from pathlib import Path
if TYPE_CHECKING:
    from ..run_config import CANNIniterConfig
    from ..run_state_dict import CANNIniterRunStateDict


class DebugLoop:
    """迭代调试循环 (Refactored v0.5)"""

    def __init__(self, config: "CANNIniterConfig", run_state_dict: "CANNIniterRunStateDict"):
        self.config = config
        self.run_state_dict = run_state_dict
        self.analyzer = ErrorAnalyzer(config.running_llm, config.knowledge_base_path if hasattr(config, 'knowledge_base_path') else "knowledge_base.json")

        # 显式初始化 Analysis 存储
        self.run_state_dict.latest_analysis = None

    def _verbose(self, msg: str):
        if self.config.verbose:
            print(msg)

    def run(self, python_ref: str) -> dict:
        """执行调试循环 (保持原有逻辑，此处仅做占位示意)"""
        last_error_info = None
        last_analysis_result = None
        last_stage = "startup"

        # 定义阶段优劣等级 (Success > Correctness > Deploy > Compile > Startup)
        stage_rank = {
            "startup": 0, "unknown": 0,
            "compile": 1,
            "deploy": 2,
            "correctness": 3,
            "success": 4
        }
        for iteration in range(self.config.max_debug_iterations):
            self.run_state_dict.current_iteration = iteration
            self._verbose(f"[Debug] Iteration {iteration}/{self.config.max_debug_iterations}")
            if iteration == 0:
                print("[Debug] 若Iteration 为 0 则是第一次运行，没有进行debug，实际限制的是编译次数")

            # ... Solution 构建与 evaluate ...
            # 1. 组装当前代码并评估(Compile & Run)
            kernel_src = self.run_state_dict.kernel_src
            other_info = {
                "host_tiling_src": self.run_state_dict.tiling_src,
                "host_operator_src": self.run_state_dict.operator_src,
                "python_bind_src": self.run_state_dict.pybind_src,
                "save_compile_to": None
            }
            solution = Solution(sol_string=kernel_src, other_info=other_info)
            result = self.config.task.evaluate_solution(solution)

            # 获取当前错误信息 (如果是 Success，error_info 为空或无关)
            current_error_info = result.additional_info or {}
            current_stage = "success" if result.valid else current_error_info.get("stage", "unknown")

            # 2. [新增逻辑] [知识捕获核心逻辑] 检查上一轮的策略是否有效
            # 我们在 Iter N，检查的是 Iter N-1 的 Analysis 是否让 Stage 提升了
            prev_analysis = self.run_state_dict.latest_analysis  # 获取上一轮产生的分析
            if iteration > 0 and prev_analysis and last_error_info:
                curr_rank = stage_rank.get(current_stage, 0)
                last_rank = stage_rank.get(last_stage, 0)
                # 判定条件: 检测阶段跃迁，捕获成功经验 (例如 compile -> correctness)
                if curr_rank > last_rank:
                    print(f"  [Auto-Learn] Detected progress: {last_stage} -> {current_stage}")
                    print(f"  [Auto-Learn] The strategy from Iter {iteration} worked. Dumping candidate...")
                    self._dump_success_sample(
                        iter_num=iteration,  # 这里的 iter 是指“当前通过验证的这一轮”
                        error_info=last_error_info,  # 上一轮的错误 (是它触发了分析)
                        analysis=prev_analysis,  # 上一轮的分析 (是它解决了错误)
                        prev_stage=last_stage,
                        curr_stage=current_stage
                    )
            # 3. 成功退出检查
            if result.valid:
                self._verbose("[Debug] SUCCESS!")
                return {"success": True, "code": self._assemble_code()}

            # 4. 失败处理
            self._verbose(f"[Debug] Failed at Stage: {current_stage}")
            # 错误打印
            print("\n" + "-" * 60)
            print(f"[DEBUG DUMP] Iteration {iteration + 1} Failed")
            save = False
            self._dump_raw_error(current_error_info, iteration, save)
            print("-" * 60 + "\n")

            # 5. [新增] 动态分发修复任务 (Smart Dispatch)
            # 传入 prev_analysis 是为了预防逻辑死循环，但分发主要依赖新一轮的分析
            # 在这里我们不直接 _classify_error，而是交给 _dispatch_fix 内部去先分析再分发
            self._dispatch_fix(current_error_info)

            # 6. 更新状态指针 (为下一轮做准备)
            last_error_info = current_error_info  # 当前错误变成下一轮的“历史错误”
            last_stage = current_stage

        return {"success": False, "code": self._assemble_code()}

    def _assemble_code(self) -> dict:
        return {
            "kernel_src": self.run_state_dict.kernel_src,
            "host_tiling_src": self.run_state_dict.tiling_src,
            "host_operator_src": self.run_state_dict.operator_src,
            "python_bind_src": self.run_state_dict.pybind_src,
        }

    def _get_full_context(self) -> dict:
        return {
            "kernel_src": self.run_state_dict.kernel_src,
            "tiling_src": self.run_state_dict.tiling_src,
            "operator_src": self.run_state_dict.operator_src,
            "pybind_src": self.run_state_dict.pybind_src,
        }

    def _classify_error_legacy(self, error_info: dict) -> str:
        """(Legacy) 基于规则的错误分类，作为 Analysis 失败时的保底"""
        stage = error_info.get("stage", "")
        error_msg = str(error_info.get("error", ""))
        if stage == "compile":
            if "kernel" in error_msg.lower() or ".cpp" in error_msg:
                return "_fix_kernel"
            elif "tiling" in error_msg.lower() or "host" in error_msg.lower():
                return "_fix_tiling"
            elif "pybind" in error_msg.lower():
                return "_fix_pybind"
        elif stage == "correctness":
            return "_fix_kernel"
        elif stage == "deploy":
            return "_fix_pybind"
        return "_fix_kernel"

    def _dispatch_fix(self, error_info: dict):
        full_context = self._get_full_context()
        history = self.run_state_dict.debug_history
        print("\n" + "=" * 20 + " [Debug] Smart Dispatching " + "=" * 20)

        if not full_context:
            print("  [Warning] full_context is Empty or None!")
        else:
            for key, content in full_context.items():
                # 检查内容是否存在，并计算长度
                length = len(content) if content else 0
                status = "Loaded" if content else "Missing/None"
                # 截取前50个字符预览 (去除换行符方便显示)
                preview = content[:50].replace('\n', '\\n') + "..." if content else "N/A"

                print(f"  - Key: {key:<20} | Size: {length:<6} chars | Status: {status} | Preview: {preview}")
        print("=" * 65 + "\n")

        # 1. 先进行分析 (Analysis First)
        # 默认取 kernel 代码进行分析 (因为大部分错误都在 Kernel)，如果是 tiling 错误，Analysis Agent 也能通过日志看出来
        src_code_for_analysis = full_context.get("kernel_src", "")

        print(f"  [Debug] Calling Analyzer to decide fix target...")
        analysis_result = self.analyzer.analyze(error_info, src_code_for_analysis)

        # 保存分析结果供 run 循环使用
        self.run_state_dict.latest_analysis = analysis_result

        # 2. 获取 LLM 决定的修复目标 (suggestion_type)
        # 期望值: "_fix_kernel", "_fix_tiling", "_fix_pybind"
        fix_method_name = analysis_result.get("suggestion_type", "")
        diagnosis = analysis_result.get("diagnosis", "Unknown")

        print(f"  [Diagnosis]: {diagnosis}")
        print(f"  [Decision]: LLM suggests calling -> {fix_method_name}")

        # 3. 动态分发
        target_method = None # 兜底设置

        # 安全检查: 方法名是否存在
        if fix_method_name and hasattr(self, fix_method_name):
            target_method = getattr(self, fix_method_name)
        else:
            print(f"  [Warn] Invalid suggestion '{fix_method_name}', falling back to legacy classification.")
            fallback_name = self._classify_error_legacy(error_info)
            target_method = getattr(self, fallback_name)

        # 4. 执行修复
        # 注意: 这里的 analysis_result 已经包含了 strategy 等信息，不需要在 _fix_kernel 里再次分析
        # 但为了保持 _fix_with_retry 的通用性，我们传入 analysis_result 让它跳过内部的 analyze 步骤 (或者复用)
        if target_method:
            target_method(error_info, full_context, history, pre_analysis=analysis_result)
        else:
            print("  [Error] No fix method available!")

        print("=" * 65 + "\n")

    # =========================================================================
    # [新增] 通用原子能力 (Atomic Capabilities)  Knowledge Dumping & Logic Safeguard
    # =========================================================================

    def _dump_success_sample(self, iter_num, error_info, analysis, prev_stage, curr_stage):
        """
        [知识库原始文件] 生成供知识库学习的样本文件 (JSON Schema 对齐 KnowledgeLearner)
        """
        if not analysis:
            return

        # 获取算子名称
        # 从 task 配置中提取 op_name，如果取不到则默认为 "unknown"
        op_name = "unknown"
        task = self.config.task
        try:
            # 尝试多种常见的属性名，防止 crash
            if hasattr(task, "data") and isinstance(task.data, dict):
                op_name = task.data.get("op_name", "unknown")
            elif hasattr(task, "_data") and isinstance(task._data, dict):
                op_name = task._data.get("op_name", "unknown")
            elif hasattr(task, "input_data") and isinstance(task.input_data, dict):
                op_name = task.input_data.get("op_name", "unknown")
        except Exception as e:
            print(f"  [Warn] Failed to extract op_name: {e}")
            op_name = "unknown"

        # [Fixed] 我们将两个关键信息都保存下来
        action_decision = analysis.get("suggestion_type", "unknown")  # e.g., _fix_kernel
        rule_category = analysis.get("error_category", "OTHER")  # e.g., FIX_LOGIC

        sample_data = {
            "meta": {
                "iteration": iter_num,
                "timestamp": "auto-generated",
                "fix_agent": action_decision  # 明确记录是哪个 Agent 修好的
            },
            "prev_stage": prev_stage,
            "curr_stage": curr_stage,
            "raw_error": error_info,
            "effective_diagnosis": analysis.get("diagnosis", "N/A"),
            "effective_strategy": analysis.get("strategy", "N/A"),

            # [关键修改] 透传两个字段
            "action_decision": action_decision,  # 方法名
            "error_category": rule_category,  # 分类标签
        }

        # 格式: success_sample{op_name}_iter{N}_{stageA}_to_{stageB}_{ts}.json
        base_dir = Path(self.config.output_path).parent / "knowledge_candidates"
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        filename = f"success_sample{op_name}_iter{iter_num}_{prev_stage}_to_{curr_stage}_{timestamp}.json"

        with open(base_dir / filename, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print(f"  [Auto-Learn] Sample saved to {filename}")

    def _inject_logic_safeguard(self, analysis_result: dict):
        """
        [安全注入] 针对逻辑错误，额外注入防御性提示，防止死循环或越界。
        直接修改 analysis_result 的 strategy 字段。
        """
        category = analysis_result.get("error_category", "")
        diag = analysis_result.get("diagnosis", "")

        if "FIX_LOGIC" in category or "Logic Deadlock" in diag:
            print("  [Auto-Safeguard] Detected Logic Issue. Injecting anti-deadlock prompts...")
            safeguard_prompt = (
                "\n\n[SYSTEM SAFETY WARNING]: "
                "You are fixing a logic error (Deadlock/Timeout). "
                "1. DO NOT create infinite loops (e.g., check loop conditions). "
                "2. DO NOT access memory out of bounds (check coreOffset + currentSize <= total). "
                "3. Use `min()` for size calculation. "
                "4. Ensure queue EnQue/DeQue are paired strictly."
            )

            # 将提示追加到策略中，LLM 在阅读策略时会看到这个警告
            analysis_result["strategy"] += safeguard_prompt

    def _record_diagnosis_to_history(self, history: list, error_info: dict, analysis_result: dict):
        """[历史记录] 将诊断结果存入 History"""
        if history is None: return
        # 如果没有 analysis (如 Pybind/Tiling 暂时没用 analyzer)，则存简略信息
        diag = analysis_result.get("diagnosis", "N/A") if analysis_result else error_info.get("error", "Unknown")
        strat = analysis_result.get("strategy", "N/A") if analysis_result else "General Fix"

        current_record = {
            "iteration": self.run_state_dict.current_iteration,
            "stage": error_info.get("stage"),
            "diagnosis": diag,
            "strategy": strat
        }
        history.append(current_record)

    def _save_debug_prompt(self, prompt: str, agent_type: str, attempt: int = 0):
        """[Prompt落盘] 保存 Prompt。支持记录重试次数 (iter_X_retry_Y.md)"""
        try:
            base_dir = Path(self.config.output_path)
            prompt_dir = base_dir / "prompt" / f"{agent_type}_prompt"
            prompt_dir.mkdir(parents=True, exist_ok=True)

            iter_num = self.run_state_dict.current_iteration + 1
            suffix = f"_retry_{attempt}" if attempt > 0 else ""
            file_name = f"{agent_type}_prompt_iter_{iter_num}{suffix}.md"
            save_path = prompt_dir / file_name

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(prompt)
        except Exception as e:
            print(f"[Warn] Failed to save prompt: {e}")

    def _apply_code_cleaning(self, code: str, analysis_result: dict = None) -> str:
        """[动态清洗] 正则去毒 + 幻觉去除"""
        new_code = code
        # [适配] 读取 error_category
        category = analysis_result.get("error_category", "") if analysis_result else ""

        # 1. 专家建议清洗
        if "REMOVE_HEADER" in category:
            print(f"  [Auto-Fix] Applying surgical regex for header removal...")
            new_code = re.sub(r'#include\s+["<]register/.*[">]', '// [Auto-Removed Poison Header]', new_code)
            std_libs = r"vector|memory|iostream|map|string|algorithm|cmath|thread|mutex|limits|functional"
            new_code = re.sub(r'#include\s+<(' + std_libs + r')>', '// [Auto-Removed Std Header]', new_code)

        # 2. 幻觉清洗 (针对 C++ 代码)
        # 注意: 对于 JSON 格式的 Tiling 返回，这个清洗可能需要跳过或特化，这里假设传入的是源码字符串
        if "{" in code and "}" in code:
            lines = new_code.split('\n')
            clean_lines = []
            for line in lines:
                line_code_part = line.split("//")[0]
                if any(u'\u4e00' <= c <= u'\u9fff' for c in line_code_part):
                    print(f"  [Auto-Clean] Removing hallucinated line: {line.strip()}")
                    continue
                clean_lines.append(line)
            new_code = "\n".join(clean_lines)
        return new_code

    def _dump_raw_error(self, error_info: dict, iteration: int, save: bool):
        # 1) 控制台打印（方便你实时看）
        print("[RAW ERROR] stage:", error_info.get("stage"))
        print("[RAW ERROR] error:\n", str(error_info.get("error", "")))
        if "details" in error_info and error_info.get("details"):
            print("[RAW ERROR] details:\n", str(error_info.get("details", "")))

        # 2) 落盘（方便后续整理知识库）
        if save:
            try:
                base_dir = Path(self.config.output_path)
                dump_dir = base_dir / "raw_error"
                dump_dir.mkdir(parents=True, exist_ok=True)

                iter_num = iteration + 1
                dump_path = dump_dir / f"iter_{iter_num}.json"
                with open(dump_path, "w", encoding="utf-8") as f:
                    json.dump(error_info, f, ensure_ascii=False, indent=2)
                print(f"[RAW ERROR] saved to: {dump_path}")
            except Exception as e:
                print(f"[Warn] Failed to dump raw error: {e}")


    def _check_circuit_breaker(self, content: Any, min_length: int, history: list) -> bool:
        """
        [差异化熔断机制]
        Args:
            content: 解析后的内容 (str 代码 或 dict/json)
            min_length: 最小长度阈值
        """
        is_valid = True
        current_len = 0

        if isinstance(content, str):
            current_len = len(content)
            if current_len < min_length:
                is_valid = False
        elif isinstance(content, dict):  # 针对 Tiling JSON
            # 简单检查 JSON 是否为空或 key 缺失
            if not content:
                is_valid = False
            else:
                # 粗略计算 value 的长度
                current_len = sum(len(str(v)) for v in content.values())
                if current_len < min_length:
                    is_valid = False

        if not is_valid:
            print(f"\n[CRITICAL WARN] Circuit Breaker Triggered! Content len={current_len} < {min_length}")
            if history:
                last_record = history[-1]
                last_record["diagnosis"] += " \n[SYSTEM ERROR]: 上一轮修复代码严重坍塌/截断。"
                last_record["strategy"] += " \n[URGENT FIX]: 上一次生成失败！务必输出完整内容！"
            return False
        return True

    # =========================================================================
    # [核心引擎] 通用重试修复逻辑 (The Engine)
    # =========================================================================
    def _fix_with_retry(
            self,
            agent_type: str,
            error_info: dict,
            context: dict,
            history: list,
            parser_func: Callable,
            min_length: int,
            use_analyzer: bool = False,
            max_retries: int = 2,
            pre_analysis: dict = None  # [新增] 接收外部传入的 analysis
    ) -> Any:

        # 1. 分析 (Analysis)
        # 如果外部传入了 pre_analysis (来自 Smart Dispatch)，直接使用，不再重复调用 Analyzer
        analysis_result = pre_analysis

        if analysis_result is None and use_analyzer:
            # 只有在没有 pre_analysis 且 use_analyzer=True 时才自己分析 (Fallback)
            print(f"  [Debug] {agent_type.capitalize()} Agent Analyzing (Fallback)...")
            code_key_map = {"kernel": "kernel_src", "tiling": "tiling_src", "pybind": "pybind_src"}
            src_code = context.get(code_key_map.get(agent_type, "kernel_src"), "")
            analysis_result = self.analyzer.analyze(error_info, src_code)
            print(f"  [Diagnosis]: {analysis_result.get('diagnosis')}")
            print(f"  [Strategy]:  {analysis_result.get('strategy')}")
            self.run_state_dict.latest_analysis = analysis_result

        if analysis_result:
            self._inject_logic_safeguard(analysis_result)

        self._record_diagnosis_to_history(history, error_info, analysis_result)

        # 2. 原地重试循环
        for attempt in range(max_retries):
            prompt = self.config.interface.get_debug_prompt(
                agent_type, context, error_info, history=history, analysis=analysis_result
            )
            if attempt > 0:
                prompt += "\n\n[SYSTEM NOTICE]: Previous attempt failed. OUTPUT FULL CODE."

            self._save_debug_prompt(prompt, agent_type, attempt)

            response, _ = self.config.running_llm.get_response(prompt)
            parsed_content = parser_func(response)

            if isinstance(parsed_content, str):
                parsed_content = self._apply_code_cleaning(parsed_content, analysis_result)

            if self._check_circuit_breaker(parsed_content, min_length, history):
                print(f"  [Retry Success] {agent_type} fix succeeded at attempt {attempt}.")
                return parsed_content

        print(f"  [Retry Fail] {agent_type} fix failed after {max_retries} retries.")
        return None

    # =========================================================================
    # [具体实现] 各个 Agent 的入口
    # =========================================================================

    def _fix_kernel(self, error_info: dict, context: dict, history: list = None, pre_analysis: dict = None):
        """修复 Kernel 代码"""
        new_code = self._fix_with_retry(
            agent_type="kernel",
            error_info=error_info,
            context=context,
            history=history,
            parser_func=parse_code,
            min_length=600,
            use_analyzer=False,  # Analysis 已经在 dispatch 阶段完成了
            max_retries=3,
            pre_analysis=pre_analysis
        )
        if new_code:
            self.run_state_dict.kernel_src = new_code

    def _fix_tiling(self, error_info: dict, context: dict, history: list = None, pre_analysis: dict = None):
        """修复 Tiling 代码"""
        result_dict = self._fix_with_retry(
            agent_type="tiling",
            error_info=error_info,
            context=context,
            history=history,
            parser_func=parse_json,
            min_length=100,
            use_analyzer=False,
            max_retries=2,
            pre_analysis=pre_analysis
        )
        if result_dict:
            if result_dict.get("host_tiling_src"):
                self.run_state_dict.tiling_src = result_dict.get("host_tiling_src")
            if result_dict.get("host_operator_src"):
                self.run_state_dict.operator_src = result_dict.get("host_operator_src")

    def _fix_pybind(self, error_info: dict, context: dict, history: list = None, pre_analysis: dict = None):
        """修复 Pybind 代码"""
        new_code = self._fix_with_retry(
            agent_type="pybind",
            error_info=error_info,
            context=context,
            history=history,
            parser_func=parse_code,
            min_length=50,
            use_analyzer=False,
            max_retries=3,
            pre_analysis=pre_analysis
        )
        if new_code:
            self.run_state_dict.pybind_src = new_code


