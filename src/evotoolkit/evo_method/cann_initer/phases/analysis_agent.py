# Copyright (c) 2025 Sun Yansong
# Licensed under the MIT License
# evotoolkit/evo_method/cann_initer/phases/analysis_agent.py
'''
当前版本v0.5
实际的优化过程是同时动态进行的，没有严格的先后顺序）。
所以要根据实际的analysis_agent.py中得到的分析结果，动态的调用fix逻辑
因此 改变analysis_agent.py中的逻辑，
严格限制返回的suggestion_type 必须是_fix_kernel还是_fix_tiling还是_fix_pybind方法名称，
然后改变debug_loop.py中的分发逻辑，识别返回的suggestion_type，去动态调用方法fix方法，
把正则选择的选择权力交给LLM，实现更加智能，高效的debug
'''
import json
from pathlib import Path
from ..parsers import parse_json


class ErrorAnalyzer:
    def __init__(self, llm_config, knowledge_base_path="knowledge_base.json"):
        self.llm = llm_config
        self.kb_path = Path(knowledge_base_path)
        self.default_rules = "目前没有知识库，请根据报错自行分析。"  # 保底规则

    def _load_knowledge_base(self) -> str:
        """加载外部知识库，如果不存在则使用默认值"""
        if self.kb_path.exists():
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    rules = json.load(f)
                    kb_str = ""
                    for idx, rule in enumerate(rules):
                        kb_str += f"{idx + 1}. **【{rule['title']}】**\n"
                        kb_str += f"   - **特征**: {rule['feature']}\n"
                        kb_str += f"   - **原因**: {rule['reason']}\n"
                        kb_str += f"   - **对策**: {rule['fix']}\n"
                        # [Fixed] 这里的名称改为 Category Tag，避免与 Output 中的 suggestion_type (Action) 混淆
                        kb_str += f"   - **Category Tag**: \"{rule['type']}\"\n\n"
                    return kb_str
            except Exception as e:
                print(f"[Analyzer Warn] Load KB failed: {e}, using default.")
        return self.default_rules

    def analyze(self, error_info: dict, code_snippet: str) -> dict:
        raw_error = str(error_info.get("error", ""))
        if len(raw_error) > 3000:
            raw_error = raw_error[:1500] + "\n...[省略中间日志]...\n" + raw_error[-1500:]
        # 动态加载当前的知识库
        knowledge_base_content = self._load_knowledge_base()

        # ... (其余代码保持不变，将原 prompt 中的 [专家知识库...] 部分替换为 {knowledge_base_content}) ...
        prompt = f"""
你是一名资深的 C++ 编译器与 Ascend C 算子开发专家。请参考以下报错信息，同时基于[专家知识库]分析报错。

## [专家知识库 - 常见错误特征 & 修复方案]
(这些经验源自大量实战 Debug 总结，优先级极高)
{knowledge_base_content}
## [重要] Environment: 
CANN 8.1.rc1, Ascend 910B. Use strictly modern 'Ascend C' APIs (namespace AscendC). Avoid deprecated TBE or Tik syntax.

## [报错现场]
阶段: {error_info.get('stage')}
日志摘要: 
{raw_error}

## [相关代码]
{code_snippet} 

## [任务]
请像一名资深 Ascend C 算子开发专家一样精准分析问题，并给出 JSON 格式的debug诊断书。
1. **diagnosis**: 发生了什么错误？如果有"redefinition"，是指哪个结构体？如果有"undeclared"，是指哪个变量？如果是"illegal character"，是指什么字符？(如果是知识库中的错误，请明确指出是哪一类)(限200字)
2. **strategy**: 下一步具体怎么改？必须给出代码级别的指导，用**自然语言**描述。
3. **suggestion_type (Core Decision)**: 
   **必须**从以下三个方法名中选择一个，决定调用哪个 Agent 进行修复：
   - `_fix_kernel`: 绝大多数编译错误（kernel_operator.h相关）、逻辑错误、Kernel内Tiling参数使用错误。
   - `_fix_tiling`: 仅当 Host 端 Tiling 计算逻辑出错、Host 端头文件定义冲突、InferShape 错误时选择。
   - `_fix_pybind`: 仅当 Python 调用接口不匹配、Pybind 编译错误时选择。
4. **error_category**: 
   从知识库中提取的错误分类 Tag，用于辅助代码清洗和知识库归档。
   - 可选值: REMOVE_HEADER | USE_RAW_POINTER | CLAMP_TILE_SIZE | FIX_POINTER_CAST | FIX_SYNTAX | FIX_LOGIC | FIX_API | OTHER (可以自行定义，但不超过10个单词，必须保持相同格式）
   - 如果命中了知识库规则，请填写规则中的 "类型Tag"。

## [输出格式]
**Strictly return JSON only**:
{{
    "stage": "{error_info.get('stage')}",
    "diagnosis": "...",
    "strategy": "...",
    "suggestion_type": "_fix_kernel" | "_fix_tiling" | "_fix_pybind",
    "error_category": "REMOVE_HEADER" | "OTHER" ...
}}
"""
        response, _ = self.llm.get_response(prompt)
        try:
            return parse_json(response)
        except:
            return {
                "stage": "unknown",
                "diagnosis": "LLM 解析失败，请查看原始日志。",
                "strategy": "根据上下文和报错信息检查改正。",
                "suggestion_type": "_fix_kernel",  # Fallback
                "error_category": "OTHER"
            }