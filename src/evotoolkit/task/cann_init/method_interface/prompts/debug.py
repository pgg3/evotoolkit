# Copyright (c) 2025 Sun Yansong
# Licensed under the MIT License

"""Debug 专员 Prompt 默认实现"""

from typing import Any, List, Union


class DebugPromptMixin:
    """Debug 专员 Prompt (支持多文件上下文+ 历史记忆 + 智能降噪)"""

    def _clean_error_log(self, log: str) -> str:
        """
        清洗 Ascend NPU 报错日志，移除无意义的寄存器 Dump，保留关键错误信息。
        """
        if not log: return ""

        lines = log.split('\n')
        cleaned_lines = []

        # 关键词黑名单：出现这些词的行通常是机器码垃圾
        blacklist = [
            "vec error info:", "mte error info:", "ifu error info:",
            "ccu error info:", "cube error info:", "biu error info:",
            "aic error mask:", "para base:", "pc start:",
            "dump info:", "The extend info:", "fixp_error",
            "t-trace-id", "TraceBack (most recent call last):"
        ]

        for line in lines:
            # 1. 过滤掉包含黑名单关键词的行
            if any(bad_word in line for bad_word in blacklist):
                continue

            # 2. 过滤纯地址行 (例如 0x12c0c0014000)
            if line.strip().startswith("0x") and len(line.strip()) > 10:
                continue

            # 3. 保留关键错误行
            # 507035 是 Vector Core Exception
            if "Error" in line or "error" in line or "RuntimeError" in line or "code is" in line:
                cleaned_lines.append(line)
            # 保留简短的描述行 (防止把必要的 C++ 报错行删掉)
            elif len(line) < 150:
                cleaned_lines.append(line)

            # 再次截断，防止过长
        if len(cleaned_lines) > 40:
            return "\n".join(cleaned_lines[:15] + ["\n... [中间大量寄存器日志已过滤] ...\n"] + cleaned_lines[-15:])

        return "\n".join(cleaned_lines)


    def _format_history(self, history: List[dict]) -> str:
        if not history:
            return "这是第一次修复尝试 (基于 Iter 1 报错)"
        summary = ""
        # 只取最近 2 次的分析结果
        for idx, record in enumerate(history[-2:]):
            iter_num = record.get('iteration', '-1')
            diag = record.get('diagnosis', 'N/A')
            strat = record.get('strategy', 'N/A')

            summary += f"\n--- [失败尝试 #{iter_num + 1}] ---\n"
            summary += f"诊断: {diag}\n"
            summary += f"策略: {strat}\n"
        return summary


    def get_debug_prompt(self, agent_type: str, code: Any, error: dict, history: list = None, analysis: dict = None) -> str:
        """
        生成调试修复 Prompt

        Args:
            agent_type: 'kernel', 'tiling', 'pybind'
            code: 现在的 code 是一个包含所有源码的字典 full_context
                  {
                      'kernel_src': ...,
                      'tiling_src': ...,
                      'operator_src': ...,
                      'pybind_src': ...
                  }
            error: 错误信息字典
        """
        if history is None:
            history = []
        # 在生成 Prompt 前智能清洗与重组
        raw_error = str(error.get('error', ''))
        raw_details = str(error.get('details', ''))

        # 1. 如果 details 为空，但 error 很长(包含堆栈)，说明 Task 把日志全塞进 error 了
        #    这时候我们需要把 error 的内容转移到 details，并清洗
        if not raw_details and len(raw_error) > 300:
            raw_details = raw_error
            # 尝试提取第一行作为摘要，如果第一行太乱，就用固定文案
            first_line = raw_error.split('\n')[0]
            if len(first_line) < 200:
                raw_error = first_line
            else:
                raw_error = "Runtime Error (See Details for logs)"

        # 2. 对两个字段都进行清洗 (防止 Hex Dump 污染)
        clean_error_msg = self._clean_error_log(raw_error)
        clean_details = self._clean_error_log(raw_details)

        # 3. 重新打包 error 字典供后续使用
        #    这样 Prompt 里不会出现几千行的 NPU 寄存器信息
        clean_error_dict = error.copy()
        clean_error_dict['error'] = clean_error_msg
        clean_error_dict['details'] = clean_details

        if agent_type == "kernel":
            return self._get_kernel_debug_prompt(code, clean_error_dict, history, analysis)
        elif agent_type == "tiling":
            return self._get_tiling_debug_prompt(code, clean_error_dict, history, analysis)
        elif agent_type == "pybind":
            return self._get_pybind_debug_prompt(code, clean_error_dict, history, analysis)
        else:
            return self._get_kernel_debug_prompt(code, clean_error_dict, history, analysis)

    def _get_kernel_debug_prompt(self, context: dict, error: dict, history: list, analysis: dict = None) -> str:
        """Kernel 调试 Prompt - 能够看到 Tiling 和 Host 定义"""
        print("这是新的kernel prompt")
        kernel_src = context.get('kernel_src', '')
        tiling_src = context.get('tiling_src', '')
        pybind_src = context.get('pybind_src', '')
        operator_src = context.get('operator_src', '')
        stage = error.get('stage', 'unknown')

        history_str = self._format_history(history)
        current_strategy = "无特殊策略，请根据错误信息自行分析"
        current_diagnosis = "未知错误"

        # [Fixed] 增强信息展示：同时显示 Category 和 Action
        expert_info = ""
        if analysis:
            current_strategy = analysis.get("strategy", "")
            diagnosis_text = analysis.get("diagnosis", "")
            cat = analysis.get("error_category", "N/A")
            act = analysis.get("suggestion_type", "N/A")
            expert_info = f"""
        **专家诊断认为核心错误是**: {diagnosis_text}
        **错误分类 (Tag)**: {cat}
        **建议修复动作**: {act}
        **你必须执行的修复策略**: 
        >>> {current_strategy} <<<
        """
        else:
            expert_info = "**专家分析**: (未启用或分析失败)"

        error_summary = str(error.get('details', ''))[:500] + "\n...(详细日志已由专家分析，见上文**当前策略**部分)..."

        return f"""
你是昇腾 Ascend C Kernel 调试专家。当前 kernel 代码编译或运行出错，请结合全局上下文修复它。**CRITICAL**: You MUST return the FULL content of the file, not just the modified parts. The system will reject incomplete code.
## [重要] Environment: 
CANN 8.1.rc1, Ascend 910B. Use strictly modern 'Ascend C' APIs (namespace AscendC). Avoid deprecated TBE or Tik syntax.

## [重要] 历史失败记录
{history_str}

## [当前最高优先级指令] 专家分析 (Expert Analysis)
{expert_info}
(注意：如果上述专家分析与下方的"强制规则"冲突，请以**专家分析**为准！例如，如果专家分析说"删除结构体"，请务必删除，忽略强制规则中的"必须定义"。)

## 错误信息
请仔细阅读以下的编译器报错/运行时报错，定位具体的错误行号：
- **阶段**: {stage}
- **错误**: {error_summary}

## [参考] 全局上下文 (Context)
为了确保 Kernel 正确，你需要参考以下定义:
**1. Tiling 定义 (tiling.h) **
**约束**: Kernel 内部 **必须** 重新定义一个与此完全一致的结构体 (成员类型、顺序、属性 packed 必须一模一样)
```cpp
{tiling_src}
```
**2. Host 端逻辑 (op_host.cpp) **
**约束**: 检查 Tiling 计算逻辑，确认 Host 传递的参数数量和类型。注意: DT_FLOAT 对应 float，DT_FLOAT16 对应 half
```cpp
{operator_src}
```
**3. pybind 接口 (pybind_src.cpp) **
**约束**: 检查 EXEC_NPU_CMD(aclnn..., ...) 调用的算子名称。Kernel 的入口函数名必须与系统生成的调用名匹配
```cpp
{pybind_src}
```

## [待修复] 当前 Kernel 代码 (op_kernel.cpp)
```cpp
{kernel_src}
```

## 强制修复规则 (Strict Rules)
0. **[头文件洁癖 (关键)]**: 
    Kernel 文件 (`op_kernel.cpp`) **只能** 包含 `#include "kernel_operator.h"`。
    **严禁** 包含 `register/...`、`tiling.h` 或任何 C++ 标准库头文件（如 <vector>, <cmath>）。
    如果需要 Tiling 参数，请在 Kernel 内 **手写定义结构体** 或 **使用 int32_t* 指针强转**。
1. **[CANN 8.1 规范]**: 
    严禁使用过时的 TIK 语法，代码中不能出现中文字符（注释除外）。
2. **类型匹配**: 
    严查 GlobalTensor<T> 的类型 T 是否与 Host 端的 DataType 一致, 如果 Host 是 DT_FLOAT，Kernel 必须用 float, 如果 Host 是 DT_FLOAT16，Kernel 必须用 half
3. **矢量编程**: 
    Ascend C 是矢量编程架构。严禁使用标量 for 循环配合GetValue/SetValue逐个处理 tensor 元素！必须使用向量指令：Add(...), Relu(...), Mul(...), DataCopy(...) 等
4. **完整性**: 
    请输出修复后的完整 op_kernel.cpp 代码，不要省略任何 includes 或辅助函数。**严禁**只返回修改的片段或函数。**严禁**省略未修改的部分
5. **编译错误**: 
    检查 API 使用是否正确，参数类型是否匹配
6. **正确性错误**: 
    检查计算逻辑，数据搬运是否正确
7. **内存错误**: 
    检查 tensor 大小，避免越界访问

## 要求
- **必须**根据**当前策略**返回修复后**完整的** op_kernel.cpp 源代码（用 ```cpp 包裹）
- 如果你返回的代码少于 50 行，将被视为失败
"""

    def _get_tiling_debug_prompt(self, context: dict, error: dict, history: list, analysis: dict = None) -> str:
        """Tiling 调试 Prompt- 能够参考 Kernel 的使用方式"""
        print("这是新的tiling prompt")
        kernel_src = context.get('kernel_src', '')
        tiling_src = context.get('tiling_src', '')
        pybind_src = context.get('pybind_src', '')
        operator_src = context.get('operator_src', '')
        stage = error.get('stage', 'unknown')

        history_str = self._format_history(history)
        current_strategy = "无特殊策略，请根据错误信息自行分析"
        current_diagnosis = "未知错误"

        # [Fixed] 增强信息展示：同时显示 Category 和 Action
        expert_info = ""
        if analysis:
            current_strategy = analysis.get("strategy", "")
            diagnosis_text = analysis.get("diagnosis", "")
            cat = analysis.get("error_category", "N/A")
            act = analysis.get("suggestion_type", "N/A")
            expert_info = f"""
                **专家诊断认为核心错误是**: {diagnosis_text}
                **错误分类 (Tag)**: {cat}
                **建议修复动作**: {act}
                **你必须执行的修复策略**: 
                >>> {current_strategy} <<<
                """
        else:
            expert_info = "**专家分析**: (未启用或分析失败)"

        error_summary = str(error.get('details', ''))[:500] + "\n...(详细日志已由专家分析，见上文**当前策略**部分)..."

        return f"""
你是昇腾 Ascend C Kernel 调试专家。当前 tiling 代码编译或运行出错，请结合全局上下文修复它。**CRITICAL**: You MUST return the FULL content of the file, not just the modified parts. The system will reject incomplete code.
## [重要] Environment: 
CANN 8.1.rc1, Ascend 910B. Use strictly modern 'Ascend C' APIs (namespace AscendC). Avoid deprecated TBE or Tik syntax.

## [重要] 历史失败记录
{history_str}

## [当前最高优先级指令] 专家分析 (Expert Analysis)
{expert_info}
(注意：如果上述专家分析与下方的"强制规则"冲突，请以**专家分析**为准！例如，如果专家分析说"删除结构体"，请务必删除，忽略强制规则中的"必须定义"。)

## 错误信息
请仔细阅读以下的编译器报错/运行时报错，定位具体的错误行号：
- **阶段**: {stage}
- **错误**: {error_summary}

## [参考] 全局上下文 (Context)
为了确保 **Tiling 定义 (tiling.h)** 和 **Host 端逻辑 (op_host.cpp)**正确，你需要参考以下定义:
**1. 当前 Kernel 代码 (op_kernel.cpp)**
```cpp
{kernel_src}
```

**2. 当前 pybind 接口 (pybind_src.cpp) **
```cpp
{pybind_src}
```

## [待修复]  Host 端代码
**1. op_host.cpp**
```cpp
{operator_src}
```
**2. tiling.h**
```cpp
{tiling_src}
```

## [要求] 返回 JSON 格式
```json
{{
  "host_tiling_src": "修复后的 tiling.h 完整代码",
  "host_operator_src": "修复后的 op_host.cpp 完整代码"
}}
```
"""

    def _get_pybind_debug_prompt(self, context: dict, error: dict, history: list, analysis: dict = None) -> str:
        """Pybind 调试 Prompt- 能够参考 Kernel 和 Tiling的使用方式"""
        print("这是新的pybind prompt")
        kernel_src = context.get('kernel_src', '')
        tiling_src = context.get('tiling_src', '')
        pybind_src = context.get('pybind_src', '')
        operator_src = context.get('operator_src', '')
        stage = error.get('stage', 'unknown')

        history_str = self._format_history(history)
        current_strategy = "无特殊策略，请根据错误信息自行分析"
        current_diagnosis = "未知错误"

        # [Fixed] 增强信息展示：同时显示 Category 和 Action
        expert_info = ""
        if analysis:
            current_strategy = analysis.get("strategy", "")
            diagnosis_text = analysis.get("diagnosis", "")
            cat = analysis.get("error_category", "N/A")
            act = analysis.get("suggestion_type", "N/A")
            expert_info = f"""
                **专家诊断认为核心错误是**: {diagnosis_text}
                **错误分类 (Tag)**: {cat}
                **建议修复动作**: {act}
                **你必须执行的修复策略**: 
                >>> {current_strategy} <<<
                """
        else:
            expert_info = "**专家分析**: (未启用或分析失败)"

        error_summary = str(error.get('details', ''))[:500] + "\n...(详细日志已由专家分析，见上文**当前策略**部分)..."

        return f"""
你是昇腾 Ascend C Kernel 调试专家。当前 pybind 代码编译或运行出错，请结合全局上下文修复它。**CRITICAL**: You MUST return the FULL content of the file, not just the modified parts. The system will reject incomplete code.
## [重要] Environment: 
CANN 8.1.rc1, Ascend 910B. Use strictly modern 'Ascend C' APIs (namespace AscendC). Avoid deprecated TBE or Tik syntax.

## [重要] 历史失败记录
{history_str}

## [当前最高优先级指令] 专家分析 (Expert Analysis)
{expert_info}
(注意：如果上述专家分析与下方的"强制规则"冲突，请以**专家分析**为准！例如，如果专家分析说"删除结构体"，请务必删除，忽略强制规则中的"必须定义"。)

## 错误信息
请仔细阅读以下的编译器报错/运行时报错，定位具体的错误行号：
- **阶段**: {stage}
- **错误**: {error_summary}

## [参考] 全局上下文 (Context)
为了确保 **Tiling 定义 (tiling.h)** 和 **Host 端逻辑 (op_host.cpp)**正确，你需要参考以下定义:
**1. 当前 Kernel 代码 (op_kernel.cpp)**
```cpp
{kernel_src}
```

**2. Tiling 定义 (tiling.h) **
**约束**: Kernel 内部 **必须** 重新定义一个与此完全一致的结构体 (成员类型、顺序、属性 packed 必须一模一样)
```cpp
{tiling_src}
```
**3. Host 端逻辑 (op_host.cpp) **
**约束**: 检查 Tiling 计算逻辑，确认 Host 传递的参数数量和类型。注意: DT_FLOAT 对应 float，DT_FLOAT16 对应 half
```cpp
{operator_src}
```

## [待修复] 当前 pybind 接口 (pybind_src.cpp)
```cpp
{pybind_src}
```

## 要求
返回完整的修复后 pybind 代码（用 ```cpp 包裹）
"""
