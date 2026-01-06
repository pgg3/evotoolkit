# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Debug 专员 Prompt 默认实现"""

from typing import Any


class DebugPromptMixin:
    """Debug 专员 Prompt"""

    def get_debug_prompt(self, agent_type: str, code: Any, error: dict) -> str:
        """生成调试修复 Prompt"""
        if agent_type == "kernel":
            return self._get_kernel_debug_prompt(code, error)
        elif agent_type == "tiling":
            return self._get_tiling_debug_prompt(code, error)
        elif agent_type == "pybind":
            return self._get_pybind_debug_prompt(code, error)
        else:
            return self._get_kernel_debug_prompt(code, error)

    def _get_kernel_debug_prompt(self, code: str, error: dict) -> str:
        """Kernel 调试 Prompt"""
        return f"""
你是昇腾 Ascend C Kernel 调试专家。当前 kernel 代码执行出错，请修复。

## 当前代码
```cpp
{code}
```

## 错误信息
- 阶段: {error.get('stage', 'unknown')}
- 错误: {error.get('error', 'unknown')}
- 详情: {error.get('details', '')}

## 常见错误及修复
1. **编译错误**: 检查 API 使用是否正确，参数类型是否匹配
2. **正确性错误**: 检查计算逻辑，数据搬运是否正确
3. **内存错误**: 检查 tensor 大小，避免越界访问

## 要求
- 分析错误原因
- 修复代码
- 返回完整的修复后 kernel 代码（用 ```cpp 包裹）
"""

    def _get_tiling_debug_prompt(self, code: dict, error: dict) -> str:
        """Tiling 调试 Prompt"""
        tiling_src = code.get('tiling', '') if isinstance(code, dict) else code
        operator_src = code.get('operator', '') if isinstance(code, dict) else ''

        return f"""
你是昇腾 Ascend C Host 端调试专家。当前 tiling 代码出错，请修复。

## 当前 tiling.h
```cpp
{tiling_src}
```

## 当前 op_host.cpp
```cpp
{operator_src}
```

## 错误信息
- 阶段: {error.get('stage', 'unknown')}
- 错误: {error.get('error', 'unknown')}

## 常见错误
1. **编译错误**: 检查头文件，命名空间，宏定义
2. **Tiling 计算错误**: 检查参数计算逻辑
3. **InferShape 错误**: 检查输出 shape 推断

## 返回 JSON 格式
```json
{{
  "host_tiling_src": "修复后的 tiling.h",
  "host_operator_src": "修复后的 op_host.cpp"
}}
```
"""

    def _get_pybind_debug_prompt(self, code: str, error: dict) -> str:
        """Pybind 调试 Prompt"""
        return f"""
你是 PyTorch C++ 扩展调试专家。当前 pybind 代码出错，请修复。

## 当前代码
```cpp
{code}
```

## 错误信息
- 阶段: {error.get('stage', 'unknown')}
- 错误: {error.get('error', 'unknown')}

## 常见错误
1. **编译错误**: 检查头文件引用，类型转换
2. **部署错误**: 检查 NPU kernel 调用方式
3. **Shape 错误**: 检查输出 tensor shape 计算

## 要求
返回完整的修复后 pybind 代码（用 ```cpp 包裹）
"""
