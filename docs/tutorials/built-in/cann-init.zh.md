# CANN 算子初始化 (Ascend NPU) 教程

学习如何使用 LLM 驱动的进化算法为华为昇腾 NPU 生成和优化 Ascend C 算子内核代码。

!!! warning "硬件要求"
    此任务需要 **华为昇腾 NPU 硬件** 以及已安装的 **CANN 工具包**。无法在标准 CPU/GPU 环境中运行。

!!! tip "完整示例代码"
    查看示例目录中的脚本：

    - [:material-folder: examples/cann_init/](https://github.com/pgg3/evotoolkit/blob/master/examples/cann_init/) - Agent 和 Evaluator 脚本
    - [:material-file-document: README.md](https://github.com/pgg3/evotoolkit/blob/master/examples/cann_init/README.md) - 使用指南

---

## 概述

本教程演示：

- 为 Ascend C 算子生成创建 CANN Init 任务
- 使用 LLM 驱动的进化算法生成优化的 Ascend C 内核代码
- 理解算子签名和模板系统
- 在昇腾 NPU 硬件上评估算子正确性和性能

EvoToolkit 将 Ascend C 算子生成视为优化问题：给定 Python 参考实现，进化出正确且高效的 Ascend C 内核代码。

---

## 前置要求

### 硬件

- 华为昇腾 NPU（已在 Ascend910B2 上测试）

### 软件

```bash
# 从华为安装 CANN 工具包（参见官方文档）
# https://www.hiascend.com/software/cann

# 安装支持 CANN 的 EvoToolkit
pip install evotoolkit[cann_init]
```

这将安装：

- `pybind11` - 用于 Python/C++ 绑定生成
- 其他 CANN 相关依赖

---

## 理解 CANN Init 任务

### 任务生成什么？

该任务进化 **Ascend C 内核代码**（面向昇腾 NPU 的 C++）。给定：

1. **算子名称**（如 `"relu"`、`"layer_norm"`）
2. **Python 参考实现**（正确但未优化）

LLM 使用 Ascend C API（数据搬移、计算、Tiling 等）生成实现相同操作的 Ascend C 内核代码。

### 模板系统

EvoToolkit 自动从模板生成周围代码（Host 代码、Tiling 配置、Python 绑定）。LLM 只需提供 **内核实现**。

### 评估

每个生成的内核：

1. 使用 CANN 工具包 **编译**
2. 对照 Python 参考 **验证正确性**
3. **性能基准测试**（吞吐量、延迟）

---

## 快速开始

### 步骤 1：定义 Python 参考实现

```python
PYTHON_REFERENCE = '''
def relu(x):
    """ReLU 激活函数: max(0, x)"""
    import numpy as np
    return np.maximum(0, x)
'''
```

### 步骤 2：创建任务

```python
from evotoolkit.task.cann_init import CANNInitTask

task = CANNInitTask(
    data={
        "op_name": "relu",
        "python_reference": PYTHON_REFERENCE,
        "npu_type": "Ascend910B2",   # 你的 NPU 型号
        "cann_version": "8.0",        # 你的 CANN 版本
    },
    project_path="/tmp/cann_projects",  # 编译产物的目录
)

print(f"算子: {task.task_info['op_name']}")
print(f"NPU 类型: {task.task_info['npu_type']}")
```

### 步骤 3：评估内核代码

```python
kernel_code = '''
// Ascend C ReLU 内核实现
class KernelRelu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {
        // ... 初始化代码 ...
    }

    __aicore__ inline void Process() {
        // ... 计算代码 ...
    }
};
'''

result = task.evaluate_code(kernel_code)

if result.valid:
    print(f"得分: {result.score:.4f}")
    print(f"正确性: {result.additional_info.get('correctness')}")
    print(f"性能: {result.additional_info.get('performance')}")
else:
    print(f"错误: {result.additional_info.get('error')}")
```

### 步骤 4：运行进化

```python
import evotoolkit
from evotoolkit.task.cann_init.method_interface import CANNIniterInterface
from evotoolkit.tools.llm import HttpsApi

# 创建接口
interface = CANNIniterInterface(task)

# 配置 LLM
llm_api = HttpsApi(
    api_url="api.openai.com",
    key="your-api-key-here",
    model="gpt-4o"
)

# 运行进化
result = evotoolkit.solve(
    interface=interface,
    output_path='./cann_results',
    running_llm=llm_api,
    max_generations=5,
    pop_size=3,
)

print(f"发现的最优内核：")
print(result.sol_string)
print(f"得分: {result.evaluation_res.score:.4f}")
```

---

## `CANNInitTask` API

```python
class CANNInitTask(BaseTask):
    def __init__(
        self,
        data: dict,            # 任务配置（见下文）
        project_path: str | None = None,  # 编译产物的默认目录
        fake_mode: bool = False,          # 跳过评估（用于测试）
    )
```

**`data` 字典键：**

| 键 | 必需 | 描述 |
|----|------|------|
| `op_name` | 是 | 算子名称（如 `"relu"`、`"layer_norm"`）|
| `python_reference` | 是 | Python 参考实现（字符串）|
| `npu_type` | 否 | NPU 型号（默认：`"Ascend910B2"`）|
| `cann_version` | 否 | CANN 版本（默认：`"8.0"`）|

**主要方法：**

| 方法 | 描述 |
|------|------|
| `evaluate_code(kernel_src)` | 评估内核代码字符串，返回 `EvaluationResult` |
| `evaluate_solution(solution)` | 支持 `other_info` 的高级接口 |

**通过 `other_info` 的高级 `evaluate_solution` 选项：**

```python
from evotoolkit.core import Solution

# 仅编译模式（用于并行工作流）
solution = Solution(
    sol_string=kernel_src,
    other_info={
        "project_path": "/compile/sol_001",
        "compile_only": True,
        "save_compile_to": "/compile/sol_001",
    }
)
compile_result = task.evaluate_solution(solution)

# 加载预编译产物进行测试
solution = Solution(
    sol_string="",
    other_info={
        "load_from": "/compile/sol_001",
    }
)
test_result = task.evaluate_solution(solution)
```

---

## 支持的算子类型

CANN Init 任务可应用于任何可用 Python 表达的算子：

| 类别 | 示例 |
|------|------|
| 逐元素 | ReLU、Sigmoid、GELU、Add、Multiply |
| 规约 | Softmax、LayerNorm、Sum、Mean |
| 矩阵乘 | GEMM、Attention (SDPA) |
| 自定义 | 任何有 Python 参考实现的算子 |

---

## 获得更好结果的技巧

1. **提供清晰的 Python 参考实现** — LLM 依此理解算子语义
2. **从简单算子开始**（逐元素），再处理复杂算子（矩阵乘）
3. **使用 `fake_mode=True`** 在开发期间无需硬件测试流程
4. **参考 CANN 文档** 了解可用的 Ascend C API 和 Tiling 模式

---

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| 编译错误 | 检查 CANN 环境变量和工具包安装 |
| 正确性失败 | 检查 Python 参考实现的边界情况 |
| 性能低于基线 | LLM 可能需要关于 Ascend C Tiling 的领域知识 |

---

## 下一步

- [自定义进化方法](../customization/customizing-evolution.zh.md) — 在提示中添加领域知识
- [高级用法](../advanced-overview.zh.md) — 并行编译和高级工作流
- [API 参考](../../api/index.md) — 完整 API 文档
