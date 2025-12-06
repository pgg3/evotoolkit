# CANN Init Task 评估流程指南

## 概述

`CANNInitTask` 用于评估 Ascend C 算子代码。与 `CudaTask` 类似，LLM 只需生成 kernel 代码，其他组件（host、tiling、binding）由模板自动生成。

---

## 1. 环境准备

### 1.1 必需环境

```bash
# Ascend NPU 环境
- Ascend 910B NPU
- CANN 8.0+
- torch_npu

# Python 环境
- Python 3.10+
- PyTorch 2.0+
```

### 1.2 安装 evotoolkit

```bash
cd /root/Huawei_CANN/evotoolkit
pip install -e .
```


> **Note:** Python 绑定编译所需的 `CppExtension` 目录（包含 `build_and_run.sh`、`setup.py`）已内置于 evotoolkit，评估时会自动创建，无需手动准备。

---

## 2. 文件结构

```
evotoolkit/src/evotoolkit/task/cann_init/
├── __init__.py                 # 模块导出
├── cann_init_task.py           # 主 Task 类
├── evaluator.py                # 评估器接口
├── templates.py                # 模板生成器
├── signature_parser.py         # Python 签名解析
├── pybind_templates/           # Python 绑定内置模板
│   └── __init__.py             # build_and_run.sh, setup.py 模板
└── backend/                    # 评估后端（复用 MultiKernelBench）
    ├── __init__.py
    ├── ascend_compile.py       # 编译流程
    ├── correctness.py          # 正确性验证
    └── performance.py          # 性能测试
```

---

## 3. 评估流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        完整评估流程                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  输入                                                                    │
│  ├── op_name: "add"                                                     │
│  ├── python_reference: Python 参考实现                                  │
│  └── kernel_src: Ascend C kernel 代码 (LLM 生成)                        │
│                                                                          │
│  Step 1: 签名解析 (OperatorSignatureParser)                             │
│  ├── 解析 python_reference                                              │
│  └── 提取: inputs, outputs, dtypes                                      │
│                                                                          │
│  Step 2: 模板生成 (AscendCTemplateGenerator)                            │
│  ├── 基于 signature 生成 6 个代码组件:                                  │
│  │   ├── project_json_src   ← 自动生成                                  │
│  │   ├── host_tiling_src    ← 自动生成                                  │
│  │   ├── host_operator_src  ← 自动生成                                  │
│  │   ├── kernel_src         ← LLM 提供                                  │
│  │   ├── python_bind_src    ← 自动生成                                  │
│  │   └── model_src          ← 自动生成                                  │
│  └── 输出: full_code dict                                               │
│                                                                          │
│  Step 3: 编译部署 (ascend_compile)                                      │
│  ├── msopgen gen -i xxx.json -c Ascend910B -lan cpp -out XxxCustom     │
│  ├── 写入源文件到 op_host/, op_kernel/                                  │
│  ├── 自动创建 CppExtension/ (内置模板)                                  │
│  ├── ./build.sh                                                         │
│  ├── ./custom_opp_ubuntu_aarch64.run                                    │
│  ├── bash build_and_run.sh (Python 绑定)                                │
│  └── 设置环境变量 ASCEND_CUSTOM_OPP_PATH                                │
│                                                                          │
│  Step 4: 正确性验证 (execute_correctness_check)                         │
│  ├── 执行 python_reference 获取 Model                                  │
│  ├── 加载 model_src 获取 ModelNew                                       │
│  ├── 对比 Model(*inputs) vs ModelNew(*inputs)                          │
│  └── 验证 shape 和 values (atol=1e-2, rtol=1e-2)                       │
│                                                                          │
│  Step 5: 性能测试 (measure_performance)                                 │
│  ├── Warmup: 3 次                                                       │
│  ├── 测量: 100 次                                                       │
│  └── 返回: mean, std, min, max                                          │
│                                                                          │
│  输出                                                                    │
│  └── EvaluationResult(valid, score, additional_info)                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Python Reference 格式要求

Python Reference 必须包含以下内容，格式与 MultiKernelBench 一致：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    参考模型类（必需）
    - __init__ 可以有参数，参数值由 get_init_inputs() 提供
    - forward 方法实现算子逻辑
    """
    def __init__(self, alpha: float = 1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)

# 测试数据配置
batch_size = 16
dim = 16384

def get_inputs():
    """生成测试输入，返回列表"""
    x = torch.randn(batch_size, dim, dtype=torch.float16)
    return [x]

def get_init_inputs():
    """返回 Model.__init__ 的参数列表"""
    return [1.0]  # 对应 alpha 参数
```

**关键要求：**
- `Model` 类：使用 PyTorch 原生实现，`__init__` 和 `forward` 可以有任意参数
- `get_inputs()`: 返回 `forward()` 的输入张量列表
- `get_init_inputs()`: 返回 `__init__()` 的参数列表（不含 self）
- `ModelNew` 类：由 `model_src` 模板自动生成，接口与 `Model` 一致

---

## 5. Kernel 代码格式要求

Ascend C Kernel 需要遵循以下格式：

```cpp
#include "kernel_operator.h"

using namespace AscendC;

class KernelXxx {
public:
    __aicore__ inline KernelXxx() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum) {
        // 初始化 GlobalTensor, LocalTensor, TPipe 等
    }

    __aicore__ inline void Process() {
        // 主处理循环: CopyIn -> Compute -> CopyOut
    }

private:
    // 成员变量
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;
};

// 入口函数（函数名必须是 {op_name}_custom）
extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelXxx op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
```

---

## 6. 测试步骤

### 6.1 Fake Mode 测试（无需 NPU）

```bash
cd /root/Huawei_CANN/evotoolkit
python examples/cann_init/0_test_task.py --fake
```

**预期输出：**
```
============================================================
Test 1: Signature Parser
============================================================
Op Name: add
Inputs: [{'name': 'x', 'dtype': 'float'}, {'name': 'y', 'dtype': 'float'}]
...

============================================================
Test 3: CANNInitTask (fake_mode=True)
============================================================
Valid: True
Score: 1.0
```

### 6.2 Real Mode 测试（需要 NPU）

```bash
cd /root/Huawei_CANN/evotoolkit
python examples/cann_init/0_test_task.py --real
```

**预期输出：**
```
============================================================
Test 4: CANNInitTask (Real NPU Evaluation)
============================================================
Project path: /tmp/xxx
[INFO] Creating operator project with msopgen...
[INFO] Operator project created successfully
[INFO] Writing source files...
[INFO] Building operator...
[INFO] Build succeeded
[INFO] Deploying operator package...
[INFO] Deploy succeeded
[INFO] Building Python bindings...
[INFO] Python binding succeeded

Result:
  Valid: True
  Score: -0.1234
  Stage: success
  Runtime: 0.1234 ms
```

### 6.3 手动测试代码

```python
from evotoolkit.task.cann_init import CANNInitTask

# 准备 Python Reference
python_ref = '''
import torch
import torch.nn as nn

def module_fn(x, y):
    return x + y

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, fn=module_fn):
        return fn(x, y)

def get_inputs():
    return [torch.randn(1024, 1024, dtype=torch.float16),
            torch.randn(1024, 1024, dtype=torch.float16)]

def get_init_inputs():
    return []
'''

# 准备 Kernel 代码
kernel_src = '''
#include "kernel_operator.h"
// ... 完整的 kernel 代码 ...
'''

# 创建 Task
task = CANNInitTask(
    data={
        "op_name": "add",
        "python_reference": python_ref,
    },
    project_path="/tmp/cann_test",  # 可选，默认使用临时目录
    fake_mode=False,
)

# 评估
result = task.evaluate_code(kernel_src)

print(f"Valid: {result.valid}")
print(f"Score: {result.score}")
print(f"Stage: {result.additional_info.get('stage')}")
if result.valid:
    print(f"Runtime: {result.additional_info.get('runtime'):.4f} ms")
else:
    print(f"Error: {result.additional_info.get('error')}")
```

---

## 7. 常见问题

### Q1: msopgen 找不到

```bash
# 确保 CANN 环境已加载
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Q2: Build failed

检查 kernel 代码中：
- 函数名是否为 `{op_name}_custom`
- 是否包含 `GET_TILING_DATA` 宏
- 数据类型是否匹配（half/float）

### Q3: Python binding failed

CppExtension 目录由 evotoolkit 自动创建，如果失败请检查：
- torch_npu 是否已正确安装
- CANN 环境变量是否已设置

### Q4: Correctness check failed

- 检查 `python_reference` 中的数据类型与 kernel 是否匹配
- 调整 atol/rtol 容差（默认 1e-2）
- 检查边界条件处理

---

## 8. 接口对比

| 接口 | 输入 | 适用场景 |
|------|------|----------|
| `evaluate_code(str)` | 仅 kernel 代码 | 简单场景，使用默认模板配置 |
| `evaluate_solution(Solution)` | kernel + other_info | 需要自定义 block_dim、tiling 等 |

```python
# 使用 evaluate_code
result = task.evaluate_code(kernel_src)

# 使用 evaluate_solution（可传额外配置）
from evotoolkit.core import Solution
solution = Solution(
    sol_string=kernel_src,
    other_info={
        "block_dim": 16,
        "tiling_fields": [
            {"name": "totalLength", "type": "uint32_t"},
            {"name": "tileNum", "type": "uint32_t"},
        ],
    }
)
result = task.evaluate_solution(solution)
```

---

## 9. 并行编译支持

evotoolkit 支持对同一算子的多个版本进行并行编译评估。通过自动生成唯一标识符来隔离不同编译实例的 Python 绑定包：

```python
# 并行评估示例
import concurrent.futures
from evotoolkit.task.cann_init import CANNInitTask

def evaluate_kernel(kernel_src):
    task = CANNInitTask(
        data={"op_name": "add", "python_reference": python_ref},
        # 每次调用自动使用不同的 tempfile 目录
        # Python 绑定包名自动带唯一后缀，如 custom_abc123_ops
    )
    return task.evaluate_code(kernel_src)

# 并行评估多个 kernel 版本
kernel_versions = [kernel_v1, kernel_v2, kernel_v3]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(evaluate_kernel, kernel_versions))
```

**实现原理：**
- 每个编译实例的 `setup.py` 使用唯一包名（如 `custom_abc123_ops`）
- 唯一标识符通过 `tempfile` 机制自动生成
- 避免并行安装时 wheel 包互相覆盖

---

## 10. 目录结构（自动生成）

评估过程中自动创建的项目目录结构：

```
/tmp/tmpXXXXXX/                    # project_path (tempfile 生成)
├── CppExtension/                  # Python 绑定编译（自动创建）
│   ├── build_and_run.sh           # 内置模板
│   ├── setup.py                   # 动态生成（包含唯一 ID）
│   └── csrc/
│       └── op.cpp                 # 动态生成
├── AddCustom/                     # 算子项目（自动创建）
│   ├── op_host/
│   │   ├── add_custom_tiling.h
│   │   └── add_custom.cpp
│   ├── op_kernel/
│   │   └── add_custom.cpp
│   ├── build.sh
│   └── build_out/
│       └── custom_opp_ubuntu_aarch64.run
├── add_custom.json                # 算子配置（自动创建）
└── opp/                           # 部署目录（自动创建）
    └── vendors/
        └── customize/
```

---

## 11. 测试清单

以下是需要验证的功能点：

### 11.1 Signature Parser 测试

```python
from evotoolkit.task.cann_init import OperatorSignatureParser

# 测试 1: 无 init 参数的算子 (add)
ADD_REF = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

def get_inputs():
    return [torch.randn(1024, 1024), torch.randn(1024, 1024)]
def get_init_inputs():
    return []
'''

parser = OperatorSignatureParser()
sig = parser.parse(ADD_REF, "add")

# 验证:
# - sig["inputs"] == [{"name": "x", ...}, {"name": "y", ...}]
# - sig["init_params"] == []
print(f"Inputs: {sig['inputs']}")
print(f"Init Params: {sig['init_params']}")
```

```python
# 测试 2: 有 init 参数的算子 (elu)
ELU_REF = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)

def get_inputs():
    return [torch.randn(16, 16384)]
def get_init_inputs():
    return [1.0]
'''

sig = parser.parse(ELU_REF, "elu")

# 验证:
# - sig["inputs"] == [{"name": "x", ...}]
# - sig["init_params"] == [{"name": "alpha", "dtype": "float", "default": 1.0}]
print(f"Inputs: {sig['inputs']}")
print(f"Init Params: {sig['init_params']}")
```

### 11.2 Template Generator 测试

```python
from evotoolkit.task.cann_init import AscendCTemplateGenerator

# 测试 model_src 生成
gen = AscendCTemplateGenerator(sig)  # 使用 ELU 的 sig
full_code = gen.generate(kernel_src="// kernel code")

print("=== model_src ===")
print(full_code["model_src"])
# 验证 ModelNew 应该有:
# - def __init__(self, alpha = 1.0)
# - self.alpha = alpha
# - custom_ops_lib.elu_custom(x, self.alpha)

print("=== python_bind_src ===")
print(full_code["python_bind_src"])
# 验证 pybind 应该有:
# - elu_custom_impl_npu(const at::Tensor& x, float alpha)
```

### 11.3 并行编译隔离测试

```python
from evotoolkit.task.cann_init.pybind_templates import generate_unique_id, get_setup_py

# 验证每次生成不同的 ID
id1 = generate_unique_id()
id2 = generate_unique_id()
print(f"ID1: {id1}, ID2: {id2}")
assert id1 != id2, "IDs should be different"

# 验证 setup.py 包含唯一 ID
setup_content = get_setup_py(id1)
assert f"custom_{id1}_ops" in setup_content
print("Parallel compilation isolation: OK")
```

### 11.4 Fake Mode 完整流程测试

```bash
cd /root/Huawei_CANN/evotoolkit
python examples/cann_init/0_test_task.py --fake
```

验证输出包含:
- `Test 1: Signature Parser` - 正确解析 inputs
- `Test 2: Template Generator` - 生成 6 个组件
- `Test 3: CANNInitTask (fake_mode=True)` - Valid: True, Score: 1.0

### 11.5 Real Mode 测试（需要 NPU 环境）

```bash
# 确保 CANN 环境已加载
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd /root/Huawei_CANN/evotoolkit
python examples/cann_init/0_test_task.py --real
```

验证:
- msopgen 成功创建项目
- build.sh 编译成功
- Python binding 安装成功
- 正确性验证通过
- 返回性能数据
