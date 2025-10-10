# CUDA 内核优化教程

学习如何使用 LLM 驱动的进化算法来优化 CUDA 内核，在保持正确性的同时降低运行时间。

!!! note "学术引用"
    CUDA 内核优化任务基于 EvoEngineer 研究。如果您在学术工作中使用此功能，请引用：

    ```bibtex
    @misc{guo2025evoengineermasteringautomatedcuda,
        title={EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models},
        author={Ping Guo and Chenyu Zhu and Siyuan Chen and Fei Liu and Xi Lin and Zhichao Lu and Qingfu Zhang},
        year={2025},
        eprint={2510.03760},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2510.03760}
    }
    ```

!!! tip "完整示例代码"
    本教程提供完整可运行的示例（点击查看/下载）：

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/basic_example.py) - 基础用法
    - [:material-download: dataset_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/dataset_example.py) - 使用预定义数据集
    - [:material-download: custom_prompt.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/custom_prompt.py) - 自定义提示示例
    - [:material-download: compare_algorithms.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/compare_algorithms.py) - 算法对比
    - [:material-file-document: README.zh.md](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/README.zh.md) - 示例文档和使用指南

    本地运行：
    ```bash
    cd examples/cuda_task
    python basic_example.py
    # 或使用预定义数据集
    python dataset_example.py
    ```

---

## 概述

本教程演示：

- 创建 CUDA 内核优化任务
- 使用 LLM 驱动的进化优化内核运行时间
- 自动验证内核正确性
- 进化高性能 GPU 代码

---

## 安装

!!! tip "推荐使用 GPU"
    CUDA 内核优化需要 GPU 和 PyTorch。在安装 EvoToolkit 之前请先安装支持 CUDA 的 PyTorch。
    我们推荐使用 **CUDA 12.9**（最新稳定版）。

### 步骤 1：安装 PyTorch（支持 GPU）

```bash
# CUDA 12.9（推荐 - 用于自定义任务）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 其他版本请访问：https://pytorch.org/get-started/locally/
# CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU（不推荐用于 CUDA 任务）
# pip install torch torchvision
```

!!! note "关于 PyTorch 版本"
    我们推荐安装最新的 CUDA 12.9 版本用于自定义任务开发。但请注意：

    - **预定义数据集**：我们提供的示例数据集是基于 **CUDA 12.4 + PyTorch 2.4.0** 构建的
    - **版本兼容性**：不同 PyTorch 版本生成的 CUDA 代码可能不同，使用预定义数据集时建议安装匹配的 PyTorch 版本
    - **自定义任务**：如果您创建自己的任务，可以使用任何 PyTorch 版本

### 步骤 2：安装 EvoToolkit

```bash
pip install evotoolkit[cuda_engineering]
```

这会安装：

- Ninja（高性能构建系统）
- Portalocker（跨进程文件锁）
- Psutil（系统和进程工具）

### 步骤 3：安装 C++ 编译器（必需）

!!! danger "关键前置条件：C++ 编译器"
    **CUDA 内核编译需要 C++ 编译器！** 如果缺少编译器，运行时会报错：
    ```
    Error checking compiler version for cl: [WinError 2] 系统找不到指定的文件。
    ```

#### Windows 用户

**必须安装 Visual Studio 及 MSVC 编译器：**

1. **下载 Visual Studio**
   - 访问：[https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)
   - 推荐：**Visual Studio 2022 Community**（免费）

2. **安装时选择工作负载**
   - 勾选 **"使用 C++ 的桌面开发"**（Desktop development with C++）
   - 这会安装 MSVC 编译器和必要的构建工具

3. **CUDA 版本与 MSVC 兼容性**

   | CUDA 版本 | 支持的 Visual Studio 版本 | 支持的 MSVC 版本 |
   |---------|----------------------|--------------|
   | 12.9    | VS 2022 (17.x)<br>VS 2019 (16.x) | MSVC 193x<br>MSVC 192x |
   | 12.4    | VS 2022 (17.x)<br>VS 2019 (16.x) | MSVC 193x<br>MSVC 192x |
   | 12.1    | VS 2022 (17.x)<br>VS 2019 (16.x)<br>VS 2017 (15.x) | MSVC 193x<br>MSVC 192x<br>MSVC 191x |

!!! warning "重要说明"
    - Visual Studio 2017 在 CUDA 12.5 被弃用，在 12.9 已完全移除支持
    - 从 CUDA 12.0 开始仅支持 64 位编译（不再支持 32 位）
    - 支持 C++14（默认）、C++17 和 C++20

4. **验证编译器安装**
   ```bash
   # 打开 "x64 Native Tools Command Prompt for VS 2022"（从开始菜单找到）
   cl

   # 应该看到类似输出：
   # Microsoft (R) C/C++ Optimizing Compiler Version 19.39.xxxxx for x64
   ```

   如果 `cl` 命令在普通命令提示符中不可用，有两种解决方案：

   **方案 A：使用 VS 开发者命令提示符（推荐）**
   - 从开始菜单搜索 "x64 Native Tools Command Prompt for VS 2022"
   - 在此命令提示符中运行你的 Python 脚本

   **方案 B：添加到系统 PATH（永久）**
   ```bash
   # 将 MSVC 添加到系统环境变量 PATH（示例路径，根据实际安装位置调整）
   # C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.xxxxx\bin\Hostx64\x64
   ```

#### Linux/Ubuntu 用户

**安装 GCC/G++ 编译器：**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential

# 验证安装
gcc --version
g++ --version

# 推荐版本：GCC 9.x 或更高
```

**CUDA 版本与 GCC 兼容性：**

| CUDA 版本 | 支持的 GCC 版本 |
|---------|-------------|
| 12.9    | GCC 9.x - 13.x |
| 12.4    | GCC 9.x - 13.x |
| 12.1    | GCC 9.x - 12.x |

!!! tip "检查 CUDA 与编译器兼容性"
    如果遇到编译错误，请检查：

    1. CUDA 版本：`nvcc --version`
    2. 编译器版本：Windows 用 `cl`，Linux 用 `gcc --version`
    3. 确认版本在上述兼容性表格范围内

**前置条件总结：**

- ✅ 支持 CUDA 的 NVIDIA GPU
- ✅ 已安装 CUDA 工具包（12.1+ 推荐）
- ✅ **已安装兼容的 C++ 编译器**（Windows: MSVC，Linux: GCC）
- ✅ PyTorch >= 2.0（支持 CUDA）
- ✅ CUDA 编程基础知识
- ✅ 熟悉内核优化概念

---

## 理解 CUDA 任务

### 什么是 CUDA 任务？

CUDA 任务优化 GPU 内核代码以最小化运行时间，同时确保正确性。框架会：

1. 接收你的 Python 函数实现
2. 转换为函数式 Python 代码（如需要）
3. 翻译为初始 CUDA 内核
4. 进化内核以提升性能
5. 对照 Python 参考验证正确性

### 任务组件

一个 CUDA 任务需要：

- **原始 Python 代码** (`org_py_code`)：原始 PyTorch 模型代码（可选，可留空）
- **功能性 Python 代码** (`func_py_code`)：提取的功能函数实现，用于正确性比较和性能基准测量
- **CUDA 代码** (`cuda_code`)：初始 CUDA 内核实现
- **GPU 信息**：GPU 类型和 CUDA 版本

!!! note "关于 org_py_code 和 func_py_code"
    - `func_py_code` 必须提供，是实际用于 CUDA 正确性验证和性能对比的 Python 参考实现
    - 如果只有 `org_py_code`，可以使用 AI-CUDA-Engineer 工作流（Stage 0）让 LLM 将其转换为 `func_py_code`
    - `org_py_code` 可以留空，直接提供 `func_py_code`（推荐用于进化优化）

---

!!! warning "Windows 用户必读：多进程保护"
    **CUDA 任务评估器使用 multiprocessing 模块执行超时控制。在 Windows 上运行时，必须使用 `if __name__ == '__main__':` 保护所有主代码，否则会导致进程无限递归创建！**

    **错误示例（会导致 RuntimeError）：**
    ```python
    # ❌ 错误 - 没有保护
    import os
    from evotoolkit.task.cuda_engineering import CudaTask

    evaluator = Evaluator(temp_path)  # 会在 Windows 上崩溃！
    task_info = CudaTaskInfoMaker.make_task_info(...)
    ```

    **正确示例：**
    ```python
    # ✅ 正确 - 使用 if __name__ == '__main__': 保护
    import os
    from evotoolkit.task.cuda_engineering import CudaTask

    def main():
        evaluator = Evaluator(temp_path)
        task_info = CudaTaskInfoMaker.make_task_info(...)
        # ... 其他代码

    if __name__ == '__main__':
        main()
    ```

    **为什么需要这个保护？**

    - Windows 不支持 `fork`，只支持 `spawn` 方式启动子进程
    - `spawn` 会重新导入主模块来创建子进程
    - 如果没有保护，每次导入都会执行主代码，导致无限递归

    **规则：凡是调用 CUDA 任务评估的代码，都必须放在 `if __name__ == '__main__':` 保护内！**

---

## 使用预定义数据集

EvoToolkit 提供了预定义的 CUDA 优化数据集，包含多种常见的深度学习操作。

### 下载数据集

数据集未包含在主仓库中，需要单独下载：

**下载方式：**

```bash
# 方式 1: 使用 wget
cd /path/to/evotool/project/root
wget https://github.com/pgg3/evotoolkit/releases/download/data-v1.0.0/rtx4090_cu12_4_py311_torch_2_4_0.json

# 方式 2: 使用 curl
curl -L -O https://github.com/pgg3/evotoolkit/releases/download/data-v1.0.0/rtx4090_cu12_4_py311_torch_2_4_0.json
```

**数据集信息：**

- **文件名：** `rtx4090_cu12_4_py311_torch_2_4_0.json`
- **大小：** ~580 KB
- **格式：** JSON
- **优化目标：** RTX 4090 GPU + CUDA 12.4.1 + PyTorch 2.4.0

!!! warning "数据集说明"
    此数据集是针对特定硬件和软件配置的示例数据集，不像 scientific_regression 任务那样支持自动下载。你可以根据自己的硬件环境创建类似的数据集。

### 加载数据集

```python
import json

# 加载针对 RTX 4090 + CUDA 12.4.1 + PyTorch 2.4.0 的数据集
with open('rtx4090_cu12_4_py311_torch_2_4_0.json', 'r') as f:
    dataset = json.load(f)

# 查看可用任务
print(f"可用任务数量: {len(dataset)}")
print(f"任务列表: {list(dataset.keys())[:5]}...")  # 显示前5个

# 选择一个任务
task_name = "10_3D_tensor_matrix_multiplication"
task_data = dataset[task_name]

print(f"\n任务: {task_name}")
print(f"- org_py_code: {'已提供' if task_data['org_py_code'] else '空'}")
print(f"- func_py_code: {'已提供' if task_data['func_py_code'] else '空'}")
print(f"- cuda_code: {'已提供' if task_data['cuda_code'] else '空'}")
```

**数据集包含的任务类型：**

- 矩阵乘法变体（3D、4D 张量，对角矩阵，对称矩阵等）
- 激活函数（ReLU、Sigmoid、Tanh、GELU 等）
- 损失函数（CrossEntropy、HingeLoss 等）
- 归一化层（LayerNorm、BatchNorm 等）
- 注意力机制和 Transformer 组件

### 从数据集创建任务

```python
from evotoolkit.task.cuda_engineering import CudaTask, CudaTaskInfoMaker
from evotoolkit.task.cuda_engineering.evaluator import Evaluator
import tempfile
import os


def main():
    # 配置 CUDA 环境变量（运行前必须设置）
    # Windows: 设置为你的 CUDA 安装路径
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
    # Linux/Ubuntu: 通常为默认路径
    # os.environ["CUDA_HOME"] = "/usr/local/cuda"

    # 指定 GPU 架构以节省编译时间（根据你的 GPU 设置）
    # RTX 4090: 8.9, RTX 3090: 8.6, V100: 7.0
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"

    # 使用数据集中的任务数据
    task_data = dataset["10_3D_tensor_matrix_multiplication"]

    # 创建评估器和任务
    temp_path = tempfile.mkdtemp()
    evaluator = Evaluator(temp_path)

    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=task_data["org_py_code"],      # 可为空
        func_py_code=task_data["func_py_code"],    # 功能性实现
        cuda_code=task_data["cuda_code"],          # 初始 CUDA 内核
        fake_mode=False
    )

    task = CudaTask(data=task_info, temp_path=temp_path, fake_mode=False)
    print(f"任务已创建，初始运行时间: {task.task_info['cuda_info']['runtime']:.4f} ms")


if __name__ == '__main__':
    main()
```

---

## 示例：从头创建矩阵乘法优化任务

如果你想从头创建自己的 CUDA 优化任务：

### 步骤 1：准备 Python 函数

!!! note "func_py_code 格式要求"
    `func_py_code` 必须包含以下组件：

    1. **`module_fn` 函数**：核心功能实现
    2. **`Model` 类**：继承 `nn.Module`，其 `forward` 方法接受 `fn=module_fn` 参数
    3. **`get_inputs()` 函数**：生成测试输入数据
    4. **`get_init_inputs()` 函数**：生成初始化输入（通常为空列表）

    这种设计允许 CUDA 内核通过传入不同的 `fn` 替换 `module_fn`，从而进行正确性验证。

```python
# 要优化的原始函数（可选）
org_py_code = '''
import torch

def matmul(A, B):
    """使用 PyTorch 的矩阵乘法。"""
    return torch.matmul(A, B)
'''

# 功能性实现版本（用于正确性比较和性能基准）
func_py_code = '''
import torch
import torch.nn as nn

def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """功能性矩阵乘法实现。"""
    return torch.matmul(A, B)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, fn=module_fn):
        return fn(A, B)

M = 1024
K = 2048
N = 1024

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
'''
```

### 步骤 2：创建初始 CUDA 内核

```python
# 初始 CUDA 实现（朴素版本）
cuda_code = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(float* A, float* B, float* C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA)");
}
'''
```

### 步骤 3：创建 CUDA 任务

```python
from evotoolkit.task.cuda_engineering import CudaTask, CudaTaskInfoMaker
from evotoolkit.task.cuda_engineering.evaluator import Evaluator
import tempfile
import os


def main():
    # 配置 CUDA 环境变量（运行前必须设置）
    # Windows: 设置为你的 CUDA 安装路径
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
    # Linux/Ubuntu: 通常为默认路径
    # os.environ["CUDA_HOME"] = "/usr/local/cuda"

    # 指定 GPU 架构以节省编译时间
    # RTX 4090: 8.9, RTX 3090: 8.6, V100: 7.0
    os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"

    # 创建评估器
    temp_path = tempfile.mkdtemp()
    evaluator = Evaluator(temp_path)

    # 创建任务信息
    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=org_py_code,
        func_py_code=func_py_code,
        cuda_code=cuda_code,
        fake_mode=False  # 设为 True 可在无 GPU 情况下测试
    )

    # 创建任务
    task = CudaTask(
        data=task_info,
        temp_path=temp_path,
        fake_mode=False
    )

    print(f"GPU 类型: {task.task_info['gpu_type']}")
    print(f"CUDA 版本: {task.task_info['cuda_version']}")
    print(f"初始运行时间: {task.task_info['cuda_info']['runtime']:.4f} ms")


if __name__ == '__main__':
    main()
```

**输出：**
```
GPU 类型: RTX 4090
CUDA 版本: 12.4.1
初始运行时间: 2.3456 ms
```

### 步骤 4：测试初始解决方案

```python
def main():
    # ... (前面的步骤 3 代码)

    # 获取初始解决方案
    init_sol = task.make_init_sol_wo_other_info()

    print("初始内核信息：")
    print(f"运行时间: {-init_sol.evaluation_res.score:.4f} ms")
    print(f"得分: {init_sol.evaluation_res.score:.6f}")


if __name__ == '__main__':
    main()
```

**理解评估：**

- **得分**：负的运行时间（越高越好，所以更快的内核得分更高）
- **运行时间**：内核执行时间（毫秒）
- **正确性**：自动对照 Python 参考验证
- **性能分析字符串**：CUDA 分析器输出，显示瓶颈

### 步骤 5：使用 EvoEngineer 运行进化

!!! note "完整代码示例"
    以下代码假设你已经完成了前面的步骤（步骤 1-4），并且 `task` 对象已经创建。如果需要完整的可运行代码，请参考 [basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/cuda_task/basic_example.py)。

```python
import os
import evotoolkit
from evotoolkit.task.cuda_engineering import EvoEngineerFullCudaInterface
from evotoolkit.tools.llm import HttpsApi


def main():
    # === 前面的步骤（步骤 1-4）===
    # 这里应该包含前面步骤中的代码：
    # - 定义 org_py_code, func_py_code, cuda_code
    # - 创建 evaluator 和 task_info
    # - 创建 task 对象
    # 完整代码请参考 basic_example.py

    # 设置 CUDA 环境变量（CUDA 内核编译必需）
    # CUDA_HOME: CUDA 安装目录路径
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    # TORCH_CUDA_ARCH_LIST: GPU 计算能力（例如 RTX 4090 为 "8.9"）
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    # 创建接口（使用前面步骤创建的 task 对象）
    interface = EvoEngineerFullCudaInterface(task)

    # 配置 LLM API
    # 设置 LLM_API_URL 和 LLM_API_KEY 环境变量
    llm_api = HttpsApi(
        api_url=os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        key=os.environ.get("LLM_API_KEY", "your-api-key-here"),
        model="gpt-4o"
    )

    # 运行进化
    result = evotoolkit.solve(
        interface=interface,
        output_path='./cuda_optimization_results',
        running_llm=llm_api,
        max_generations=10,
        pop_size=5,
        max_sample_nums=20
    )

    print(f"找到最佳内核！")
    print(f"运行时间: {-result.evaluation_res.score:.4f} ms")
    print(f"加速比: {task.task_info['cuda_info']['runtime'] / (-result.evaluation_res.score):.2f}x")
    print(f"\n优化后的内核：\n{result.sol_string}")


if __name__ == '__main__':
    main()
```

!!! tip "尝试其他算法"
    EvoToolkit 支持多种 CUDA 优化进化算法：

    ```python
    # 使用 EoH
    from evotoolkit.task.cuda_engineering import EoHCudaInterface
    interface = EoHCudaInterface(task)

    # 使用 FunSearch
    from evotoolkit.task.cuda_engineering import FunSearchCudaInterface
    interface = FunSearchCudaInterface(task)

    # 使用 EvoEngineer 洞察模式
    from evotoolkit.task.cuda_engineering import EvoEngineerInsightCudaInterface
    interface = EvoEngineerInsightCudaInterface(task)

    # 使用 EvoEngineer 自由模式
    from evotoolkit.task.cuda_engineering import EvoEngineerFreeCudaInterface
    interface = EvoEngineerFreeCudaInterface(task)
    ```

    然后使用相同的 `evotoolkit.solve()` 调用运行进化。不同的接口可能在不同的内核上表现更好。

---

## 自定义进化行为

进化过程的质量主要由 **进化方法** 及其内部的 **提示设计** 控制。如果想提升结果：

- **调整提示**：继承现有的 Interface 类并自定义 LLM 提示
- **开发新算法**：创建全新的进化策略和算子

!!! tip "了解更多"
    这些是适用于所有任务的通用技术。详细教程请参见：

    - **[自定义进化方法](../customization/customizing-evolution.zh.md)** - 如何修改提示和开发新算法
    - **[高级用法](../advanced-overview.zh.md)** - 更多高级配置选项

**快速示例 - 为 CUDA 优化自定义提示：**

````python
from evotoolkit.task.cuda_engineering import EvoEngineerFullCudaInterface

class OptimizedCudaInterface(EvoEngineerFullCudaInterface):
    """为内存受限内核优化的接口。"""

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """自定义变异提示以强调内存访问模式。"""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            individual = selected_individuals[0]

            prompt = f"""# CUDA 内核优化 - 内存优化重点
{task_description}

## 当前最佳
**名称：** {current_best_sol.other_info['name']}
**运行时间：** {-current_best_sol.evaluation_res.score:.5f} 毫秒

## 待变异内核
**名称：** {individual.other_info['name']}
**运行时间：** {-individual.evaluation_res.score:.5f} 毫秒

## 优化重点
重点优化内存访问模式：
- 使用共享内存减少全局内存访问
- 实现内存合并以提高带宽
- 考虑内存 bank 冲突
- 使用适当的内存访问模式（纹理内存、常量内存）

生成一个减少内存瓶颈的改进内核。

## 响应格式：
name: [描述性名称]
code:
```cpp
[您的 CUDA 内核实现]
```
thought: [内存优化理由]
"""
            return [{"role": "user", "content": prompt}]

        # 其他算子使用默认提示
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# 使用自定义接口
interface = OptimizedCudaInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10
)
````

!!! note "关于 EvoEngineer 算子"
    EvoEngineer 使用三个算子：**init**（初始化）、**mutation**（变异）、**crossover**（交叉）。
    父类 `EvoEngineerFullCudaInterface` 已经定义了这些算子和默认提示。
    你只需重写 `get_operator_prompt()` 来自定义特定算子的提示 - 其他算子会自动使用默认实现。

完整的自定义教程和更多示例，请参见 [自定义进化方法](../customization/customizing-evolution.zh.md)。

---

## 理解评估

### 评分机制

1. **正确性验证**：CUDA 内核输出与 Python 参考实现进行比较
2. **运行时测量**：使用 CUDA 事件和分析工具测量内核执行时间
3. **适应度**：负的运行时间（越高越好，所以越低的运行时间 = 越高的适应度）

### 评估输出

```python
result = task.evaluate_code(candidate_cuda_code)

if result.valid:
    print(f"得分: {result.score}")                                    # 越高越好
    print(f"运行时间: {-result.score:.4f} ms")                        # 实际运行时间
    print(f"性能分析: {result.additional_info['prof_string']}")       # CUDA 分析器输出
else:
    if result.additional_info['compilation_error']:
        print(f"编译错误: {result.additional_info['error_msg']}")
    elif result.additional_info['comparison_error']:
        print(f"正确性错误: {result.additional_info['error_msg']}")
```

### 用于测试的假模式

你可以使用假模式在无 GPU 的情况下测试：

```python
def main():
    task_info = CudaTaskInfoMaker.make_task_info(
        evaluator=evaluator,
        gpu_type="RTX 4090",
        cuda_version="12.4.1",
        org_py_code=org_py_code,
        func_py_code=func_py_code,
        cuda_code=cuda_code,
        fake_mode=True  # 跳过实际 CUDA 评估
    )

    task = CudaTask(data=task_info, fake_mode=True)


if __name__ == '__main__':
    main()
```

---

## 常见问题（Q&A）

### Q: 运行时出现 `_get_vc_env is private` 警告怎么办？

**问题描述：**

在 Windows 上编译 CUDA 扩展时，可能会看到以下警告：

```
UserWarning: _get_vc_env is private; find an alternative (pypa/distutils#340)
```

**原因分析：**

这是 setuptools/distutils 在 Windows 上检测 MSVC 编译器时的兼容性警告。具体原因：

- CUDA 扩展编译需要 Visual Studio C++ 编译器（MSVC）
- setuptools 调用了内部函数 `_get_vc_env()` 来获取编译器环境
- Python 正在将 distutils 迁移到 setuptools，过程中一些内部 API 标记为私有

**影响程度：**

- ⚠️ 这只是一个 UserWarning，**不影响程序运行**
- ✅ **不影响 CUDA 内核编译**
- ✅ **不影响优化结果**

**解决方案：**

**方案 1：过滤警告（推荐）**

在脚本开头添加警告过滤：

```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='setuptools')

# 或者更精确地过滤
warnings.filterwarnings('ignore', message='.*_get_vc_env is private.*')

# 然后导入其他模块
from evotoolkit.task.cuda_engineering import CudaTask
# ...
```

**方案 2：升级 setuptools**

尝试升级到最新版本（可能已修复此问题）：

```bash
pip install --upgrade setuptools
```

**方案 3：忽略**

如果不介意看到警告，可以直接忽略。这个警告不会影响功能，只是提醒开发者内部 API 可能在未来版本中改变。

---

### Q: 为什么在 Windows 上必须使用 `if __name__ == '__main__':` 保护？

**原因：**

- Windows 不支持 `fork` 进程创建方式，只支持 `spawn`
- `spawn` 方式会重新导入主模块来创建子进程
- CUDA 任务评估器使用 `multiprocessing` 模块进行超时控制
- 如果没有保护，每次导入都会执行主代码，导致无限递归创建进程

**正确示例：**

```python
from evotoolkit.task.cuda_engineering import CudaTask

def main():
    evaluator = Evaluator(temp_path)
    task = CudaTask(...)
    # 所有任务代码

if __name__ == '__main__':
    main()
```

**错误示例（会崩溃）：**

```python
from evotoolkit.task.cuda_engineering import CudaTask

# ❌ 直接在模块级别执行
evaluator = Evaluator(temp_path)  # 会导致 RuntimeError
```

---

## 下一步

### 探索不同的优化策略

- 尝试不同的进化算法（EvoEngineer 变体、EoH、FunSearch）
- 比较不同接口的结果
- 分析性能分析以识别瓶颈
- 实验不同的内核模式（分块、共享内存等）

### 自定义和改进进化过程

- 检查现有 Interface 类中的提示设计
- 继承并重写 Interface 以自定义提示
- 为不同的优化目标设计专门的提示（内存受限、计算受限等）
- 如有需要，开发全新的进化算法

### 了解更多

- [自定义进化方法](../customization/customizing-evolution.zh.md) - 深入了解提示自定义和算法开发
- [高级用法](../advanced-overview.zh.md) - 高级配置和技巧
- [API 参考](../../api/index.md) - 完整的 API 文档
- [开发文档](../../development/contributing.zh.md) - 贡献新方法和功能
