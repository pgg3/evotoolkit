# CUDA 内核优化示例

[English](README.md) | 简体中文

本目录包含使用 LLM 驱动进化优化 CUDA 内核的完整可运行示例。

## 安装

```bash
pip install evotoolkit[cuda_engineering]
```

**前置条件：**

- 支持 CUDA 的 NVIDIA GPU
- 已安装 CUDA 工具包（11.0+）
- 支持 CUDA 的 PyTorch

## 配置

创建 `.env` 文件并添加 API 凭据：

```bash
HOST=api.openai.com
KEY=sk-your-api-key-here
```

## 示例

### 1. 基础示例 (`basic_example.py`)

**功能：**

- 定义矩阵乘法的 Python 参考实现
- 创建初始的朴素 CUDA 内核
- 使用 EvoEngineerFull 算法运行进化
- 优化内核以降低运行时间，同时保持正确性

**运行：**

```bash
python basic_example.py
```

**预计运行时间：** 10-20 分钟（10 代进化）

---

### 2. 数据集示例 (`dataset_example.py`)

**功能：**

- 加载预定义的 CUDA 优化数据集（RTX 4090 + CUDA 12.4.1 + PyTorch 2.4.0）
- 选择 3D 张量矩阵乘法任务
- 从数据集创建 CUDA 任务
- 运行进化以优化内核

**运行：**

```bash
python dataset_example.py
```

**数据集下载：**

由于数据集较大，未包含在仓库中。可以从以下位置下载：

- **GitHub Release：** [下载 rtx4090_cu12_4_py311_torch_2_4_0.json](https://github.com/pgg3/evotoolkit/releases/download/v0.1.0/rtx4090_cu12_4_py311_torch_2_4_0.json)
- **大小：** ~580 KB（JSON 格式）
- **保存到：** `../../../rtx4090_cu12_4_py311_torch_2_4_0.json`（项目根目录）

或使用 `wget`：
```bash
cd ../../../  # 进入项目根目录
wget https://github.com/pgg3/evotoolkit/releases/download/v0.1.0/rtx4090_cu12_4_py311_torch_2_4_0.json
```

**数据集包含：**

- 100+ 个预定义的 CUDA 优化任务
- 矩阵运算、激活函数、损失函数
- 归一化层、注意力机制
- 完整的 org_py_code、func_py_code 和 cuda_code
- 针对 RTX 4090 + CUDA 12.4.1 + PyTorch 2.4.0 优化

**预计运行时间：** 10-20 分钟（10 代进化）

---

### 3. 自定义提示示例 (`custom_prompt.py`)

**功能：**

- 演示如何自定义进化提示
- 继承自 `EvoEngineerFullCudaInterface`
- 重写变异提示以强调内存优化策略
- 展示如何引导 LLM 朝向特定的优化目标

**运行：**

```bash
python custom_prompt.py
```

**关键学习点：**

- 如何创建自定义 CUDA 接口
- 如何重写 `get_operator_prompt()`
- 如何在提示中注入领域特定的优化策略

---

### 4. 算法对比 (`compare_algorithms.py`)

**功能：**

- 在同一 CUDA 任务上比较五种进化算法：
  - EvoEngineerFull：具有初始化、变异、交叉的完整工作流
  - EvoEngineerFree：自由形式优化
  - EvoEngineerInsight：洞察引导的优化
  - EoH：启发式进化
  - FunSearch：函数搜索优化
- 在同一任务上运行所有算法
- 按运行时性能排名结果
- 展示哪种算法实现了最佳加速比

**运行：**

```bash
python compare_algorithms.py
```

**预计运行时间：** 40-80 分钟（5 种算法 × 每种 10 代）

## 可用接口

示例演示了不同的 CUDA 优化接口：

- **EvoEngineerFullCudaInterface**：具有初始化、变异和交叉算子的完整进化
- **EvoEngineerFreeCudaInterface**：自由形式优化方法
- **EvoEngineerInsightCudaInterface**：带性能分析的洞察引导优化
- **EoHCudaInterface**：CUDA 的启发式进化
- **FunSearchCudaInterface**：GPU 内核的函数搜索

## 输出

所有示例将结果保存到各自的输出目录：

- `basic_example.py` → `./cuda_optimization_results/`
- `custom_prompt.py` → `./custom_prompt_results/`
- `compare_algorithms.py` → `./results_evoengineer_full/`、`./results_evoengineer_free/` 等

每个目录包含：

- `run_state.json` - 进化统计和历史
- `best_solution.cu` - 最佳进化的 CUDA 内核
- `generation_N/` - 每代的解决方案

## 理解 CUDA 任务评估

### 正确性验证

所有进化的内核都会自动对照 Python 参考实现进行验证，以确保正确性。

### 性能测量

- **运行时间**：使用 CUDA 事件测量的内核执行时间
- **得分**：负的运行时间（得分越高 = 内核越快）
- **性能分析**：CUDA 分析器输出，显示性能瓶颈

### 用于测试的假模式

你可以通过设置 `fake_mode=True` 在无 GPU 的情况下测试进化工作流：

```python
task_info = CudaTaskInfoMaker.make_task_info(
    evaluator=evaluator,
    gpu_type="RTX 4090",
    cuda_version="12.4.1",
    org_py_code=org_py_code,
    func_py_code=func_py_code,
    cuda_code=cuda_code,
    fake_mode=True  # 跳过实际 CUDA 评估
)
```

## 下一步

- 尝试不同的 CUDA 操作（卷积、归约、扫描等）
- 调整进化参数（`max_generations`、`pop_size`、`max_sample_nums`）
- 为特定优化目标自定义提示（内存受限、计算受限）
- 开发针对 GPU 优化的新进化算法

## 文档

详细教程请参见：

- [CUDA 工程教程](../../docs/tutorials/cuda-engineering.md)
- [自定义进化方法](../../docs/tutorials/customizing-evolution.md)
- [API 参考](../../docs/api/index.md)
