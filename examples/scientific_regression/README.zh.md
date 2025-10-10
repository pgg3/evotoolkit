# 科学符号回归示例

[English](README.md) | 简体中文

本目录包含科学符号回归任务的完整可运行示例。

## 安装

```bash
pip install evotoolkit[scientific_regression]
```

## 配置

设置环境变量或直接在代码中配置 API key：

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

或者直接在代码中设置（参见各示例文件）。

## 示例列表

### 1. 基础示例 (`basic_example.py`)

**功能：**
- 加载细菌生长数据集
- 创建科学符号回归任务
- 使用 EvoEngineer 算法运行进化
- 从数据中发现数学方程

**运行：**
```bash
python basic_example.py
```

**预计运行时间：** 3-5 分钟（3 代）

---

### 2. 自定义 Prompt 示例 (`custom_prompt.py`)

**功能：**
- 演示如何自定义进化 prompt
- 继承 `EvoEngineerPythonInterface` 类
- 重写 mutation prompt 以强调科学原理
- 展示如何引导 LLM 生成符合生物学的方程

**运行：**
```bash
python custom_prompt.py
```

**学习要点：**
- 如何创建自定义接口
- 如何重写 `get_operator_prompt()`
- 如何在 prompt 中注入领域知识

---

### 3. 算法对比 (`compare_algorithms.py`)

**功能：**
- 对比三种进化算法：EvoEngineer、EoH、FunSearch
- 在同一任务上运行所有算法
- 根据性能排名结果
- 展示哪种算法最适合该任务

**运行：**
```bash
python compare_algorithms.py
```

**预计运行时间：** 10-15 分钟（3 个算法 × 3 代）

## 可用数据集

- `bactgrow`: 大肠杆菌细菌生长率（4 个输入：种群、底物、温度、pH）
- `oscillator1`: 阻尼非线性振荡器加速度（2 个输入：位置、速度）
- `oscillator2`: 阻尼非线性振荡器变体（2 个输入：位置、速度）
- `stressstrain`: 铝棒应力预测（2 个输入：应变、温度）

## 输出结果

所有示例将结果保存到各自的输出目录：
- `basic_example.py` → `./scientific_regression_results/`
- `custom_prompt.py` → `./custom_prompt_results/`
- `compare_algorithms.py` → `./results_evoengineer/`、`./results_eoh/`、`./results_funsearch/`

每个目录包含：
- `run_state.json` - 进化统计数据
- `best_solution.py` - 最佳进化方程
- `generation_N/` - 每一代的解

## 下一步

- 通过更改 `dataset_name` 参数尝试不同数据集
- 调整进化参数（`max_generations`、`pop_size` 等）
- 为特定领域自定义 prompt
- 开发新的进化算法

## 相关资源

完整的使用教程和 API 文档请访问 EvoToolkit 的在线文档站点。
