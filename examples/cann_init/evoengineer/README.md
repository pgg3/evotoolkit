# EvoEngineer CANN Kernel Generation

使用进化算法 (EvoEngineer) 自动生成和优化 Ascend C 算子实现。

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 LLM API 配置
```

### 2. 测试 Prompt 生成 (不需要 LLM)

```bash
python run_evoengineer.py --dry-run
```

### 3. 运行完整测试

```bash
python run_evoengineer.py --max-samples 10 --max-generations 3
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--npu` | Ascend910B | NPU 型号 |
| `--max-samples` | 10 | 最大采样数 |
| `--max-generations` | 3 | 最大代数 |
| `--pop-size` | 3 | 种群大小 |
| `--num-samplers` | 2 | 并行采样器数 |
| `--dry-run` | - | 仅测试 prompt 生成 |

## 算法说明

EvoEngineer 使用进化策略优化 kernel 实现：

1. **Init**: 从零生成初始实现
2. **Crossover**: 组合两个 parent 的优化策略
3. **Mutation**: 对现有实现进行变异探索

评估指标：`-runtime` (运行时间越短越好)
