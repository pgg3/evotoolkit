# 探索结果

运行后，检查 `./results/` 目录：

---

## 结果目录结构

```
results/
├── run_state.json              # 运行状态和统计信息
├── history/                    # 历史记录
│   ├── gen_-1.json            # 初始种群
│   ├── gen_1.json             # 第 1 代的所有解
│   ├── gen_2.json             # 第 2 代的所有解
│   └── ...
└── summary/                    # 摘要信息
    ├── usage_history.json     # LLM 使用统计
    └── best_per_generation.json  # 每代最优解（如有）
```

---

## 以编程方式分析结果

每个 `gen_N.json` 文件包含该代的所有解决方案、评估结果和统计信息。您可以通过编程方式加载和分析这些结果：

```python
import json

# 加载某一代的历史
with open('./results/history/gen_1.json', 'r') as f:
    gen_1 = json.load(f)

# 查看该代的所有解决方案
for sol in gen_1['solutions']:
    print(f"Score: {sol['evaluation_res']['score']}")
    print(f"Solution:\n{sol['sol_string']}\n")
```

---

下一步： [尝试不同的算法](try-algorithms.zh.md)
