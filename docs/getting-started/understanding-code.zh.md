# 理解代码

让我们分解每个部分的作用：

---

## 1. 任务创建

```python
task = ScientificRegressionTask(dataset_name="bactgrow")
```

`Task` 定义了您要解决的问题以及如何评估解决方案。`ScientificRegressionTask` 是用于从真实科学数据中发现数学方程的内置任务。数据集会在首次运行时自动下载（懒加载）。

---

## 2. 接口创建

```python
interface = EvoEngineerPythonInterface(task)
```

`Interface` 将您的任务连接到特定的进化算法。这里我们使用 `EvoEngineerPythonInterface` 来使用 EvoEngineer 算法。

---

## 3. LLM 配置

```python
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",
    key="your-api-key-here",
    model="gpt-4o"
)
```

这会设置 LLM API 客户端，用于生成和改进代码解决方案。请将 `your-api-key-here` 替换为您的实际 API 密钥。

---

## 4. 求解问题

```python
result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    max_sample_nums=10,
    pop_size=5
)
```

`evotoolkit.solve()` 函数：

- 运行进化算法 5 代
- 使用种群大小为 5
- 每代最多采样 10 个 LLM 响应
- 将结果保存到 `./results/`

---

下一步： [探索结果](exploring-results.zh.md)
