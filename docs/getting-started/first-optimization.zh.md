# 第一个优化任务：科学符号回归

让我们使用 EvoToolkit 从真实科学数据中发现数学方程。

!!! note "学术引用"
    本指南使用的科学符号回归任务和数据集基于 CoEvo 的研究。如果您在学术工作中使用此功能，请引用：

    ```bibtex
    @misc{guo2024coevocontinualevolutionsymbolic,
        title={CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models},
        author={Ping Guo and Qingfu Zhang and Xi Lin},
        year={2024},
        eprint={2412.18890},
        archivePrefix={arXiv},
        primaryClass={cs.AI},
        url={https://arxiv.org/abs/2412.18890}
    }
    ```

---

## 步骤 1: 创建新项目

```bash
mkdir my-evotool-project
cd my-evotool-project
```

## 步骤 2: 编写第一个脚本

创建名为 `first_optimization.py` 的文件：

```python
import evotoolkit
from evotoolkit.task.python_task.scientific_regression import ScientificRegressionTask
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools import HttpsApi

# 步骤 1: 创建任务
print("创建科学符号回归任务...")
task = ScientificRegressionTask(dataset_name="bactgrow")

# 步骤 2: 创建接口
print("设置 EvoEngineer 接口...")
interface = EvoEngineerPythonInterface(task)

# 步骤 3: 配置 LLM API
print("配置 LLM API...")
llm_api = HttpsApi(
    api_url="https://api.openai.com/v1/chat/completions",  # 填写您的 API 地址
    key="your-api-key-here",  # 填写您的 API 密钥
    model="gpt-4o"
)

# 步骤 4: 运行优化
print("\n使用 EvoEngineer 开始优化...")
print("这可能需要几分钟...\n")

result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=5,
    max_sample_nums=10,
    pop_size=5
)

print("\n" + "="*60)
print("优化完成！")
print("="*60)
print(f"\n最佳解适应度: {result.evaluation_res.score}")
print(f"结果已保存到: ./results/")
```

## 步骤 3: 运行脚本

```bash
python first_optimization.py
```

您应该看到类似以下的输出：

```
创建科学符号回归任务...
设置 EvoEngineer 接口...
配置 LLM API...

使用 EvoEngineer 开始优化...
这可能需要几分钟...

第 1/5 代: 最佳适应度 = 0.245
第 2/5 代: 最佳适应度 = 0.189
第 3/5 代: 最佳适应度 = 0.134
第 4/5 代: 最佳适应度 = 0.098
第 5/5 代: 最佳适应度 = 0.067

============================================================
优化完成！
============================================================

最佳解适应度: 0.067
结果已保存到: ./results/
```

---

下一步： [理解代码](understanding-code.zh.md)
