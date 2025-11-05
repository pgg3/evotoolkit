# 对抗攻击示例

[English](README.md) | 简体中文

本目录包含使用 LLM 驱动进化来进化对抗攻击算法的示例。

## 概述

`AdversarialAttackTask` 是一个 **Python 任务**，演示如何使用 EvoToolkit 进化黑盒对抗攻击的 `draw_proposals` 函数。

### 与其他 Python 任务的主要区别

| 方面 | 科学符号回归 | 对抗攻击 |
|------|-------------|---------|
| **进化目标** | 数学方程 | 提案生成算法 |
| **函数名称** | `equation` | `draw_proposals` |
| **输入** | 数据特征 + 参数 | 图像 + 噪声 + 超参数 |
| **评估** | 预测的 MSE | 对抗样本的 L2 距离 |
| **目标** | 发现方程 | 在欺骗模型的同时最小化扰动 |

## 快速开始

### 基础示例（模拟模式）

使用模拟评估运行基础示例（无需模型）：

```bash
python basic_example.py
```

**输出：**
```
Initial draw_proposals function created
Initial score: -2.34 (avg L2 distance: 2.34)

Custom algorithm tested
Score: -1.89 (avg L2 distance: 1.89)

Key points:
- Solutions are Python FUNCTIONS (draw_proposals)
- Function generates adversarial perturbations
- Lower L2 distance = better attack
```

### 使用真实模型

1. 安装依赖：
   ```bash
   # 步骤 1: 安装 PyTorch（支持 GPU，推荐 CUDA 12.9）
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

   # 步骤 2: 安装 EvoToolkit
   pip install evotoolkit[adversarial_attack]
   ```

2. 在代码中直接配置 LLM API 凭证：
   ```python
   llm_api = HttpsApi(
       api_url="api.openai.com",  # 你的 API URL
       key="your-api-key-here",   # 你的 API 密钥
       model="gpt-4o"
   )
   ```

3. 运行进化（参见 `advanced_example.py`）

## 任务结构

### 创建任务

```python
from evotoolkit.task.python_task import AdversarialAttackTask

# 选项 1: 模拟模式（无需模型）
task = AdversarialAttackTask(
    use_mock=True,
    attack_steps=1000,
    n_test_samples=10
)

# 选项 2: 使用真实模型
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms

# 加载 CIFAR-10 预训练的 ResNet18（来自 Hugging Face Hub）
# CIFAR-10 的 ResNet18 使用修改过的架构（3x3 conv1, 移除 maxpool）
base_model = timm.create_model("resnet18", num_classes=10, pretrained=False)
base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
base_model.maxpool = nn.Identity()

# 加载预训练权重
base_model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet18_cifar10.pth"
    )
)
base_model.eval()

# 创建带 Normalization 的模型包装器
# 重要：Foolbox 需要输入在 [0, 1] 范围，所以 normalization 必须在模型内部
class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # x 在 [0, 1] 范围，进行标准化
        x_normalized = (x - self.mean) / self.std
        return self.model(x_normalized)

model = NormalizedModel(base_model,
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2471, 0.2435, 0.2616])
model.eval()

# 加载数据（只做 ToTensor，不做 Normalize）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换到 [0, 1] 范围
])
test_set = datasets.CIFAR10(root='./data', train=False,
                            download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=1000,
    n_test_samples=10,
    use_mock=False
)
```

### 测试函数

```python
code = '''
import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    # 你的算法
    ...
    return candidate
'''

result = task.evaluate_code(code)

print(f"得分: {result.score:.2f}")  # 负的 L2 距离
print(f"平均 L2: {result.additional_info['avg_distance']:.2f}")
```

### 运行进化

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi

interface = EvoEngineerPythonInterface(task)

llm_api = HttpsApi(
    api_url="api.openai.com",
    key="your-api-key-here",
    model="gpt-4o"
)

result = evotoolkit.solve(
    interface=interface,
    output_path='./results',
    running_llm=llm_api,
    max_generations=10
)

# 找到的最佳算法
print(result.sol_string)
print(f"最佳 L2 距离: {-result.evaluation_res.score:.2f}")
```

## 可用算法

```python
# EvoEngineer（推荐）
from evotoolkit.task.python_task import EvoEngineerPythonInterface
interface = EvoEngineerPythonInterface(task)

# EoH (Evolution of Heuristics)
from evotoolkit.task.python_task import EoHPythonInterface
interface = EoHPythonInterface(task)

# FunSearch
from evotoolkit.task.python_task import FunSearchPythonInterface
interface = FunSearchPythonInterface(task)
```

## 函数规范

### draw_proposals 函数

```python
def draw_proposals(
    org_img: np.ndarray,      # 原始图像 (3, H, W) 范围 [0, 1]
    best_adv_img: np.ndarray, # 最佳对抗样本 (3, H, W) 范围 [0, 1]
    std_normal_noise: np.ndarray,  # 随机噪声 (3, H, W)
    hyperparams: np.ndarray   # 步长 (1,) 范围 [0.5, 1.5]
) -> np.ndarray:              # 返回: 新候选 (3, H, W)
    """生成新的对抗样本候选。"""
    ...
```

### 关键概念

1. **原始图像 (org_img)**: 要攻击的干净图像
2. **最佳对抗样本 (best_adv_img)**: 当前最佳对抗样本
3. **噪声 (std_normal_noise)**: 随机探索组件
4. **超参数 (Hyperparams)**: 自适应步长（找到对抗样本时增大）

### 策略指南

- **利用（Exploitation）**: 沿着从 org_img 到 best_adv_img 的方向移动
- **探索（Exploration）**: 添加噪声以发现新区域
- **自适应（Adaptive）**: 使用 hyperparams 控制步长
- **目标**: 找到更接近 org_img 的对抗样本（更小的 L2 距离）

## 用例

此任务可适用于各种对抗攻击场景：

1. **黑盒攻击**: 无梯度信息可用
2. **基于决策的攻击**: 仅分类输出可用
3. **迁移攻击**: 攻击不同的模型
4. **鲁棒性评估**: 测试模型防御

## 自定义

### 自定义评估逻辑

```python
from evotoolkit.task.python_task import AdversarialAttackTask

class CustomAttackTask(AdversarialAttackTask):
    def _evaluate_attack(self, draw_proposals_func):
        # 你的自定义攻击评估
        return avg_distance
```

### 不同的攻击预算

```python
task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=5000,  # 更多迭代
    n_test_samples=50,  # 更多测试样本
)
```

## 架构

```
src/evotool/task/python_task/
├── adversarial_attack/
│   ├── __init__.py
│   ├── adversarial_attack_task.py  # 任务实现
│   └── evo_attack.py               # 攻击算法
└── method_interface/               # 进化接口
    ├── evoengineer_interface.py
    ├── eoh_interface.py
    └── funsearch_interface.py
```

## 下一步

- 查看[教程](../../docs/tutorials/)了解更高级的用法
- 学习[自定义进化](../../docs/tutorials/customizing-evolution.zh.md)
- 探索其他任务类型（科学回归、CUDA、提示优化）

## 参考文献

- **L-AutoDA 论文**: [L-AutoDA: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks](https://doi.org/10.1145/3638530.3664121) (GECCO 2024)
- **Foolbox**: [https://github.com/bethgelab/foolbox](https://github.com/bethgelab/foolbox)
- **PyTorch Models**: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)

## 注意事项

- **Python 任务**: 对抗攻击是 Python 任务（进化 Python 函数）
- **函数签名**: 必须包含具有精确签名的 `draw_proposals`
- **模拟模式**: 适用于无需模型/数据的测试
- **真实评估**: 需要 PyTorch、Foolbox 和目标模型
