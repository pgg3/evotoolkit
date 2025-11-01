# 对抗攻击教程

学习如何使用 LLM 驱动的进化算法来发现有效的对抗攻击算法。

!!! note "学术引用"
    对抗攻击任务基于 L-AutoDA 研究。如果您在学术工作中使用此功能，请引用：

    ```bibtex
    @inproceedings{10.1145/3638530.3664121,
        author = {Guo, Ping and Liu, Fei and Lin, Xi and Zhao, Qingchuan and Zhang, Qingfu},
        title = {L-AutoDA: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks},
        year = {2024},
        isbn = {9798400704956},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3638530.3664121},
        doi = {10.1145/3638530.3664121},
        pages = {1846–1854},
        numpages = {9},
        keywords = {large language models, adversarial attacks, automated algorithm design, evolutionary algorithms},
        location = {Melbourne, VIC, Australia},
        series = {GECCO '24 Companion}
    }
    ```

!!! tip "完整示例代码"
    本教程提供了完整的可运行示例（点击查看/下载）：

    - [:material-download: basic_example.py](https://github.com/pgg3/evotoolkit/blob/master/examples/adversarial_attack/basic_example.py) - 使用模拟评估的基础用法
    - [:material-file-document: README.zh.md](https://github.com/pgg3/evotoolkit/blob/master/examples/adversarial_attack/README.zh.md) - 示例文档和使用指南

    本地运行：
    ```bash
    cd examples/adversarial_attack
    python basic_example.py
    ```

---

## 概述

本教程演示：

- 创建对抗攻击任务
- 使用 LLM 驱动的进化算法发现攻击算法
- 理解 `draw_proposals` 函数
- 在神经网络上评估攻击
- 自动进化有效的黑盒攻击

---

## 安装

!!! tip "推荐使用 GPU"
    为获得最佳性能，建议在安装 EvoToolkit 之前先安装支持 CUDA 的 PyTorch。
    我们推荐使用 **CUDA 12.9**（最新稳定版）。

### 步骤 1：安装 PyTorch（支持 GPU）

```bash
# CUDA 12.9（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 其他版本请访问：https://pytorch.org/get-started/locally/
# CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU（不推荐，性能较慢）
# pip install torch torchvision
```

### 步骤 2：安装 EvoToolkit

```bash
pip install evotoolkit[adversarial_attack]
```

这将安装：

- `timm` - PyTorch Image Models（提供来自 Hugging Face 的 CIFAR-10 预训练模型）
- `foolbox` - 对抗攻击库

**前置条件：**

- Python >= 3.11
- PyTorch >= 2.0（建议带 CUDA 支持）
- LLM API 访问权限（OpenAI、Claude 或其他兼容的提供商）
- 对抗机器学习的基础理解

---

## 理解对抗攻击任务

### 什么是对抗攻击任务？

对抗攻击任务进化 **提议生成算法** 来创建对抗样本，以最小的扰动欺骗神经网络。

| 方面         | 科学回归       | 对抗攻击             |
| ------------ | -------------- | -------------------- |
| **解类型**   | 数学方程       | 提议算法             |
| **函数名称** | `equation`     | `draw_proposals`     |
| **输入**     | 数据 + 参数    | 图像 + 噪声 + 超参数 |
| **评估**     | 预测的 MSE     | 对抗样本的 L2 距离   |
| **目标**     | 最小化预测误差 | 最小化扰动           |

### 任务组件

对抗攻击任务需要：

- **目标模型**：要攻击的神经网络
- **测试数据**：用于生成对抗样本的图像
- **攻击预算**：迭代/查询次数
- **评估指标**：原始图像与对抗图像之间的 L2 距离

---

## 创建您的第一个对抗攻击任务

### 步骤 1：加载目标模型和数据

```python
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms

# 加载 CIFAR-10 预训练的 ResNet18 模型（来自 Hugging Face Hub）
# 该模型在 CIFAR-10 上达到 94.98% 准确率
# CIFAR-10 的 ResNet18 使用修改过的架构（3x3 conv1, 移除 maxpool）
model = timm.create_model("resnet18", num_classes=10, pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# 加载预训练权重
model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet18_cifar10.pth"
    )
)
model.eval()

if torch.cuda.is_available():
    model.cuda()

# 加载 CIFAR-10 测试集
# 使用 CIFAR-10 标准的归一化参数
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2471, 0.2435, 0.2616])
])
test_set = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)
```

### 步骤 2：创建任务并测试初始解

```python
from evotoolkit.task.python_task import AdversarialAttackTask

# 创建任务
task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=1000,
    n_test_samples=10,
    use_mock=False
)

# 获取初始解
init_sol = task.make_init_sol_wo_other_info()

print(f"初始算法:")
print(init_sol.sol_string)
print(f"\n得分: {init_sol.evaluation_res.score:.2f}")
print(f"平均 L2 距离: {init_sol.evaluation_res.additional_info['avg_distance']:.2f}")
```

**输出：**
```
初始算法:
import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """基线提议生成..."""
    ...

得分: -2.34
平均 L2 距离: 2.34
```

### 步骤 3：测试自定义算法

```python
custom_code = '''import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """简单算法：带噪声地向原始图像移动。"""
    org = org_img.flatten()
    best = best_adv_img.flatten()
    noise = std_normal_noise.flatten()

    # 带随机扰动地向原始图像移动
    direction = org - best
    step = hyperparams[0] * 0.1
    candidate = best + step * direction + step * noise * 0.5

    return candidate.reshape(org_img.shape)
'''

result = task.evaluate_code(custom_code)
print(f"得分: {result.score:.2f}")
print(f"平均 L2 距离: {result.additional_info['avg_distance']:.2f}")
```

---

## 理解 draw_proposals 函数

### 函数签名

进化的函数必须具有以下确切签名：

```python
def draw_proposals(
    org_img: np.ndarray,         # 原始干净图像
    best_adv_img: np.ndarray,    # 当前最佳对抗样本
    std_normal_noise: np.ndarray,# 用于探索的随机噪声
    hyperparams: np.ndarray      # 自适应步长
) -> np.ndarray:                 # 新的候选对抗样本
    """生成新的候选对抗样本。"""
    ...
```

### 输入详情

**org_img**（原始图像）：
- 形状：RGB 图像为 `(3, H, W)`（例如，CIFAR-10 为 `(3, 32, 32)`）
- 值：`[0, 1]` 归一化的像素值
- 用途：我们正在攻击的干净图像

**best_adv_img**（最佳对抗样本）：
- 形状：`(3, H, W)` - 与 org_img 相同
- 值：`[0, 1]`
- 用途：当前最佳对抗样本（欺骗模型，最接近原始图像）

**std_normal_noise**（随机噪声）：
- 形状：`(3, H, W)` - 与 org_img 相同
- 值：从标准正态分布 N(0, 1) 采样
- 用途：为探索提供随机性

**hyperparams**（自适应参数）：
- 形状：`(1,)` - 单个标量值
- 值：通常在 `[0.5, 1.5]` 范围内
- 用途：找到对抗样本时增加的自适应步长

### 返回值

必须返回一个 numpy 数组，具有：
- 形状：`(3, H, W)` - 与 org_img 相同
- 值：任意（将自动裁剪到 `[0, 1]`）
- 用途：新的候选对抗样本

### 算法设计原则

**1. 利用（优化）**

沿从 org_img 到决策边界的方向移动：

```python
direction = org_img - best_adv_img
candidate = best_adv_img + step_size * direction
```

**2. 探索（发现）**

添加随机噪声以发现新区域：

```python
candidate = best_adv_img + noise_component
```

**3. 自适应步长**

使用 hyperparams 平衡探索/利用：

```python
# hyperparams 在找到对抗样本时增加
step = hyperparams[0] * base_step_size
```

**4. 完整示例**

```python
import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """结合平行和垂直分量。"""
    # 展平为向量
    org = org_img.flatten()
    best = best_adv_img.flatten()
    noise = std_normal_noise.flatten()

    # 计算方向
    direction = org - best
    direction_norm = np.linalg.norm(direction)

    # 平行分量（朝向原始图像）
    noise_norm = np.linalg.norm(noise)
    step_size = (noise_norm * hyperparams[0]) ** 2
    d_parallel = step_size * direction

    # 垂直分量（探索）
    if direction_norm > 1e-8:
        dot_product = np.dot(direction, noise)
        projection = (dot_product / direction_norm) * direction
        d_perpendicular = (projection / direction_norm - direction_norm * noise) * hyperparams[0]
    else:
        d_perpendicular = noise * hyperparams[0]

    # 组合
    candidate = best + d_parallel + d_perpendicular

    return candidate.reshape(org_img.shape)
```

---

## 运行进化以发现攻击

### 步骤 1：创建接口

```python
import evotoolkit
from evotoolkit.task.python_task import EvoEngineerPythonInterface
from evotoolkit.tools.llm import HttpsApi

# 创建接口
interface = EvoEngineerPythonInterface(task)
```

### 步骤 2：配置 LLM

```python
llm_api = HttpsApi(
    api_url="api.openai.com",  # 你的 API URL
    key="your-api-key-here",   # 你的 API 密钥
    model="gpt-4o"
)
```

### 步骤 3：运行进化

```python
result = evotoolkit.solve(
    interface=interface,
    output_path='./attack_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

print(f"找到的最佳算法:")
print(result.sol_string)
print(f"\n平均 L2 距离: {-result.evaluation_res.score:.2f}")
```

!!! tip "尝试不同的算法"
    EvoToolkit 支持多种进化算法用于对抗攻击：

    ```python
    # 使用 EoH
    from evotoolkit.task.python_task import EoHPythonInterface
    interface = EoHPythonInterface(task)

    # 使用 FunSearch
    from evotoolkit.task.python_task import FunSearchPythonInterface
    interface = FunSearchPythonInterface(task)

    # 使用 EvoEngineer（默认）
    from evotoolkit.task.python_task import EvoEngineerPythonInterface
    interface = EvoEngineerPythonInterface(task)
    ```

    然后使用相同的 `evotoolkit.solve()` 调用来运行进化。不同的接口可能会发现不同的攻击策略。

---

## 攻击进化示例

在进化过程中，LLM 发现越来越有效的算法：

**第 1 代：简单基线**
```python
def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    return best_adv_img + 0.01 * std_normal_noise
# 平均 L2: 3.5
```

**第 3 代：基于方向**
```python
def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    direction = org_img - best_adv_img
    return best_adv_img + hyperparams[0] * 0.1 * direction
# 平均 L2: 2.1
```

**第 7 代：复杂组合**
```python
def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    # 结合多个组件的复杂算法
    ...
# 平均 L2: 0.8
```

---

## 自定义进化行为

进化攻击的质量由 **进化方法** 及其内部的 **提示设计** 控制。要改进结果：

- **调整提示**：继承现有的 Interface 类并自定义 LLM 提示
- **开发新算法**：创建全新的进化策略

!!! tip "了解更多"
    这些是适用于所有任务的通用技术。有关详细教程，请参阅：

    - **[自定义进化方法](../customization/customizing-evolution.zh.md)** - 如何修改提示和开发新算法
    - **[高级用法](../advanced-overview.zh.md)** - 更多高级配置选项

**快速示例 - 对抗攻击的自定义提示：**

```python
from evotoolkit.task.python_task import EvoEngineerPythonInterface

class EvoEngineerCustomAttackInterface(EvoEngineerPythonInterface):
    """针对对抗攻击进化优化的接口。"""

    def get_operator_prompt(self, operator_name, selected_individuals,
                           current_best_sol, random_thoughts, **kwargs):
        """自定义变异提示以强调攻击有效性。"""

        if operator_name == "mutation":
            task_description = self.task.get_base_task_description()
            individual = selected_individuals[0]

            prompt = f"""# 对抗攻击算法进化

{task_description}

## 当前最佳算法
**平均 L2 距离:** {-current_best_sol.evaluation_res.score:.2f}
**算法:** {current_best_sol.sol_string}

## 要变异的算法
**平均 L2 距离:** {-individual.evaluation_res.score:.2f}
**算法:** {individual.sol_string}

## 优化指南
通过以下方式专注于改进算法：
- 更好地平衡利用（优化）和探索（发现）
- 更有效地使用自适应 hyperparams
- 巧妙组合方向向量和噪声
- 数值稳定性和效率

生成一个改进的 draw_proposals 函数，实现更低的 L2 距离。

## 响应格式：
name: [描述性名称]
code:
[您改进的 draw_proposals 函数]
thought: [更改的推理]
"""
            return [{"role": "user", "content": prompt}]

        # 对其他算子使用默认设置
        return super().get_operator_prompt(operator_name, selected_individuals,
                                          current_best_sol, random_thoughts, **kwargs)

# 使用自定义接口
interface = EvoEngineerCustomAttackInterface(task)
result = evotoolkit.solve(
    interface=interface,
    output_path='./custom_results',
    running_llm=llm_api,
    max_generations=10
)
```

---

## 理解评估

### 评分机制

1. **攻击执行**：在测试样本上运行进化的算法
2. **对抗样本生成**：使用 draw_proposals 创建对抗样本
3. **距离测量**：计算与原始图像的 L2 距离
4. **适应度计算**：得分 = -(平均 L2 距离)

更低的 L2 距离 = 更好的攻击 = 更高的得分（更不负）

### 评估输出

```python
result = task.evaluate_code(algorithm_code)

if result.valid:
    print(f"得分: {result.score:.2f}")  # 负 L2 距离
    print(f"平均 L2: {result.additional_info['avg_distance']:.2f}")
    print(f"攻击步数: {result.additional_info['attack_steps']}")
else:
    print(f"错误: {result.additional_info['error']}")
```

### 用于测试的模拟模式

使用模拟模式在不需要模型的情况下进行测试：

```python
# 模拟模式返回随机适应度用于测试
task = AdversarialAttackTask(
    use_mock=True,
    attack_steps=1000,
    n_test_samples=10
)

# 适用于：
# - 测试任务设置
# - 调试函数格式
# - 理解工作流程
# - 开发自定义接口
```

---

## 用例和应用

### 1. 黑盒攻击发现

在梯度不可用的黑盒场景中进化算法：

```python
task = AdversarialAttackTask(
    model=black_box_model,
    test_loader=test_loader,
    attack_steps=5000,  # 黑盒需要更多迭代
    n_test_samples=50
)
```

### 2. 鲁棒性评估

通过进化强攻击来测试模型防御：

```python
# 加载更鲁棒的模型（例如，对抗训练的模型）
# 注意：需要您自己训练或获取鲁棒模型
from torchvision import models
model = models.resnet50(pretrained=True)  # 或您自己的鲁棒模型
model.eval()

task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=10000,  # 彻底评估
    n_test_samples=100
)
```

!!! note "关于鲁棒模型"
    EvoToolkit 不再依赖 robustbench 库。如果您需要测试鲁棒模型，请：

    - 使用自己训练的对抗鲁棒模型
    - 从其他来源加载预训练的鲁棒模型
    - 或使用标准模型进行基础测试

### 3. 迁移攻击开发

进化可跨模型迁移的攻击：

```python
# 在替代模型上训练
from torchvision import models
surrogate_model = models.resnet18(pretrained=True)
surrogate_model.eval()

task = AdversarialAttackTask(
    model=surrogate_model,
    test_loader=test_loader,
    attack_steps=5000,
    n_test_samples=50
)

# 进化攻击
result = evotoolkit.solve(interface, ...)

# 在目标模型上测试
target_model = models.resnet50(pretrained=True)  # 不同架构
target_model.eval()
# 在 target_model 上评估进化的算法
```

### 4. 查询高效攻击

优化对目标模型的最少查询：

```python
task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=100,  # 有限的查询
    n_test_samples=20
)
```

---

## 完整示例

这是一个完整的工作示例：

```python
import torch
import torch.nn as nn
import timm
import evotoolkit
from torchvision import datasets, transforms
from evotoolkit.task.python_task import (
    AdversarialAttackTask,
    EvoEngineerPythonInterface
)
from evotoolkit.tools.llm import HttpsApi

# 1. 加载 CIFAR-10 预训练的 ResNet18 模型
model = timm.create_model("resnet18", num_classes=10, pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# 加载预训练权重
model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet18_cifar10.pth"
    )
)
model.eval()

if torch.cuda.is_available():
    model.cuda()

# 2. 准备测试数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2471, 0.2435, 0.2616])
])
test_set = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=32, shuffle=False
)

# 3. 创建任务
task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=1000,
    n_test_samples=10,
    use_mock=False
)

# 4. 配置 LLM API
llm_api = HttpsApi(
    api_url="api.openai.com",  # 你的 API URL
    key="your-api-key-here",   # 你的 API 密钥
    model="gpt-4o"
)

# 5. 创建接口
interface = EvoEngineerPythonInterface(task)

# 6. 运行进化
result = evotoolkit.solve(
    interface=interface,
    output_path='./attack_results',
    running_llm=llm_api,
    max_generations=10,
    pop_size=5,
    max_sample_nums=20
)

# 7. 显示结果
print(f"找到的最佳攻击算法:")
print(result.sol_string)
print(f"\n平均 L2 距离: {-result.evaluation_res.score:.2f}")
print(f"攻击步数: {result.evaluation_res.additional_info['attack_steps']}")
```

---

## 下一步

### 探索不同的攻击场景

- 尝试不同的目标模型（标准 vs 鲁棒）
- 实验不同的数据集（CIFAR-10、ImageNet）
- 比较不同的进化算法
- 在多个模型上测试进化的攻击

### 自定义和改进进化

- 检查现有 Interface 类中的提示设计
- 继承并重写 Interface 以自定义提示
- 为不同的攻击类型设计专门的提示
- 如有需要，开发新的进化算法

### 了解更多

- [自定义进化方法](../customization/customizing-evolution.zh.md) - 深入了解提示自定义
- [高级用法](../advanced-overview.zh.md) - 高级配置和技术
- [API 参考](../../api/index.md) - 完整的 API 文档
- [L-AutoDA 论文](https://doi.org/10.1145/3638530.3664121) - GECCO 2024

---

## 参考文献

- **L-AutoDA**: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks (GECCO 2024)
- **Foolbox**: A Python toolbox to create adversarial examples
- **PyTorch Models**: Pretrained computer vision models (https://pytorch.org/vision/stable/models.html)
