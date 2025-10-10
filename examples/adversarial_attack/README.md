# Adversarial Attack Examples

English | [简体中文](README.zh.md)

This directory contains examples for evolving adversarial attack algorithms using LLM-driven evolution.

## Overview

The `AdversarialAttackTask` is a **Python task** that demonstrates how to use EvoToolkit to evolve the `draw_proposals` function for black-box adversarial attacks.

### Key Difference from Other Python Tasks

| Aspect | Scientific Regression | Adversarial Attack |
|--------|----------------------|---------------------|
| **Evolution target** | Mathematical equation | Proposal generation algorithm |
| **Function name** | `equation` | `draw_proposals` |
| **Inputs** | Data features + params | Images + noise + hyperparams |
| **Evaluation** | MSE on predictions | L2 distance of adversarial examples |
| **Goal** | Discover equations | Minimize distortion while fooling model |

## Quick Start

### Basic Example (Mock Mode)

Run the basic example with mock evaluation (no model needed):

```bash
python basic_example.py
```

**Output:**
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

### With Real Model

1. Install dependencies:
   ```bash
   # Step 1: Install PyTorch with GPU support (CUDA 12.9 recommended)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

   # Step 2: Install EvoToolkit
   pip install evotoolkit[adversarial_attack]
   ```

2. Configure LLM API credentials directly in your code:
   ```python
   llm_api = HttpsApi(
       api_url="api.openai.com",  # Your API URL
       key="your-api-key-here",   # Your API key
       model="gpt-4o"
   )
   ```

3. Run evolution (see `advanced_example.py`)

## Task Structure

### Creating a Task

```python
from evotoolkit.task.python_task import AdversarialAttackTask

# Option 1: Mock mode (no model needed)
task = AdversarialAttackTask(
    use_mock=True,
    attack_steps=1000,
    n_test_samples=10
)

# Option 2: With real model
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms

# Load CIFAR-10 pretrained ResNet18 (from Hugging Face Hub)
# CIFAR-10 ResNet18 uses modified architecture (3x3 conv1, removed maxpool)
model = timm.create_model("resnet18", num_classes=10, pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# Load pretrained weights
model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet18_cifar10.pth"
    )
)
model.eval()

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2471, 0.2435, 0.2616])
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

### Testing a Function

```python
code = '''
import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    # Your algorithm here
    ...
    return candidate
'''

result = task.evaluate_code(code)

print(f"Score: {result.score:.2f}")  # Negative L2 distance
print(f"Avg L2: {result.additional_info['avg_distance']:.2f}")
```

### Running Evolution

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

# Best algorithm found
print(result.sol_string)
print(f"Best L2 distance: {-result.evaluation_res.score:.2f}")
```

## Available Algorithms

```python
# EvoEngineer (recommended)
from evotoolkit.task.python_task import EvoEngineerPythonInterface
interface = EvoEngineerPythonInterface(task)

# EoH (Evolution of Heuristics)
from evotoolkit.task.python_task import EoHPythonInterface
interface = EoHPythonInterface(task)

# FunSearch
from evotoolkit.task.python_task import FunSearchPythonInterface
interface = FunSearchPythonInterface(task)
```

## Function Specification

### draw_proposals Function

```python
def draw_proposals(
    org_img: np.ndarray,      # Original image (3, H, W) in [0, 1]
    best_adv_img: np.ndarray, # Best adversarial (3, H, W) in [0, 1]
    std_normal_noise: np.ndarray,  # Random noise (3, H, W)
    hyperparams: np.ndarray   # Step size (1,) in [0.5, 1.5]
) -> np.ndarray:              # Returns: new candidate (3, H, W)
    """Generate new adversarial candidate."""
    ...
```

### Key Concepts

1. **Original image (org_img)**: The clean image to attack
2. **Best adversarial (best_adv_img)**: Current best adversarial example
3. **Noise (std_normal_noise)**: Random exploration component
4. **Hyperparams**: Adaptive step size (increases when finding adversarials)

### Strategy Guidelines

- **Exploitation**: Move along direction from org_img to best_adv_img
- **Exploration**: Add noise for discovering new regions
- **Adaptive**: Use hyperparams to control step size
- **Goal**: Find adversarials closer to org_img (smaller L2 distance)

## Use Cases

This task can be adapted for various adversarial attack scenarios:

1. **Black-box attacks**: No gradient information available
2. **Decision-based attacks**: Only classification output available
3. **Transfer attacks**: Attacking different models
4. **Robustness evaluation**: Testing model defenses

## Customization

### Custom Evaluation Logic

```python
from evotoolkit.task.python_task import AdversarialAttackTask

class CustomAttackTask(AdversarialAttackTask):
    def _evaluate_attack(self, draw_proposals_func):
        # Your custom attack evaluation
        return avg_distance
```

### Different Attack Budgets

```python
task = AdversarialAttackTask(
    model=model,
    test_loader=test_loader,
    attack_steps=5000,  # More iterations
    n_test_samples=50,  # More test samples
)
```

## Architecture

```
src/evotool/task/python_task/
├── adversarial_attack/
│   ├── __init__.py
│   ├── adversarial_attack_task.py  # Task implementation
│   └── evo_attack.py               # Attack algorithm
└── method_interface/               # Evolution interfaces
    ├── evoengineer_interface.py
    ├── eoh_interface.py
    └── funsearch_interface.py
```

## Next Steps

- Check out the [tutorials](../../docs/tutorials/) for more advanced usage
- Learn about [customizing evolution](../../docs/tutorials/customizing-evolution.md)
- Explore other task types (Scientific Regression, CUDA, Prompt Optimization)

## References

- **L-AutoDA Paper**: [L-AutoDA: Large Language Models for Automatically Evolving Decision-based Adversarial Attacks](https://doi.org/10.1145/3638530.3664121) (GECCO 2024)
- **Foolbox**: [https://github.com/bethgelab/foolbox](https://github.com/bethgelab/foolbox)
- **PyTorch Models**: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)

## Notes

- **Python Task**: Adversarial attack is a Python task (evolves Python functions)
- **Function Signature**: Must include `draw_proposals` with exact signature
- **Mock Mode**: Useful for testing without model/data
- **Real Evaluation**: Requires PyTorch, Foolbox, and a target model
