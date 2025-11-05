# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


"""
Basic Adversarial Attack Example

This example demonstrates how to use EvoToolkit to evolve adversarial attack
algorithms (draw_proposals functions) for fooling neural networks.

Requirements:
- pip install evotoolkit
- For real attacks: pip install evotoolkit[adversarial_attack]
  (or: pip install torch torchvision foolbox)
"""

from evotoolkit.task.python_task import (
    AdversarialAttackTask,
)


def main():
    print("=" * 60)
    print("Adversarial Attack Example")
    print("=" * 60)

    # Step 1: Create task (with mock mode for testing)
    print("\n[1/4] Creating adversarial attack task...")

    # Option 1: Use mock mode (no model needed, good for testing)
    task = AdversarialAttackTask(
        use_mock=True,  # Set to False to use real model
        attack_steps=1000,
        n_test_samples=10,
    )

    # Option 2: Use real model (uncomment and configure)
    # import torch
    # import torch.nn as nn
    # import timm
    # from torchvision import datasets, transforms
    #
    # # Load CIFAR-10 pretrained ResNet18 model (from Hugging Face Hub)
    # # This model achieves 94.98% accuracy on CIFAR-10
    # # CIFAR-10 ResNet18 uses modified architecture (3x3 conv1, removed maxpool)
    # base_model = timm.create_model("resnet18", num_classes=10, pretrained=False)
    # base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # base_model.maxpool = nn.Identity()
    #
    # # Load pretrained weights
    # base_model.load_state_dict(
    #     torch.hub.load_state_dict_from_url(
    #         "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
    #         map_location="cpu",
    #         file_name="resnet18_cifar10.pth"
    #     )
    # )
    # base_model.eval()
    #
    # # Create model wrapper with normalization
    # # Important: Foolbox expects inputs in [0, 1], so normalization must be inside the model
    # class NormalizedModel(nn.Module):
    #     def __init__(self, model, mean, std):
    #         super().__init__()
    #         self.model = model
    #         self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
    #         self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    #
    #     def forward(self, x):
    #         # x is in [0, 1], normalize it
    #         x_normalized = (x - self.mean) / self.std
    #         return self.model(x_normalized)
    #
    # model = NormalizedModel(base_model,
    #                         mean=[0.4914, 0.4822, 0.4465],
    #                         std=[0.2471, 0.2435, 0.2616])
    # model.eval()
    #
    # if torch.cuda.is_available():
    #     model.cuda()
    #
    # # Load CIFAR-10 test set (only ToTensor, no Normalize in transform)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Converts to [0, 1] range
    # ])
    # test_set = datasets.CIFAR10(
    #     root='./data',
    #     train=False,
    #     download=True,
    #     transform=transform
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=32,
    #     shuffle=False
    # )
    #
    # # Create task
    # task = AdversarialAttackTask(
    #     model=model,
    #     test_loader=test_loader,
    #     attack_steps=1000,
    #     n_test_samples=10,
    #     use_mock=False
    # )

    print(
        f"Task created: attack_steps={task.task_info['attack_steps']}, "
        f"n_samples={task.task_info['n_test_samples']}, "
        f"mock={task.task_info['use_mock']}"
    )

    # Step 2: Test initial solution
    print("\n[2/4] Testing initial solution...")

    init_sol = task.make_init_sol_wo_other_info()
    print("Initial draw_proposals function created")
    print(
        f"Initial score: {init_sol.evaluation_res.score:.2f} "
        f"(avg L2 distance: {init_sol.evaluation_res.additional_info['avg_distance']:.2f})"
    )

    # Step 3: Test a custom algorithm
    print("\n[3/4] Testing custom algorithm...")

    custom_code = '''import numpy as np

def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams):
    """
    Simple algorithm: move toward original image with noise.
    """
    # Flatten images
    org = org_img.flatten()
    best = best_adv_img.flatten()
    noise = std_normal_noise.flatten()

    # Move toward original with random perturbation
    direction = org - best
    step = hyperparams[0] * 0.1
    candidate = best + step * direction + step * noise * 0.5

    return candidate.reshape(org_img.shape)
'''

    result = task.evaluate_code(custom_code)
    print("Custom algorithm tested")
    print(
        f"Score: {result.score:.2f} "
        f"(avg L2 distance: {result.additional_info['avg_distance']:.2f})"
    )

    # Step 4: Show function structure
    print("\n[4/4] Understanding the draw_proposals function...")

    print("\nFunction signature:")
    print("  def draw_proposals(org_img, best_adv_img, std_normal_noise, hyperparams)")
    print("\nInputs:")
    print("  - org_img: Original clean image, shape (3, H, W)")
    print("  - best_adv_img: Current best adversarial, shape (3, H, W)")
    print("  - std_normal_noise: Random noise, shape (3, H, W)")
    print("  - hyperparams: Step size parameter, shape (1,)")
    print("\nOutput:")
    print("  - New candidate adversarial example, shape (3, H, W)")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey points:")
    print("- Solutions are Python FUNCTIONS (draw_proposals)")
    print("- Function generates adversarial perturbations")
    print("- Lower L2 distance = better attack")
    print("- Evolution optimizes the algorithm automatically")

    print("\nTo run evolution:")
    print("1. Configure LLM API credentials in the code")
    print("2. Set use_mock=False to use real model")
    print("3. Run evolution (example below):")
    print("""
    interface = EvoEngineerPythonInterface(task)
    llm_api = HttpsApi(
        api_url="api.openai.com",  # Your API URL
        key="your-api-key-here",   # Your API key
        model="gpt-4o"
    )
    result = evotoolkit.solve(
        interface=interface,
        output_path='./results',
        running_llm=llm_api,
        max_generations=10
    )
    """)


if __name__ == "__main__":
    main()
