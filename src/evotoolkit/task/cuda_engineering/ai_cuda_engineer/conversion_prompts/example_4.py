# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


EG_4 = f'''
--- Original code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the LeNet-5 model.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)

        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))

        # Final fully connected layer
        x = self.fc3(x)

        return x

# Test code for the LeNet-5 model
batch_size = 1
num_classes = 10

def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]
```

--- Functional code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    conv1_weight: nn.Parameter,
    conv1_bias: nn.Parameter,
    conv2_weight: nn.Parameter,
    conv2_bias: nn.Parameter,
    fc1_weight: nn.Parameter,
    fc1_bias: nn.Parameter,
    fc2_weight: nn.Parameter,
    fc2_bias: nn.Parameter,
    fc3_weight: nn.Parameter,
    fc3_bias: nn.Parameter,
) -> torch.Tensor:
    """
    Implements a LeNet-5 architecture with ReLU activation.

    Args:
        x (torch.Tensor): The input tensor, shape (batch_size, 1, 32, 32)
        conv1_weight (nn.Parameter): Parameters for first conv layer
        conv1_bias (nn.Parameter): Parameters for first conv layer
        conv2_weight (nn.Parameter): Parameters for second conv layer
        conv2_bias (nn.Parameter): Parameters for second conv layer
        fc1_weight (nn.Parameter): Parameters for first FC layer
        fc1_bias (nn.Parameter): Parameters for first FC layer
        fc2_weight (nn.Parameter): Parameters for second FC layer
        fc3_weight (nn.Parameter): Parameters for third FC layer
        fc3_bias (nn.Parameter): Parameters for third FC layer

    Returns:
        torch.Tensor: The output tensor, shape (batch_size, num_classes)
    """
    # First convolutional layer with ReLU activation and max pooling
    x = F.conv2d(x, conv1_weight, conv1_bias, stride=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Second convolutional layer with ReLU activation and max pooling
    x = F.conv2d(x, conv2_weight, conv2_bias, stride=1)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Flatten the output for the fully connected layers
    x = x.view(-1, 16 * 5 * 5)

    # First fully connected layer with ReLU activation
    x = F.linear(x, fc1_weight, fc1_bias)
    x = F.relu(x)

    # Second fully connected layer with ReLU activation
    x = F.linear(x, fc2_weight, fc2_bias)
    x = F.relu(x)

    # Final fully connected layer
    x = F.linear(x, fc3_weight, fc3_bias)

    return x


class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()

        # Extract parameters from convolutional layers
        conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv1_weight = nn.Parameter(conv1.weight.data.clone())
        self.conv1_bias = nn.Parameter(conv1.bias.data.clone())

        conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv2_weight = nn.Parameter(conv2.weight.data.clone())
        self.conv2_bias = nn.Parameter(conv2.bias.data.clone())

        # Extract parameters from fully connected layers
        fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc1_weight = nn.Parameter(fc1.weight.data.clone())
        self.fc1_bias = nn.Parameter(fc1.bias.data.clone())

        fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc2_weight = nn.Parameter(fc2.weight.data.clone())
        self.fc2_bias = nn.Parameter(fc2.bias.data.clone())

        fc3 = nn.Linear(in_features=84, out_features=num_classes)
        self.fc3_weight = nn.Parameter(fc3.weight.data.clone())
        self.fc3_bias = nn.Parameter(fc3.bias.data.clone())

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.conv1_weight,
            self.conv1_bias,
            self.conv2_weight,
            self.conv2_bias,
            self.fc1_weight,
            self.fc1_bias,
            self.fc2_weight,
            self.fc2_bias,
            self.fc3_weight,
            self.fc3_bias,
        )


# Test code for the LeNet-5 model
batch_size = 1
num_classes = 10


def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]


def get_init_inputs():
    return [num_classes]
```
'''