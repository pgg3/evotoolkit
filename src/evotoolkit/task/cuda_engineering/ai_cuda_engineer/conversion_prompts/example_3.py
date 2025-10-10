# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


EG_3 = f'''
--- Original code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()

        layers = []
        current_input_size = input_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)

# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [50, 50, 50, 50, 50, 50, 50, 50]  # Example of deep and narrow layers
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
```

--- Functional code ---
```Python
import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor, weights: nn.ParameterList, biases: nn.ParameterList
) -> torch.Tensor:
    """
    Implements a deep narrow multi-layer perceptron with ReLU activation.

    Args:
        x (torch.Tensor): The input tensor, shape (batch_size, input_size)
        weights (nn.ParameterList): A list of weight tensors for each linear layer
        biases (nn.ParameterList): A list of bias tensors for each linear layer

    Returns:
        torch.Tensor: The output tensor, shape (batch_size, output_size)
    """
    for weight, bias in zip(weights[:-1], biases[:-1]):
        x = F.linear(x, weight, bias)
        x = F.relu(x)
    x = F.linear(x, weights[-1], biases[-1])
    return x


class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            linear = nn.Linear(current_input_size, hidden_size)
            self.weights.append(nn.Parameter(linear.weight.data.clone()))
            self.biases.append(nn.Parameter(linear.bias.data.clone()))
            current_input_size = hidden_size

        linear = nn.Linear(current_input_size, output_size)
        self.weights.append(nn.Parameter(linear.weight.data.clone()))
        self.biases.append(nn.Parameter(linear.bias.data.clone()))

    def forward(self, x, fn=module_fn):
        return fn(x, self.weights, self.biases)


# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
]  # Example of deep and narrow layers
output_size = 10


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
```
'''