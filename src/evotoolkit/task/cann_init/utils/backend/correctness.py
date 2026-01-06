# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Correctness verification for Ascend C operators.

Adapted from MultiKernelBench/utils/correctness.py
"""

import torch
from typing import Any, Dict, Tuple


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def execute_correctness_check(
    context: Dict[str, Any],
    device: torch.device,
    synchronize,
    num_trials: int = 5,
    seed: int = 1024,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Execute correctness check comparing Model and ModelNew outputs.

    Adapted from MultiKernelBench execute_template function.

    Args:
        context: Dictionary containing Model, ModelNew, get_inputs, get_init_inputs
        device: Target device (e.g., torch.device('npu:0'))
        synchronize: Synchronization function (e.g., torch_npu.npu.synchronize)
        num_trials: Number of correctness trials
        seed: Random seed for reproducibility
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        Tuple of (passed: bool, error_message: str, info: dict)
    """
    info = {}

    # Get required functions from context
    get_inputs = context.get("get_inputs")
    get_init_inputs = context.get("get_init_inputs")
    Model = context.get("Model")
    ModelNew = context.get("ModelNew")

    if not all([get_inputs, get_init_inputs, Model, ModelNew]):
        return False, "Missing required functions: get_inputs, get_init_inputs, Model, ModelNew", info

    try:
        # Initialize models
        init_inputs = get_init_inputs()
        init_inputs = [
            x.to(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]

        with torch.no_grad():
            set_seed(seed)
            original_model = Model(*init_inputs).to(device)
            synchronize(device=device)

            set_seed(seed)
            custom_model = ModelNew(*init_inputs).to(device)
            synchronize(device=device)

        # Run correctness trials
        with torch.no_grad():
            for _ in range(num_trials):
                inputs = get_inputs()
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]

                synchronize(device=device)
                ref_output = original_model(*inputs)
                synchronize(device=device)

                new_output = custom_model(*inputs)
                synchronize(device=device)

                # Check shape
                if ref_output.shape != new_output.shape:
                    error_msg = f"Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
                    info["python_output"] = str(ref_output.shape)
                    info["ascend_output"] = str(new_output.shape)
                    return False, error_msg, info

                # Check values
                if not torch.allclose(ref_output, new_output, atol=atol, rtol=rtol):
                    max_diff = torch.max(torch.abs(ref_output - new_output)).item()
                    error_msg = f"Output value mismatch (max_diff: {max_diff:.6f})"
                    info["max_diff"] = max_diff
                    return False, error_msg, info

        return True, "", info

    except Exception as e:
        return False, f"Runtime error during correctness check: {str(e)}", info
