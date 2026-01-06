# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Performance measurement for Ascend C operators.

Adapted from MultiKernelBench/utils/performance.py
"""

import torch
import numpy as np
from typing import Any, Dict


def measure_performance(
    context: Dict[str, Any],
    device: torch.device,
    synchronize,
    event_class,
    num_warmup: int = 3,
    num_trials: int = 100,
    eval_target: str = "ModelNew",
) -> Dict[str, Any]:
    """
    Measure operator performance using NPU events.

    Adapted from MultiKernelBench time_execution_event_template function.

    Args:
        context: Dictionary containing ModelNew, get_inputs, get_init_inputs
        device: Target device (e.g., torch.device('npu:0'))
        synchronize: Synchronization function (e.g., torch_npu.npu.synchronize)
        event_class: Event class (e.g., torch_npu.npu.Event)
        num_warmup: Number of warmup iterations
        num_trials: Number of measurement trials
        eval_target: Name of model class in context (default: "ModelNew")

    Returns:
        Dictionary with runtime statistics:
            - runtime: Mean execution time in ms
            - std: Standard deviation
            - min: Minimum time
            - max: Maximum time
            - num_trials: Number of trials
            - error: Error message if failed
    """
    get_inputs = context.get("get_inputs")
    get_init_inputs = context.get("get_init_inputs")
    ModelNew = context.get(eval_target)

    if not all([get_inputs, get_init_inputs, ModelNew]):
        return {"runtime": None, "error": "Missing required functions"}

    try:
        # Prepare inputs
        inputs = get_inputs()
        inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]

        init_inputs = get_init_inputs()
        init_inputs = [
            x.to(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]

        elapsed_times = []

        with torch.no_grad():
            custom_model = ModelNew(*init_inputs).to(device)

            # Warmup
            for _ in range(num_warmup):
                custom_model(*inputs)
                synchronize(device=device)

            # Measure
            for _ in range(num_trials):
                start_event = event_class(enable_timing=True)
                end_event = event_class(enable_timing=True)

                start_event.record()
                custom_model(*inputs)
                end_event.record()

                synchronize(device=device)
                elapsed_time_ms = start_event.elapsed_time(end_event)
                elapsed_times.append(elapsed_time_ms)

        return {
            "runtime": float(np.mean(elapsed_times)),
            "std": float(np.std(elapsed_times)),
            "min": float(np.min(elapsed_times)),
            "max": float(np.max(elapsed_times)),
            "num_trials": len(elapsed_times),
            "error": None,
        }

    except Exception as e:
        return {"runtime": None, "error": f"Performance measurement failed: {str(e)}"}
