# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Ascend Chip Specifications for Joint Branch Prompts

Note: UB buffer sizes are estimated values (official docs rarely disclose).
AI Core counts are from HAMi virtualization configs and official sources.

References:
- https://www.theriseunion.com/en/blog/HAMi-ascend-910b-support.html
- https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined
- https://zhuanlan.zhihu.com/p/599049070
"""

from typing import Dict, Any


CHIP_SPECS: Dict[str, Dict[str, Any]] = {
    "Ascend910A": {
        "ub_capacity": 256 * 1024,      # Estimated, not officially disclosed
        "ai_core_count": 30,            # 30 usable AI Cores (32 physical DaVinci Max)
        "total_cores": 32,
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "memory_gb": 32,                # 32 GB HBM
        "fp16_tflops": 256,
        "power_w": 310,
        "process": "7nm EUV (TSMC)",
        "description": "Training NPU (910A series)",
    },
    "Ascend910B2": {
        "ub_capacity": 256 * 1024,      # Estimated
        "ai_core_count": 24,            # Official: 24 AI Cores
        "ai_cpu_count": 6,
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "memory_gb": 64,                # 64 GB HBM2e
        "description": "Training NPU (910B2 series)",
    },
    "Ascend910B3": {
        "ub_capacity": 256 * 1024,      # Estimated
        "ai_core_count": 20,            # Official: 20 AI Cores
        "ai_cpu_count": 7,
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "memory_gb": 64,                # 64 GB HBM3e
        "memory_bandwidth": "1.2 TB/s",
        "description": "Training NPU (910B3 series)",
    },
    "Ascend310": {
        "ub_capacity": 128 * 1024,      # Estimated
        "ai_core_count": 1,             # Single AI Core (edge inference)
        "arm_cpu_cores": 8,             # 8x ARM A55 CPU
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "int8_tops": 16,
        "fp16_tflops": 8,
        "power_w": 8,
        "process": "12nm",
        "description": "Edge Inference NPU",
    },
    "Ascend310P": {
        "ub_capacity": 128 * 1024,      # Estimated
        "ai_core_count": 8,             # Estimated, not officially disclosed
        "vector_align": 32,
        "cube_m": 16,
        "cube_n": 16,
        "cube_k": 16,
        "description": "Inference NPU (310P series)",
    },
}

# Aliases for common usage
CHIP_SPECS["Ascend910B"] = CHIP_SPECS["Ascend910B3"]  # Default 910B -> 910B3

DEFAULT_CHIP = "Ascend910B3"


def get_chip_spec(npu_type: str) -> Dict[str, Any]:
    """Get chip specification by NPU type."""
    return CHIP_SPECS.get(npu_type, CHIP_SPECS[DEFAULT_CHIP])


def format_chip_spec(npu_type: str) -> str:
    """Format chip specification for prompt."""
    spec = get_chip_spec(npu_type)
    ub_kb = spec['ub_capacity'] // 1024
    core_count = spec.get('ai_core_count', 'unknown')
    mem_info = f", {spec['memory_gb']}GB HBM" if 'memory_gb' in spec else ""
    return f"""- Chip: {npu_type}
- UB Capacity: {ub_kb}KB per core (estimated)
- AI Core Count: {core_count}{mem_info}
- Vector Alignment: {spec['vector_align']} bytes
- Cube Tile Size: {spec['cube_m']}x{spec['cube_n']}x{spec['cube_k']}"""
