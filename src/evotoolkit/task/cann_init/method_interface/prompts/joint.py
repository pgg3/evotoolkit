# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Joint Branch Prompts - Compatibility re-export

This module re-exports all Joint Branch prompts from the `joint/` subdirectory
for backwards compatibility. New code should import directly from the subdirectory.

The prompts are organized as follows:
- joint/chip_specs.py: Ascend chip specifications
- joint/utils.py: Conversation extraction utilities
- joint/tiling_prompts.py: Tiling agent prompts
- joint/kernel_prompts.py: Kernel agent prompts
- joint/impl_prompts.py: Code implementation prompts
- joint/__init__.py: Combined JointPromptMixin class
"""

# Re-export everything from the joint package
from .joint import (
    # Main class
    JointPromptMixin,
    # Chip specs
    CHIP_SPECS,
    DEFAULT_CHIP,
    get_chip_spec,
    format_chip_spec,
    # Utils
    extract_current_plan,
    extract_kernel_feedback,
    extract_kernel_design,
    extract_tiling_strategy,
    # Individual mixins
    TilingPromptsMixin,
    KernelPromptsMixin,
    ImplPromptsMixin,
)

__all__ = [
    'JointPromptMixin',
    'CHIP_SPECS',
    'DEFAULT_CHIP',
    'get_chip_spec',
    'format_chip_spec',
    'extract_current_plan',
    'extract_kernel_feedback',
    'extract_kernel_design',
    'extract_tiling_strategy',
    'TilingPromptsMixin',
    'KernelPromptsMixin',
    'ImplPromptsMixin',
]
