# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Joint Branch Prompts Package

This package contains prompts for the Kernel + Tiling Joint Branch,
organized into separate modules for better maintainability:

- chip_specs.py: Ascend chip specifications (UB size, core count, etc.)
- utils.py: Conversation extraction utilities
- tiling_prompts.py: Tiling agent prompts (propose, revise)
- kernel_prompts.py: Kernel agent prompts (review, re-review, final round)
- impl_prompts.py: Code implementation prompts (kernel, tiling)

Joint Branch Design Overview:
=============================

Why Joint Design?
1. Kernel depends on Tiling:
   - Kernel needs tiling params (blockSize, tileSize) to access data
   - Kernel loop structure is determined by tiling strategy

2. Tiling depends on Kernel:
   - Tiling strategy needs to know kernel's compute pattern
   - Different kernel implementations may need different tiling granularity

Three-Phase Flow:
  Phase 1: Joint Planning Discussion
      Tiling Agent <-> Kernel Agent -> Design Consensus
  Phase 2: Knowledge Retrieval
      get_api_doc(), get_operator_example()
  Phase 3: Code Implementation
      Kernel Agent -> kernel_src
      Tiling Agent -> tiling (or decide to use default)

Default Tiling Suitable For:
  - Add, Sub, Mul, Div (element-wise)
  - ReLU, Sigmoid, Tanh (activations)
  - Exp, Log, Sqrt (math functions)

Custom Tiling Required For:
  - MatMul: custom InferShape needed
  - Reduce: need to compute reduction dimension tiling
  - LayerNorm, Softmax: special tiling strategies
"""

from .chip_specs import (
    CHIP_SPECS,
    DEFAULT_CHIP,
    get_chip_spec,
    format_chip_spec,
)
from .utils import (
    extract_current_plan,
    extract_kernel_feedback,
    extract_kernel_design,
    extract_tiling_strategy,
)
from .tiling_prompts import TilingPromptsMixin
from .kernel_prompts import KernelPromptsMixin
from .impl_prompts import ImplPromptsMixin


class JointPromptMixin(TilingPromptsMixin, KernelPromptsMixin, ImplPromptsMixin):
    """Combined mixin for all Joint Branch prompts.

    Inherits from:
    - TilingPromptsMixin: get_tiling_propose_prompt, _get_tiling_revise_prompt
    - KernelPromptsMixin: get_kernel_review_prompt, _get_kernel_re_review_prompt,
                          _get_kernel_final_round_prompt
    - ImplPromptsMixin: get_kernel_impl_prompt, get_tiling_impl_prompt
    """
    pass


__all__ = [
    # Main class
    'JointPromptMixin',
    # Chip specs
    'CHIP_SPECS',
    'DEFAULT_CHIP',
    'get_chip_spec',
    'format_chip_spec',
    # Utils
    'extract_current_plan',
    'extract_kernel_feedback',
    'extract_kernel_design',
    'extract_tiling_strategy',
    # Individual mixins (for advanced use)
    'TilingPromptsMixin',
    'KernelPromptsMixin',
    'ImplPromptsMixin',
]
