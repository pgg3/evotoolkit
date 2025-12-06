# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
CANN Init task for Ascend C operator generation.

This module provides a task class for evaluating Ascend C kernel code,
following the same design pattern as CudaTask.

Key Design:
- evaluate_code(kernel_src: str): Simple interface, only kernel code needed
- evaluate_solution(solution): Rich interface, extra config via other_info
- Template generation is internal, transparent to caller
"""

import tempfile
from typing import Any, Dict, Optional

from evotoolkit.core import BaseTask, EvaluationResult, Solution

from .evaluator import AscendCEvaluator
from .templates import AscendCTemplateGenerator
from .signature_parser import OperatorSignatureParser


class CANNInitTask(BaseTask):
    """
    Ascend C operator generation and evaluation task.

    Similar to CudaTask:
    - Input: kernel_src (str) - LLM only needs to generate kernel code
    - Other components (host, tiling, binding) are auto-generated from templates

    Usage:
        task = CANNInitTask({
            "op_name": "add",
            "python_reference": "def add(x, y): return x + y",
            "npu_type": "Ascend910B",
        })

        # Simple: just kernel code
        result = task.evaluate_code(kernel_src)

        # Rich: kernel + extra config
        solution = Solution(
            sol_string=kernel_src,
            other_info={"block_dim": 8, "tiling_fields": [...]}
        )
        result = task.evaluate_solution(solution)
    """

    def __init__(
        self,
        data: Dict[str, Any],
        project_path: Optional[str] = None,
        fake_mode: bool = False,
    ):
        """
        Initialize the CANN Init task.

        Args:
            data: Task data containing:
                - op_name: Operator name (e.g., "add", "layer_norm")
                - python_reference: Python reference implementation
                - npu_type: NPU model (default: "Ascend910B")
                - cann_version: CANN version (default: "8.0")
            project_path: Directory for operator project files
            fake_mode: Skip actual evaluation (for testing)
        """
        self.project_path = project_path or tempfile.mkdtemp()
        self.fake_mode = fake_mode

        # Initialize components (will be fully set up after _process_data)
        self._parser = None
        self._template_gen = None
        self._evaluator = None

        super().__init__(data)

    def _process_data(self, data: Dict[str, Any]):
        """Process input data and initialize components."""
        self.op_name = data["op_name"]
        self.python_reference = data["python_reference"]
        self.npu_type = data.get("npu_type", "Ascend910B")
        self.cann_version = data.get("cann_version", "8.0")

        # Parse Python reference to extract operator signature
        self._parser = OperatorSignatureParser()
        self.signature = self._parser.parse(self.python_reference, self.op_name)

        # Initialize template generator with signature
        self._template_gen = AscendCTemplateGenerator(self.signature)

        # Initialize evaluator
        self._evaluator = AscendCEvaluator(
            project_path=self.project_path,
            device=self.npu_type,
        )

        # Store task info for compatibility
        self.task_info = {
            "op_name": self.op_name,
            "python_reference": self.python_reference,
            "npu_type": self.npu_type,
            "cann_version": self.cann_version,
            "signature": self.signature,
        }

    def get_task_type(self) -> str:
        return "CANNInit"

    def get_base_task_description(self) -> str:
        return f"""You are an Ascend C operator development expert.
Your task is to implement the kernel code for the "{self.op_name}" operator.
Target device: {self.npu_type} NPU with CANN {self.cann_version}.

Python Reference:
```python
{self.python_reference}
```

Requirements:
1. Implement the kernel using Ascend C programming model
2. Ensure numerical correctness matches Python reference
3. Follow the vector/cube programming paradigm as appropriate
"""

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        """
        Evaluate kernel code using default template configuration.

        This is the simple interface - just provide kernel code,
        other components use default templates based on signature.

        Args:
            candidate_code: Ascend C kernel source code

        Returns:
            EvaluationResult with valid, score, and additional_info
        """
        solution = Solution(sol_string=candidate_code)
        return self.evaluate_solution(solution)

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        """
        Evaluate solution with optional extra configuration.

        The solution can carry additional config in other_info:
        - block_dim: Number of parallel cores (default: 8)
        - tiling_fields: Custom tiling field definitions
        - tiling_func_body: Custom TilingFunc body

        Args:
            solution: Solution object with kernel code and optional config

        Returns:
            EvaluationResult with valid, score, and additional_info
        """
        if self.fake_mode:
            return EvaluationResult(
                valid=True,
                score=1.0,
                additional_info={"fake_mode": True, "kernel_src": solution.sol_string},
            )

        kernel_src = solution.sol_string
        other_info = solution.other_info or {}

        try:
            # Step 1: Generate full code from kernel + templates
            full_code = self._template_gen.generate(
                kernel_src=kernel_src,
                block_dim=other_info.get("block_dim", 8),
                tiling_fields=other_info.get("tiling_fields"),
                tiling_func_body=other_info.get("tiling_func_body"),
            )

            # Step 2: Compile and deploy (combined in ascend_compile)
            compile_result = self._evaluator.compile(full_code, self.op_name)
            if not compile_result["success"]:
                return EvaluationResult(
                    valid=False,
                    score=None,
                    additional_info={
                        "stage": "compile",
                        "error": compile_result["error"],
                        "kernel_src": kernel_src,
                    },
                )

            # Step 3: Verify correctness
            verify_result = self._evaluator.verify_correctness(
                self.python_reference, self.op_name
            )
            if not verify_result["pass"]:
                return EvaluationResult(
                    valid=False,
                    score=None,
                    additional_info={
                        "stage": "correctness",
                        "error": verify_result["error"],
                        "python_output": verify_result.get("python_output"),
                        "ascend_output": verify_result.get("ascend_output"),
                        "max_diff": verify_result.get("max_diff"),
                        "kernel_src": kernel_src,
                    },
                )

            # Step 5: Measure performance
            perf_result = self._evaluator.measure_performance(self.op_name)
            runtime = perf_result.get("runtime")

            return EvaluationResult(
                valid=True,
                score=-runtime if runtime else 1.0,  # Negative runtime as score
                additional_info={
                    "stage": "success",
                    "runtime": runtime,
                    "runtime_std": perf_result.get("std"),
                    "kernel_src": kernel_src,
                },
            )

        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=None,
                additional_info={
                    "stage": "exception",
                    "error": str(e),
                    "kernel_src": kernel_src,
                },
            )

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial empty solution (generation task starts from scratch)."""
        return Solution("")

    def cleanup(self):
        """Clean up resources."""
        if self._evaluator:
            self._evaluator.cleanup()
