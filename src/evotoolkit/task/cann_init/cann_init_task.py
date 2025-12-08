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

v2 Changes:
- Dynamic project_path via Solution.other_info (not fixed at Task init)
- Supports compile_only mode for parallel compilation
- Supports load_from mode for decoupled testing
- Supports skip_correctness/skip_performance for staged evaluation
"""

import tempfile
from typing import Any, Dict, Optional

from evotoolkit.core import BaseTask, EvaluationResult, Solution

from .evaluator import AscendCEvaluator
from .templates import AscendCTemplateGenerator
from .signature_parser import OperatorSignatureParser
from .data_structures import CompileResult, CANNSolutionConfig


class CANNInitTask(BaseTask):
    """
    Ascend C operator generation and evaluation task.

    Similar to CudaTask:
    - Input: kernel_src (str) - LLM only needs to generate kernel code
    - Other components (host, tiling, binding) are auto-generated from templates

    v2 Design:
    - Task is lightweight: only holds static config (op_name, npu_type, etc.)
    - Dynamic config (project_path, compile_only, etc.) passed via Solution
    - Supports parallel compilation and decoupled testing

    Usage:
        # Simple: default temp directory
        task = CANNInitTask({
            "op_name": "add",
            "python_reference": PYTHON_REF,
        })
        result = task.evaluate_code(kernel_src)

        # Dynamic path via Solution
        solution = Solution(
            sol_string=kernel_src,
            other_info={"project_path": "/my/path"}
        )
        result = task.evaluate_solution(solution)

        # Compile only (for parallel compilation)
        solution = Solution(
            sol_string=kernel_src,
            other_info={
                "project_path": "/compile/sol_001",
                "compile_only": True,
                "save_compile_to": "/compile/sol_001",
            }
        )
        compile_result = task.evaluate_solution(solution)

        # Load and test (decoupled testing)
        solution = Solution(
            sol_string="",  # Not needed when loading
            other_info={
                "load_from": "/compile/sol_001",
            }
        )
        test_result = task.evaluate_solution(solution)
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
            project_path: Default directory for operator project files (optional)
                         Can be overridden per-solution via other_info["project_path"]
            fake_mode: Skip actual evaluation (for testing)
        """
        self.default_project_path = project_path
        self.fake_mode = fake_mode

        # Initialize components (will be fully set up after _process_data)
        self._parser = None
        self._template_gen = None

        super().__init__(data)

    def _process_data(self, data: Dict[str, Any]):
        """Process input data and initialize components."""
        self.op_name = data["op_name"]
        self.python_reference = data["python_reference"]
        self.npu_type = data.get("npu_type", "Ascend910B2")
        self.cann_version = data.get("cann_version", "8.0")

        # Parse Python reference to extract operator signature
        self._parser = OperatorSignatureParser()
        self.signature = self._parser.parse(self.python_reference, self.op_name)

        # Initialize template generator with signature
        self._template_gen = AscendCTemplateGenerator(self.signature)

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
        - project_path: Dynamic working directory (overrides default)
        - block_dim: Number of parallel cores (default: 8)
        - host_tiling_src: Complete tiling header (Full LLM mode)
        - host_operator_src: Complete host operator (Full LLM mode)
        - python_bind_src: Complete Python binding (Full LLM mode)
        - compile_only: Stop after compilation (for parallel compile)
        - setup_only: Only run msopgen + write files (for parallel compile phase 1)
        - build_only: Only run build.sh + deploy (for parallel compile phase 2)
        - load_from: Load pre-compiled result instead of compiling
        - skip_correctness: Skip correctness check
        - skip_performance: Skip performance measurement
        - save_compile_to: Save compilation result to this path

        Args:
            solution: Solution object with kernel code and optional config

        Returns:
            EvaluationResult with valid, score, and additional_info
        """
        # Parse configuration from other_info
        config = CANNSolutionConfig.from_dict(solution.other_info)
        kernel_src = solution.sol_string

        try:
            # Determine project path
            project_path = config.project_path or self.default_project_path
            if project_path is None:
                project_path = tempfile.mkdtemp(prefix=f"cann_{self.op_name}_")

            # Handle load_from mode FIRST (no code generation needed)
            if config.load_from:
                evaluator = AscendCEvaluator(
                    project_path=project_path,
                    device=self.npu_type,
                )
                return self._evaluate_from_loaded(evaluator, config)

            # Generate full code from kernel + templates (only if not load_from)
            full_code = self._template_gen.generate(
                kernel_src=kernel_src,
                block_dim=config.block_dim,
                host_tiling_src=config.host_tiling_src,
                host_operator_src=config.host_operator_src,
                python_bind_src=config.python_bind_src,
            )

            # Fake mode: write files but skip compile/deploy/test
            if self.fake_mode:
                from .backend import write_project_files

                write_result = write_project_files(
                    full_code=full_code,
                    op_name=self.op_name,
                    project_path=project_path,
                )

                if not write_result["success"]:
                    return EvaluationResult(
                        valid=False,
                        score=None,
                        additional_info={
                            "fake_mode": True,
                            "stage": "write_files",
                            "error": write_result["error"],
                            "project_path": project_path,
                            "kernel_src": kernel_src,
                        },
                    )

                return EvaluationResult(
                    valid=True,
                    score=1.0,
                    additional_info={
                        "fake_mode": True,
                        "stage": "files_written",
                        "project_path": project_path,
                        "kernel_src": kernel_src,
                        "generated_components": list(full_code.keys()),
                        "files_written": write_result.get("files_written", []),
                    },
                )

            # Handle setup_only mode (phase 1 of parallel compilation)
            if config.setup_only:
                from .backend import ascend_setup

                setup_result = ascend_setup(
                    full_code=full_code,
                    op_name=self.op_name,
                    project_path=project_path,
                    device=self.npu_type,
                )

                if not setup_result["success"]:
                    return EvaluationResult(
                        valid=False,
                        score=None,
                        additional_info={
                            "stage": "setup",
                            "error": setup_result["error"],
                            "project_path": project_path,
                            "kernel_src": kernel_src,
                        },
                    )

                # Save full_code for build phase
                self._save_full_code(project_path, full_code)

                return EvaluationResult(
                    valid=True,
                    score=None,
                    additional_info={
                        "stage": "setup_only",
                        "project_path": project_path,
                        "kernel_src": kernel_src,
                        "target_directory": setup_result.get("target_directory"),
                    },
                )

            # Handle build_only mode (phase 2 of parallel compilation)
            if config.build_only:
                from .backend import ascend_build

                # Load full_code from setup phase
                full_code = self._load_full_code(project_path)
                if full_code is None:
                    return EvaluationResult(
                        valid=False,
                        score=None,
                        additional_info={
                            "stage": "build",
                            "error": "No full_code found. Run setup_only first.",
                            "project_path": project_path,
                        },
                    )

                build_result = ascend_build(
                    op_name=self.op_name,
                    project_path=project_path,
                    full_code=full_code,
                )

                if not build_result["success"]:
                    return EvaluationResult(
                        valid=False,
                        score=None,
                        additional_info={
                            "stage": "build",
                            "error": build_result["error"],
                            "project_path": project_path,
                            "kernel_src": full_code.get("kernel_src", ""),
                        },
                    )

                # Save compile result if requested
                if config.save_compile_to:
                    compile_result = CompileResult(
                        success=True,
                        project_path=project_path,
                        op_name=self.op_name,
                        context=build_result.get("context", {}),
                        kernel_src=full_code.get("kernel_src", ""),
                        full_code=full_code,
                    )
                    compile_result.save(config.save_compile_to)

                return EvaluationResult(
                    valid=True,
                    score=None,
                    additional_info={
                        "stage": "build_only",
                        "project_path": project_path,
                        "kernel_src": full_code.get("kernel_src", ""),
                    },
                )

            # Full compilation (setup + build)
            evaluator = AscendCEvaluator(
                project_path=project_path,
                device=self.npu_type,
            )

            compile_result = evaluator.compile(
                full_code,
                self.op_name,
                project_path=project_path,
                kernel_src=kernel_src,
            )

            if not compile_result.success:
                return EvaluationResult(
                    valid=False,
                    score=None,
                    additional_info={
                        "stage": "compile",
                        "error": compile_result.error,
                        "kernel_src": kernel_src,
                        "project_path": project_path,
                    },
                )

            # Save compile result if requested
            if config.save_compile_to:
                compile_result.save(config.save_compile_to)

            # Handle compile_only mode (for parallel compilation)
            if config.compile_only:
                return EvaluationResult(
                    valid=True,
                    score=None,
                    additional_info={
                        "stage": "compile_only",
                        "project_path": project_path,
                        "kernel_src": kernel_src,
                        "compile_result": compile_result,
                    },
                )

            # Verify correctness (unless skipped)
            if not config.skip_correctness:
                verify_result = evaluator.verify_correctness(
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
                            "project_path": project_path,
                        },
                    )

            # Measure performance (unless skipped)
            if not config.skip_performance:
                perf_result = evaluator.measure_performance(
                    self.op_name, python_reference=self.python_reference
                )
                runtime = perf_result.get("runtime")

                return EvaluationResult(
                    valid=True,
                    score=-runtime if runtime else 1.0,  # Negative runtime as score
                    additional_info={
                        "stage": "success",
                        "runtime": runtime,
                        "runtime_std": perf_result.get("std"),
                        "kernel_src": kernel_src,
                        "project_path": project_path,
                    },
                )

            # Correctness passed but performance skipped
            return EvaluationResult(
                valid=True,
                score=None,
                additional_info={
                    "stage": "correctness_only",
                    "kernel_src": kernel_src,
                    "project_path": project_path,
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

    def _save_full_code(self, project_path: str, full_code: dict) -> None:
        """Save full_code to project directory for build phase."""
        import json
        import os
        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, "full_code.json"), "w") as f:
            json.dump(full_code, f)

    def _load_full_code(self, project_path: str) -> dict:
        """Load full_code from project directory."""
        import json
        import os
        path = os.path.join(project_path, "full_code.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _evaluate_from_loaded(
        self, evaluator: AscendCEvaluator, config: CANNSolutionConfig
    ) -> EvaluationResult:
        """
        Evaluate using a pre-compiled result.

        This enables decoupled testing: compile once, test multiple times
        or compile in parallel then test sequentially.

        Args:
            evaluator: AscendCEvaluator instance
            config: Configuration with load_from path

        Returns:
            EvaluationResult from testing the loaded compilation
        """
        try:
            # Load compile result
            compile_result = CompileResult.load(config.load_from)

            if not compile_result.is_loadable():
                return EvaluationResult(
                    valid=False,
                    score=None,
                    additional_info={
                        "stage": "load",
                        "error": "Loaded compile result is not usable",
                        "load_from": config.load_from,
                    },
                )

            # Update evaluator's project_path to match loaded result
            evaluator.project_path = compile_result.project_path

            # Rebuild context
            if not evaluator.rebuild_context(compile_result):
                return EvaluationResult(
                    valid=False,
                    score=None,
                    additional_info={
                        "stage": "load",
                        "error": "Failed to rebuild context from loaded result",
                        "load_from": config.load_from,
                    },
                )

            kernel_src = compile_result.kernel_src
            project_path = compile_result.project_path

            # Verify correctness (unless skipped)
            if not config.skip_correctness:
                verify_result = evaluator.verify_correctness(
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
                            "project_path": project_path,
                            "load_from": config.load_from,
                        },
                    )

            # Measure performance (unless skipped)
            if not config.skip_performance:
                perf_result = evaluator.measure_performance(
                    self.op_name, python_reference=self.python_reference
                )
                runtime = perf_result.get("runtime")

                # Check for errors
                if runtime is None:
                    return EvaluationResult(
                        valid=False,
                        score=None,
                        additional_info={
                            "stage": "performance",
                            "error": perf_result.get("error", "Performance measurement failed"),
                            "kernel_src": kernel_src,
                            "project_path": project_path,
                            "load_from": config.load_from,
                        },
                    )

                return EvaluationResult(
                    valid=True,
                    score=-runtime,
                    additional_info={
                        "stage": "success",
                        "runtime": runtime,
                        "runtime_std": perf_result.get("std"),
                        "kernel_src": kernel_src,
                        "project_path": project_path,
                        "load_from": config.load_from,
                    },
                )

            # Correctness passed but performance skipped
            return EvaluationResult(
                valid=True,
                score=None,
                additional_info={
                    "stage": "correctness_only",
                    "kernel_src": kernel_src,
                    "project_path": project_path,
                    "load_from": config.load_from,
                },
            )

        except Exception as e:
            return EvaluationResult(
                valid=False,
                score=None,
                additional_info={
                    "stage": "load_exception",
                    "error": str(e),
                    "load_from": config.load_from,
                },
            )

    def make_init_sol_wo_other_info(self) -> Solution:
        """Create initial empty solution (generation task starts from scratch)."""
        return Solution("")

    def cleanup(self):
        """Clean up resources (no-op in v2, each evaluate creates its own evaluator)."""
        pass

    # ============================================================
    # Utility methods for parallel compilation and batch testing
    # ============================================================

    def compile_only(
        self,
        kernel_src: str,
        project_path: str,
        save_to: Optional[str] = None,
        block_dim: int = 8,
        host_tiling_src: Optional[str] = None,
        host_operator_src: Optional[str] = None,
        python_bind_src: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Convenience method: compile only, no testing.

        Useful for parallel compilation scenarios.

        Args:
            kernel_src: Kernel source code
            project_path: Directory for compilation output
            save_to: Path to save CompileResult (default: same as project_path)
            block_dim: Number of parallel cores
            host_tiling_src: Complete tiling header (Full LLM mode)
            host_operator_src: Complete host operator (Full LLM mode)
            python_bind_src: Complete Python binding (Full LLM mode)

        Returns:
            EvaluationResult with compile status
        """
        config = CANNSolutionConfig(
            project_path=project_path,
            block_dim=block_dim,
            host_tiling_src=host_tiling_src,
            host_operator_src=host_operator_src,
            python_bind_src=python_bind_src,
            compile_only=True,
            save_compile_to=save_to or project_path,
        )
        solution = Solution(sol_string=kernel_src, other_info=config.to_dict())
        return self.evaluate_solution(solution)

    def test_compiled(
        self,
        load_from: str,
        skip_correctness: bool = False,
        skip_performance: bool = False,
    ) -> EvaluationResult:
        """
        Convenience method: test a pre-compiled result.

        Useful for decoupled testing scenarios.

        Args:
            load_from: Path to saved CompileResult
            skip_correctness: Skip correctness check
            skip_performance: Skip performance measurement

        Returns:
            EvaluationResult with test status
        """
        config = CANNSolutionConfig(
            load_from=load_from,
            skip_correctness=skip_correctness,
            skip_performance=skip_performance,
        )
        solution = Solution(sol_string="", other_info=config.to_dict())
        return self.evaluate_solution(solution)
