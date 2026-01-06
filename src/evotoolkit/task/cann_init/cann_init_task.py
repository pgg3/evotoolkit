# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import tempfile
from typing import Any, Dict, Optional

from evotoolkit.core import BaseTask, EvaluationResult, Solution

from .evaluator import AscendCEvaluator
from .templates import AscendCTemplateGenerator
from .signature_parser import OperatorSignatureParser
from .data_structures import CompileResult, CANNSolutionConfig


class CANNInitTask(BaseTask):
    def __init__(
        self,
        data: Dict[str, Any],
        project_path: Optional[str] = None,
        fake_mode: bool = False,
    ):
        self.default_project_path = project_path
        self.fake_mode = fake_mode
        self._parser = None
        self._template_gen = None
        super().__init__(data)

    def _process_data(self, data: Dict[str, Any]):
        self.op_name = data["op_name"]
        self.python_reference = data["python_reference"]
        self.npu_type = data.get("npu_type", "Ascend910B2")
        self.cann_version = data.get("cann_version", "8.0")

        self._parser = OperatorSignatureParser()
        self.signature = self._parser.parse(self.python_reference, self.op_name)
        self._template_gen = AscendCTemplateGenerator(self.signature)

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
"""

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        return EvaluationResult(
            valid=False,
            score=None,
            additional_info={
                "error": "CANNInitTask requires evaluate_solution() with other_info containing tiling_fields, tiling_func_body, infer_shape_body",
            },
        )

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        config = CANNSolutionConfig.from_dict(solution.other_info)
        kernel_src = solution.sol_string

        try:
            project_path = config.project_path or self.default_project_path
            if project_path is None:
                project_path = tempfile.mkdtemp(prefix=f"cann_{self.op_name}_")

            if config.load_from:
                evaluator = AscendCEvaluator(
                    project_path=project_path,
                    device=self.npu_type,
                )
                return self._evaluate_from_loaded(evaluator, config)

            if config.tiling_fields is None or config.tiling_func_body is None or config.infer_shape_body is None:
                return EvaluationResult(
                    valid=False,
                    score=None,
                    additional_info={
                        "stage": "validation",
                        "error": "Missing required fields: tiling_fields, tiling_func_body, infer_shape_body",
                        "kernel_src": kernel_src,
                    },
                )

            full_code = self._template_gen.generate(
                kernel_src=kernel_src,
                tiling_fields=config.tiling_fields,
                tiling_func_body=config.tiling_func_body,
                infer_shape_body=config.infer_shape_body,
                project_path=project_path,
                infer_dtype_body=config.infer_dtype_body,
                output_alloc_code=config.output_alloc_code,
            )

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

            if config.build_only:
                from .backend import ascend_build

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

            if config.save_compile_to:
                compile_result.save(config.save_compile_to)

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

            if not config.skip_performance:
                perf_result = evaluator.measure_performance(
                    self.op_name, python_reference=self.python_reference
                )
                runtime = perf_result.get("runtime")

                return EvaluationResult(
                    valid=True,
                    score=-runtime if runtime else 1.0,
                    additional_info={
                        "stage": "success",
                        "runtime": runtime,
                        "runtime_std": perf_result.get("std"),
                        "kernel_src": kernel_src,
                        "project_path": project_path,
                    },
                )

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
        import json
        import os
        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, "full_code.json"), "w") as f:
            json.dump(full_code, f)

    def _load_full_code(self, project_path: str) -> dict:
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
        try:
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

            evaluator.project_path = compile_result.project_path

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

            if not config.skip_performance:
                perf_result = evaluator.measure_performance(
                    self.op_name, python_reference=self.python_reference
                )
                runtime = perf_result.get("runtime")

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
        return Solution("")

    def cleanup(self):
        pass
