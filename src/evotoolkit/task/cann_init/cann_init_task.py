# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import tempfile
from typing import Any, Dict, Optional

from evotoolkit.core import BaseTask, EvaluationResult, Solution

from .evaluator import AscendCEvaluator
from .utils.templates import AscendCTemplateGenerator
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

    def _make_result(
        self,
        valid: bool,
        stage: str,
        score: Optional[float] = None,
        error: Optional[str] = None,
        **extra,
    ) -> EvaluationResult:
        """辅助方法：构造 EvaluationResult"""
        info = {"stage": stage}
        if error:
            info["error"] = error
        info.update(extra)
        return EvaluationResult(valid=valid, score=score, additional_info=info)

    def _run_verify_and_perf(
        self,
        evaluator: AscendCEvaluator,
        config: CANNSolutionConfig,
        kernel_src: str,
        project_path: str,
        extra_info: Optional[Dict] = None,
    ) -> EvaluationResult:
        """执行正确性验证和性能测量的公共逻辑"""
        base_info = {"kernel_src": kernel_src, "project_path": project_path}
        if extra_info:
            base_info.update(extra_info)

        if not config.skip_correctness:
            verify_result = evaluator.verify_correctness(self.python_reference, self.op_name)
            if not verify_result["pass"]:
                return self._make_result(
                    valid=False,
                    stage="correctness",
                    error=verify_result["error"],
                    python_output=verify_result.get("python_output"),
                    ascend_output=verify_result.get("ascend_output"),
                    max_diff=verify_result.get("max_diff"),
                    **base_info,
                )

        if not config.skip_performance:
            perf_result = evaluator.measure_performance(self.op_name, python_reference=self.python_reference)
            runtime = perf_result.get("runtime")

            if runtime is None:
                return self._make_result(
                    valid=False,
                    stage="performance",
                    error=perf_result.get("error", "Performance measurement failed"),
                    **base_info,
                )

            return self._make_result(
                valid=True,
                stage="success",
                score=-runtime,
                runtime=runtime,
                runtime_std=perf_result.get("std"),
                **base_info,
            )

        return self._make_result(valid=True, stage="correctness_only", **base_info)

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:  # noqa: ARG002
        return self._make_result(
            valid=False,
            stage="validation",
            error="CANNInitTask requires evaluate_solution() with other_info containing tiling_fields, tiling_func_body, infer_shape_body",
        )

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        config = CANNSolutionConfig.from_dict(solution.other_info)
        kernel_src = solution.sol_string

        try:
            project_path = config.project_path or self.default_project_path
            if project_path is None:
                project_path = tempfile.mkdtemp(prefix=f"cann_{self.op_name}_")

            # 从已保存结果加载
            if config.load_from:
                evaluator = AscendCEvaluator(project_path=project_path, device=self.npu_type)
                return self._evaluate_from_loaded(evaluator, config)

            # build_only 模式
            if config.build_only:
                return self._handle_build_only(config, project_path, kernel_src)

            # 验证必要字段 (6 个组件都必须提供)
            required_fields = [
                config.tiling_fields,
                config.tiling_func_body,
                config.infer_shape_body,
                config.output_alloc_code,
                config.kernel_impl,
                config.kernel_entry_body,
            ]
            if not all(required_fields):
                return self._make_result(
                    valid=False,
                    stage="validation",
                    error="Missing required fields: tiling_fields, tiling_func_body, infer_shape_body, output_alloc_code, kernel_impl, kernel_entry_body",
                    kernel_src=kernel_src,
                )

            # 生成完整代码
            full_code = self._template_gen.generate(
                kernel_impl=config.kernel_impl,
                kernel_entry_body=config.kernel_entry_body,
                tiling_fields=config.tiling_fields,
                tiling_func_body=config.tiling_func_body,
                infer_shape_body=config.infer_shape_body,
                project_path=project_path,
                output_alloc_code=config.output_alloc_code,
                tiling_func_includes=config.tiling_func_includes,
                kernel_includes=config.kernel_includes,
            )

            # fake_mode: 仅写入文件
            if self.fake_mode:
                return self._handle_fake_mode(full_code, project_path, kernel_src)

            # setup_only 模式
            if config.setup_only:
                return self._handle_setup_only(full_code, project_path, kernel_src)

            # 完整编译流程
            evaluator = AscendCEvaluator(project_path=project_path, device=self.npu_type)
            compile_result = evaluator.compile(full_code, self.op_name, project_path=project_path, kernel_src=kernel_src)

            if not compile_result.success:
                return self._make_result(
                    valid=False,
                    stage="compile",
                    error=compile_result.error,
                    kernel_src=kernel_src,
                    project_path=project_path,
                )

            if config.save_compile_to:
                compile_result.save(config.save_compile_to)

            if config.compile_only:
                return self._make_result(
                    valid=True,
                    stage="compile_only",
                    project_path=project_path,
                    kernel_src=kernel_src,
                    compile_result=compile_result,
                )

            return self._run_verify_and_perf(evaluator, config, kernel_src, project_path)

        except Exception as e:
            return self._make_result(
                valid=False,
                stage="exception",
                error=str(e),
                kernel_src=kernel_src,
            )

    def _handle_build_only(
        self, config: CANNSolutionConfig, project_path: str, _kernel_src: str
    ) -> EvaluationResult:
        """处理 build_only 模式"""
        from .utils.backend import ascend_build

        full_code = self._load_full_code(project_path)
        if full_code is None:
            return self._make_result(
                valid=False,
                stage="build",
                error="No full_code found. Run setup_only first.",
                project_path=project_path,
            )

        build_result = ascend_build(op_name=self.op_name, project_path=project_path, full_code=full_code)

        if not build_result["success"]:
            return self._make_result(
                valid=False,
                stage="build",
                error=build_result["error"],
                project_path=project_path,
                kernel_src=full_code.get("kernel_src", ""),
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

        return self._make_result(
            valid=True,
            stage="build_only",
            project_path=project_path,
            kernel_src=full_code.get("kernel_src", ""),
        )

    def _handle_fake_mode(
        self, full_code: Dict, project_path: str, kernel_src: str
    ) -> EvaluationResult:
        """处理 fake_mode: 仅写入文件不编译"""
        from .utils.backend import write_project_files

        write_result = write_project_files(full_code=full_code, op_name=self.op_name, project_path=project_path)

        if not write_result["success"]:
            return self._make_result(
                valid=False,
                stage="write_files",
                error=write_result["error"],
                fake_mode=True,
                project_path=project_path,
                kernel_src=kernel_src,
            )

        return self._make_result(
            valid=True,
            stage="files_written",
            score=1.0,
            fake_mode=True,
            project_path=project_path,
            kernel_src=kernel_src,
            generated_components=list(full_code.keys()),
            files_written=write_result.get("files_written", []),
        )

    def _handle_setup_only(
        self, full_code: Dict, project_path: str, kernel_src: str
    ) -> EvaluationResult:
        """处理 setup_only 模式"""
        from .utils.backend import ascend_setup

        setup_result = ascend_setup(
            full_code=full_code,
            op_name=self.op_name,
            project_path=project_path,
            device=self.npu_type,
        )

        if not setup_result["success"]:
            return self._make_result(
                valid=False,
                stage="setup",
                error=setup_result["error"],
                project_path=project_path,
                kernel_src=kernel_src,
            )

        self._save_full_code(project_path, full_code)

        return self._make_result(
            valid=True,
            stage="setup_only",
            project_path=project_path,
            kernel_src=kernel_src,
            target_directory=setup_result.get("target_directory"),
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
        """从已保存的编译结果加载并继续评估"""
        try:
            compile_result = CompileResult.load(config.load_from)

            if not compile_result.is_loadable():
                return self._make_result(
                    valid=False,
                    stage="load",
                    error="Loaded compile result is not usable",
                    load_from=config.load_from,
                )

            evaluator.project_path = compile_result.project_path

            if not evaluator.rebuild_context(compile_result):
                return self._make_result(
                    valid=False,
                    stage="load",
                    error="Failed to rebuild context from loaded result",
                    load_from=config.load_from,
                )

            return self._run_verify_and_perf(
                evaluator,
                config,
                compile_result.kernel_src,
                compile_result.project_path,
                extra_info={"load_from": config.load_from},
            )

        except Exception as e:
            return self._make_result(
                valid=False,
                stage="load_exception",
                error=str(e),
                load_from=config.load_from,
            )

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("")

    def cleanup(self):
        pass
