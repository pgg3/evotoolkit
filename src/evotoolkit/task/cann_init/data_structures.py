# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CompileResult:
    success: bool
    error: Optional[str] = None
    project_path: Optional[str] = None
    op_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    kernel_src: Optional[str] = None
    full_code: Optional[Dict[str, str]] = None

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "success": self.success,
            "error": self.error,
            "project_path": self.project_path,
            "op_name": self.op_name,
            "kernel_src": self.kernel_src,
        }

        with open(save_path / "compile_result.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if self.full_code:
            with open(save_path / "full_code.json", "w") as f:
                json.dump(self.full_code, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CompileResult":
        load_path = Path(path)

        with open(load_path / "compile_result.json") as f:
            metadata = json.load(f)

        full_code = None
        full_code_path = load_path / "full_code.json"
        if full_code_path.exists():
            with open(full_code_path) as f:
                full_code = json.load(f)

        return cls(
            success=metadata["success"],
            error=metadata.get("error"),
            project_path=metadata.get("project_path"),
            op_name=metadata.get("op_name"),
            kernel_src=metadata.get("kernel_src"),
            full_code=full_code,
            context={},
        )

    def is_loadable(self) -> bool:
        return self.success and self.project_path is not None


@dataclass
class CANNSolutionConfig:
    project_path: Optional[str] = None

    # LLM outputs - TilingData
    tiling_fields: Optional[List[Dict[str, str]]] = None

    # LLM outputs - Function bodies (new design: all logic in function body)
    tiling_func_body: Optional[str] = None      # TilingFunc complete body
    infer_shape_body: Optional[str] = None      # InferShape complete body

    # LLM outputs - Full source (alternative: LLM generates complete file)
    host_operator_src: Optional[str] = None
    kernel_src: Optional[str] = None
    python_bind_src: Optional[str] = None

    # Output allocation (for python_bind)
    output_alloc_code: Optional[str] = None

    # Execution control
    compile_only: bool = False
    load_from: Optional[str] = None
    skip_correctness: bool = False
    skip_performance: bool = False
    setup_only: bool = False
    build_only: bool = False
    save_compile_to: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CANNSolutionConfig":
        if not d:
            return cls()

        return cls(
            project_path=d.get("project_path"),
            tiling_fields=d.get("tiling_fields"),
            tiling_func_body=d.get("tiling_func_body"),
            infer_shape_body=d.get("infer_shape_body"),
            host_operator_src=d.get("host_operator_src"),
            kernel_src=d.get("kernel_src"),
            python_bind_src=d.get("python_bind_src"),
            output_alloc_code=d.get("output_alloc_code"),
            compile_only=d.get("compile_only", False),
            load_from=d.get("load_from"),
            skip_correctness=d.get("skip_correctness", False),
            skip_performance=d.get("skip_performance", False),
            setup_only=d.get("setup_only", False),
            build_only=d.get("build_only", False),
            save_compile_to=d.get("save_compile_to"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {}

        if self.project_path is not None:
            result["project_path"] = self.project_path
        if self.tiling_fields is not None:
            result["tiling_fields"] = self.tiling_fields
        if self.tiling_func_body is not None:
            result["tiling_func_body"] = self.tiling_func_body
        if self.infer_shape_body is not None:
            result["infer_shape_body"] = self.infer_shape_body
        if self.host_operator_src is not None:
            result["host_operator_src"] = self.host_operator_src
        if self.kernel_src is not None:
            result["kernel_src"] = self.kernel_src
        if self.python_bind_src is not None:
            result["python_bind_src"] = self.python_bind_src
        if self.output_alloc_code is not None:
            result["output_alloc_code"] = self.output_alloc_code
        if self.compile_only:
            result["compile_only"] = True
        if self.load_from is not None:
            result["load_from"] = self.load_from
        if self.skip_correctness:
            result["skip_correctness"] = True
        if self.skip_performance:
            result["skip_performance"] = True
        if self.setup_only:
            result["setup_only"] = True
        if self.build_only:
            result["build_only"] = True
        if self.save_compile_to is not None:
            result["save_compile_to"] = self.save_compile_to

        return result
