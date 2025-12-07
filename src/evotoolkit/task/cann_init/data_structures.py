# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Data structures for CANN Init task.

This module provides dataclasses for:
- CompileResult: Compilation output that can be saved/loaded
- CANNSolutionConfig: Typed wrapper for Solution.other_info
"""

import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CompileResult:
    """
    Compilation result that can be persisted and loaded.

    This enables:
    - Parallel compilation: each solution gets independent compile result
    - Decoupled testing: load compiled artifacts without recompiling
    - Caching: save successful compilations for later use

    Attributes:
        success: Whether compilation succeeded
        error: Error message if failed
        project_path: Directory containing compiled artifacts
        op_name: Operator name (e.g., "add")
        context: Runtime context (Model, ModelNew classes, etc.)
                 Note: context is NOT serialized (contains live objects)
    """
    success: bool
    error: Optional[str] = None
    project_path: Optional[str] = None
    op_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Metadata for tracking
    kernel_src: Optional[str] = None
    full_code: Optional[Dict[str, str]] = None

    def save(self, path: str) -> None:
        """
        Save compilation result to disk.

        Saves metadata as JSON and context separately (if needed).
        The project_path directory already contains compiled binaries.

        Args:
            path: Directory to save result metadata
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save metadata (excluding context which has live objects)
        metadata = {
            "success": self.success,
            "error": self.error,
            "project_path": self.project_path,
            "op_name": self.op_name,
            "kernel_src": self.kernel_src,
        }

        with open(save_path / "compile_result.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save full_code if present (for debugging/recompilation)
        if self.full_code:
            with open(save_path / "full_code.json", "w") as f:
                json.dump(self.full_code, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CompileResult":
        """
        Load compilation result from disk.

        Note: context must be rebuilt by re-executing model_src.

        Args:
            path: Directory containing saved result

        Returns:
            CompileResult with metadata (context empty, needs rebuild)
        """
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
            context={},  # Must be rebuilt
        )

    def is_loadable(self) -> bool:
        """Check if this result can be used for testing."""
        return self.success and self.project_path is not None


@dataclass
class CANNSolutionConfig:
    """
    Typed configuration for Solution.other_info.

    This provides a clean interface for passing dynamic configuration
    through Solution, making it portable across different task types.

    Tiling Modes (2 modes):
        1. Default mode: No host_tiling_src/host_operator_src → auto-generate defaults
        2. Full LLM mode: host_tiling_src + host_operator_src → use LLM-generated code

    Python Bind Modes (2 modes):
        1. Default mode: No python_bind_src → auto-generate with at::empty_like
        2. Full LLM mode: python_bind_src → use LLM-generated code

    Usage:
        # Create from other_info dict
        config = CANNSolutionConfig.from_dict(solution.other_info)

        # Default mode (element-wise operators)
        config = CANNSolutionConfig(block_dim=8)

        # Full LLM mode (complex operators)
        config = CANNSolutionConfig(
            host_tiling_src="...",      # Complete .h file
            host_operator_src="...",    # Complete .cpp file
            python_bind_src="...",      # Complete Python binding
        )

    Attributes:
        # Dynamic path
        project_path: Working directory for this solution

        # Default mode parameter
        block_dim: Number of parallel cores

        # Full LLM mode parameters (LLM generates complete code)
        host_tiling_src: Complete tiling header file
        host_operator_src: Complete host operator implementation
        python_bind_src: Complete Python binding code

        # Execution control
        compile_only: Stop after compilation (for parallel compile)
        load_from: Load pre-compiled result instead of compiling
        skip_correctness: Skip correctness check (already verified)
        skip_performance: Skip performance measurement

        # Save control
        save_compile_to: Save compilation result to this path
    """
    # Dynamic path
    project_path: Optional[str] = None

    # Default mode parameter
    block_dim: int = 8

    # Full LLM mode parameters
    host_tiling_src: Optional[str] = None
    host_operator_src: Optional[str] = None
    python_bind_src: Optional[str] = None

    # Execution control
    compile_only: bool = False
    load_from: Optional[str] = None
    skip_correctness: bool = False
    skip_performance: bool = False

    # Save control
    save_compile_to: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CANNSolutionConfig":
        """Create config from other_info dict (with defaults for missing keys)."""
        if not d:
            return cls()

        return cls(
            project_path=d.get("project_path"),
            block_dim=d.get("block_dim", 8),
            host_tiling_src=d.get("host_tiling_src"),
            host_operator_src=d.get("host_operator_src"),
            python_bind_src=d.get("python_bind_src"),
            compile_only=d.get("compile_only", False),
            load_from=d.get("load_from"),
            skip_correctness=d.get("skip_correctness", False),
            skip_performance=d.get("skip_performance", False),
            save_compile_to=d.get("save_compile_to"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for Solution.other_info."""
        result = {}

        if self.project_path is not None:
            result["project_path"] = self.project_path
        if self.block_dim != 8:
            result["block_dim"] = self.block_dim
        if self.host_tiling_src is not None:
            result["host_tiling_src"] = self.host_tiling_src
        if self.host_operator_src is not None:
            result["host_operator_src"] = self.host_operator_src
        if self.python_bind_src is not None:
            result["python_bind_src"] = self.python_bind_src
        if self.compile_only:
            result["compile_only"] = True
        if self.load_from is not None:
            result["load_from"] = self.load_from
        if self.skip_correctness:
            result["skip_correctness"] = True
        if self.skip_performance:
            result["skip_performance"] = True
        if self.save_compile_to is not None:
            result["save_compile_to"] = self.save_compile_to

        return result
