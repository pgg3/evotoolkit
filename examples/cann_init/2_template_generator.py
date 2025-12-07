# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Template Generator

Demonstrates two tiling modes:
1. Default mode - only kernel_src (for element-wise operators)
2. Full LLM mode - kernel_src + host_tiling_src + host_operator_src + python_bind_src

Usage:
    python 2_template_generator.py
"""

from pathlib import Path

from evotoolkit.task.cann_init import (
    OperatorSignatureParser,
    AscendCTemplateGenerator,
)
from _config import (
    PYTHON_REFERENCE,
    KERNEL_SRC,
    BLOCK_DIM,
    HOST_TILING_SRC,
    HOST_OPERATOR_SRC,
    PYTHON_BIND_SRC,
    ensure_output_dir,
)

# File mapping: component name -> relative path
FILE_MAPPING = {
    "project_json_src": "add_custom.json",
    "host_tiling_src": "op_host/add_custom_tiling.h",
    "host_operator_src": "op_host/add_custom.cpp",
    "kernel_src": "op_kernel/add_custom.cpp",
    "python_bind_src": "pybind/op.cpp",
    "model_src": "model_src.py",
}


def write_components(full_code: dict, output_dir: Path) -> list:
    """Write all components to files and return list of written files."""
    files_written = []
    for component, rel_path in FILE_MAPPING.items():
        file_path = output_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(full_code[component])
        files_written.append(str(file_path))
    return files_written


def test_mode(gen, mode_name: str, output_dir: Path, **generate_kwargs) -> bool:
    """Test a generation mode and write output to directory."""
    print(f"\n[{mode_name}]")

    try:
        full_code = gen.generate(**generate_kwargs)
        write_components(full_code, output_dir)
        print(f"  Generated {len(full_code)} components -> {output_dir}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def list_directory_tree(path: Path, prefix: str = ""):
    """Print directory tree."""
    entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            list_directory_tree(entry, prefix + extension)


def main():
    print("=" * 50)
    print("Template Generator Test")
    print("=" * 50)

    output_dir = ensure_output_dir("2_generator")

    # Parse signature
    parser = OperatorSignatureParser()
    signature = parser.parse(PYTHON_REFERENCE, "add")
    gen = AscendCTemplateGenerator(signature)

    # Test Default mode
    default_ok = test_mode(
        gen,
        "Default Mode",
        output_dir / "default_mode",
        kernel_src=KERNEL_SRC,
        block_dim=BLOCK_DIM,
    )

    # Test Full LLM mode
    full_llm_ok = test_mode(
        gen,
        "Full LLM Mode",
        output_dir / "full_llm_mode",
        kernel_src=KERNEL_SRC,
        host_tiling_src=HOST_TILING_SRC,
        host_operator_src=HOST_OPERATOR_SRC,
        python_bind_src=PYTHON_BIND_SRC,
    )

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Default Mode:  {'PASS' if default_ok else 'FAIL'}")
    print(f"Full LLM Mode: {'PASS' if full_llm_ok else 'FAIL'}")

    # List output directory
    print(f"\nOutput directory: {output_dir}")
    list_directory_tree(output_dir)


if __name__ == "__main__":
    main()
