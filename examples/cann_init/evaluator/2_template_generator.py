# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from pathlib import Path

from evotoolkit.task.cann_init import (
    OperatorSignatureParser,
    AscendCTemplateGenerator,
)
from _config import (
    PYTHON_REFERENCE,
    KERNEL_SRC,
    TILING_FIELDS,
    TILING_FUNC_BODY,
    INFER_SHAPE_BODY,
    INFER_DTYPE_BODY,
    ensure_output_dir,
)

FILE_MAPPING = {
    "project_json_src": "add_custom.json",
    "host_tiling_src": "op_host/add_custom_tiling.h",
    "host_operator_src": "op_host/add_custom.cpp",
    "kernel_src": "op_kernel/add_custom.cpp",
    "python_bind_src": "pybind/op.cpp",
    "model_src": "model_src.py",
}


def write_components(full_code: dict, output_dir: Path) -> list:
    files_written = []
    for component, rel_path in FILE_MAPPING.items():
        file_path = output_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(full_code[component])
        files_written.append(str(file_path))
    return files_written


def main():
    print("=" * 50)
    print("Template Generator Test")
    print("=" * 50)

    output_dir = ensure_output_dir("2_generator")

    parser = OperatorSignatureParser()
    signature = parser.parse(PYTHON_REFERENCE, "add")
    gen = AscendCTemplateGenerator(signature)

    full_code = gen.generate(
        kernel_src=KERNEL_SRC,
        tiling_fields=TILING_FIELDS,
        tiling_func_body=TILING_FUNC_BODY,
        infer_shape_body=INFER_SHAPE_BODY,
        project_path=str(output_dir),
        infer_dtype_body=INFER_DTYPE_BODY,
    )

    write_components(full_code, output_dir)
    print(f"Generated {len(full_code)} components -> {output_dir}")

    print("\nGenerated files:")
    for component, rel_path in FILE_MAPPING.items():
        print(f"  {component}: {rel_path}")


if __name__ == "__main__":
    main()
