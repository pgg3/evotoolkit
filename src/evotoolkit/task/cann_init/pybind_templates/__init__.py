# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Python binding templates for Ascend C operators.

This module provides built-in templates for compiling custom operators
into Python-callable modules, eliminating the need for external CppExtension.
"""

import os
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent


def get_build_script() -> str:
    """Get the build_and_run.sh script content."""
    return '''#!/bin/bash
# Auto-generated build script for custom operator Python binding

BASE_DIR=$(pwd)
echo "Base directory: ${BASE_DIR}"

# Build wheel package
if ! python setup.py build bdist_wheel; then
    echo "Error: Failed to build wheel package"
    exit 1
fi

# Install wheel package
cd ${BASE_DIR}/dist
python -m pip install --no-index --no-deps --force-reinstall ./*.whl
'''


def get_setup_py(unique_id: str = "") -> str:
    """
    Get the setup.py content.

    Args:
        unique_id: Reserved for future parallel compilation isolation.
                   Currently not used to keep package name consistent.

    Returns:
        setup.py content string
    """
    # Use fixed package name to match model_src template
    # For parallel compilation, use different project_path instead
    pkg_name = "custom_ops"
    lib_name = "custom_ops_lib"

    return f'''import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"), recursive=True)

exts = []
ext = NpuExtension(
    name="{lib_name}",
    sources=source_files,
    extra_compile_args = [
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)
exts.append(ext)

setup(
    name="{pkg_name}",
    version='1.0',
    keywords='{pkg_name}',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={{"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)}},
)
'''


def generate_unique_id() -> str:
    """
    Generate a unique identifier for parallel compilation isolation.

    Uses tempfile to generate a unique suffix, ensuring no conflicts
    when the same operator is compiled in parallel.

    Returns:
        Unique identifier string (e.g., "a1b2c3d4")
    """
    import tempfile

    # Use tempfile to get a unique name
    with tempfile.NamedTemporaryFile(prefix="cann_", delete=True) as f:
        # Extract the unique part from the temp filename
        # e.g., "/tmp/cann_abc123" -> "abc123"
        unique_part = os.path.basename(f.name).replace("cann_", "")
    return unique_part


def setup_pybind_directory(target_dir: str, unique_id: str = "") -> str:
    """
    Set up the Python binding build directory with required files.

    Args:
        target_dir: Directory to create CppExtension structure in
        unique_id: Unique identifier for parallel compilation isolation.
                   If empty, one will be auto-generated.

    Returns:
        Path to the CppExtension directory
    """
    if not unique_id:
        unique_id = generate_unique_id()

    cpp_ext_dir = os.path.join(target_dir, "CppExtension")
    csrc_dir = os.path.join(cpp_ext_dir, "csrc")

    os.makedirs(csrc_dir, exist_ok=True)

    # Write build_and_run.sh
    build_script_path = os.path.join(cpp_ext_dir, "build_and_run.sh")
    with open(build_script_path, "w") as f:
        f.write(get_build_script())
    os.chmod(build_script_path, 0o755)

    # Write setup.py with unique package name
    setup_py_path = os.path.join(cpp_ext_dir, "setup.py")
    with open(setup_py_path, "w") as f:
        f.write(get_setup_py(unique_id))

    # Copy pytorch_npu_helper.hpp from bundled templates
    helper_hpp_src = TEMPLATES_DIR / "pytorch_npu_helper.hpp"
    helper_hpp_dst = os.path.join(csrc_dir, "pytorch_npu_helper.hpp")
    if helper_hpp_src.exists():
        import shutil
        shutil.copy(helper_hpp_src, helper_hpp_dst)
    else:
        raise FileNotFoundError(
            f"Required template file not found: {helper_hpp_src}. "
            "Please ensure pytorch_npu_helper.hpp is in the pybind_templates directory."
        )

    return cpp_ext_dir
