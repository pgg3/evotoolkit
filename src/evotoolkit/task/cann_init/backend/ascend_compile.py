# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Ascend C compilation pipeline.

Adapted from MultiKernelBench/utils/ascend_compile_pipeline.py
"""

import os
import subprocess
import shutil
from typing import Dict, Any, List

from ..pybind_templates import setup_pybind_directory


def underscore_to_pascalcase(underscore_str: str) -> str:
    """Convert underscore-separated string to PascalCase."""
    if not underscore_str:
        return ""
    parts = underscore_str.split("_")
    return "".join(word.capitalize() for word in parts if word)


def write_project_files(
    full_code: Dict[str, str],
    op_name: str,
    project_path: str,
) -> Dict[str, Any]:
    """
    Write all project files without compiling (for fake mode).

    Creates the same directory structure as ascend_compile but skips:
    - msopgen (no project skeleton)
    - build.sh (no compilation)
    - deploy (no installation)
    - pybind build (no wheel)

    Args:
        full_code: Dictionary containing all code components
        op_name: Operator name (e.g., "add")
        project_path: Base directory for operator projects

    Returns:
        {"success": bool, "error": str or None, "files_written": list}
    """
    op = f"{op_name}_custom"
    op_capital = underscore_to_pascalcase(op)
    target_directory = os.path.join(project_path, op_capital)
    files_written: List[str] = []

    try:
        # Create directory structure
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(os.path.join(target_directory, "op_host"), exist_ok=True)
        os.makedirs(os.path.join(target_directory, "op_kernel"), exist_ok=True)

        # Write project JSON
        json_path = os.path.join(project_path, f"{op}.json")
        with open(json_path, "w") as f:
            f.write(full_code.get("project_json_src", ""))
        files_written.append(json_path)

        # Write source files
        tiling_path = os.path.join(target_directory, "op_host", f"{op}_tiling.h")
        with open(tiling_path, "w") as f:
            f.write(full_code.get("host_tiling_src", ""))
        files_written.append(tiling_path)

        host_path = os.path.join(target_directory, "op_host", f"{op}.cpp")
        with open(host_path, "w") as f:
            f.write(full_code.get("host_operator_src", ""))
        files_written.append(host_path)

        kernel_path = os.path.join(target_directory, "op_kernel", f"{op}.cpp")
        with open(kernel_path, "w") as f:
            f.write(full_code.get("kernel_src", ""))
        files_written.append(kernel_path)

        # Set up Python binding directory
        cpp_ext_dir = setup_pybind_directory(project_path)
        csrc_dir = os.path.join(cpp_ext_dir, "csrc")

        pybind_path = os.path.join(csrc_dir, "op.cpp")
        with open(pybind_path, "w") as f:
            f.write(full_code.get("python_bind_src", ""))
        files_written.append(pybind_path)

        # Write model_src as reference
        model_path = os.path.join(project_path, "model_src.py")
        with open(model_path, "w") as f:
            f.write(full_code.get("model_src", ""))
        files_written.append(model_path)

        return {
            "success": True,
            "error": None,
            "files_written": files_written,
            "project_directory": target_directory,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to write files: {str(e)}",
            "files_written": files_written,
        }


def ascend_compile(
    full_code: Dict[str, str],
    op_name: str,
    project_path: str,
    device: str = "Ascend910B",
) -> Dict[str, Any]:
    """
    Compile Ascend C operator code.

    This function adapts the MultiKernelBench ascend_compile logic but takes
    full_code dict directly instead of executing generated code.

    Args:
        full_code: Dictionary containing all code components:
            - project_json_src
            - host_tiling_src
            - host_operator_src
            - kernel_src
            - python_bind_src
            - model_src
        op_name: Operator name (e.g., "add")
        project_path: Base directory for operator projects
        device: Target device (e.g., "Ascend910B")

    Returns:
        {"success": bool, "error": str or None, "context": dict}
    """
    op = f"{op_name}_custom"
    op_capital = underscore_to_pascalcase(op)
    target_directory = os.path.join(project_path, op_capital)
    original_cwd = os.getcwd()

    # Convert device name to msopgen compute unit format
    # e.g., "Ascend910B" -> "ai_core-Ascend910B2"
    # Note: Keep original case and add version suffix if not present
    if device.lower().startswith("ascend"):
        # Check if already has version suffix (e.g., Ascend910B2, Ascend910B3)
        if device[-1].isdigit() and device[-2].isalpha():
            compute_unit = f"ai_core-{device}"
        else:
            # Add default version suffix "2" for 910B series
            compute_unit = f"ai_core-{device}2"
    else:
        compute_unit = f"ai_core-{device}"

    # Context to store code and runtime objects
    context = {}

    try:
        # Step 1: Create operator project directory
        # Ensure project_path exists
        os.makedirs(project_path, exist_ok=True)

        if os.path.exists(target_directory):
            print("[INFO] Operator project already exists, deleting...")
            shutil.rmtree(target_directory)

        # Write project JSON
        json_path = os.path.join(project_path, f"{op}.json")
        with open(json_path, "w") as f:
            f.write(full_code.get("project_json_src", ""))

        # Step 2: Run msopgen to create project structure
        print("[INFO] Creating operator project with msopgen...")
        os.chdir(project_path)

        try:
            subprocess.run(
                [
                    "msopgen", "gen",
                    "-i", f"{op}.json",
                    "-c", compute_unit,
                    "-lan", "cpp",
                    "-out", op_capital,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            print("[INFO] Operator project created successfully")
        except subprocess.CalledProcessError as e:
            error_msg = f"msopgen failed:\nExit Code: {e.returncode}\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}"
            return {"success": False, "error": error_msg, "context": context}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "msopgen timed out", "context": context}
        finally:
            os.chdir(original_cwd)

        # Step 3: Write source files to project
        print("[INFO] Writing source files...")

        with open(os.path.join(target_directory, "op_host", f"{op}_tiling.h"), "w") as f:
            f.write(full_code.get("host_tiling_src", ""))

        with open(os.path.join(target_directory, "op_host", f"{op}.cpp"), "w") as f:
            f.write(full_code.get("host_operator_src", ""))

        with open(os.path.join(target_directory, "op_kernel", f"{op}.cpp"), "w") as f:
            f.write(full_code.get("kernel_src", ""))

        # Set up Python binding directory with built-in templates
        print("[INFO] Setting up Python binding environment...")
        cpp_ext_dir = setup_pybind_directory(project_path)
        csrc_dir = os.path.join(cpp_ext_dir, "csrc")
        with open(os.path.join(csrc_dir, "op.cpp"), "w") as f:
            f.write(full_code.get("python_bind_src", ""))

        # Step 4: Build the operator
        print("[INFO] Building operator...")
        os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
        os.chdir(target_directory)

        try:
            subprocess.run(
                ["./build.sh"],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            print("[INFO] Build succeeded")
        except subprocess.CalledProcessError as e:
            error_lines = []
            for line in (e.stdout + e.stderr).split("\n"):
                if "[ERROR]" in line or "error:" in line.lower():
                    error_lines.append(line)
            error_msg = f"Build failed:\nExit Code: {e.returncode}\nErrors:\n" + "\n".join(error_lines[:20])
            return {"success": False, "error": error_msg, "context": context}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Build timed out", "context": context}
        finally:
            os.chdir(original_cwd)

        # Step 5: Deploy the operator package
        print("[INFO] Deploying operator package...")
        os.chdir(os.path.join(target_directory, "build_out"))

        try:
            subprocess.run(
                ["./custom_opp_ubuntu_aarch64.run"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            print("[INFO] Deploy succeeded")
        except subprocess.CalledProcessError as e:
            error_msg = f"Deploy failed:\nExit Code: {e.returncode}\nOutput:\n{e.stdout}"
            return {"success": False, "error": error_msg, "context": context}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Deploy timed out", "context": context}
        finally:
            os.chdir(original_cwd)

        # Step 6: Build Python bindings
        print("[INFO] Building Python bindings...")
        os.chdir(cpp_ext_dir)

        try:
            subprocess.run(
                ["bash", "build_and_run.sh"],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            print("[INFO] Python binding succeeded")
        except subprocess.CalledProcessError as e:
            error_msg = f"Python binding failed:\nExit Code: {e.returncode}\nOutput:\n{e.stdout}"
            return {"success": False, "error": error_msg, "context": context}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Python binding timed out", "context": context}
        finally:
            os.chdir(original_cwd)

        # Step 7: Set environment variables
        custom_opp_path = os.path.join(project_path, "opp", "vendors", "customize")
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path

        if "opp/vendors/customize" not in os.environ.get("LD_LIBRARY_PATH", ""):
            custom_lib_path = os.path.join(custom_opp_path, "op_api", "lib")
            existing_path = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{custom_lib_path}:{existing_path}"

        # Step 8: Load model code into context
        print("[INFO] Loading model code...")
        try:
            model_src = full_code.get("model_src", "")
            compile(model_src, "<string>", "exec")
            exec(model_src, context)
        except Exception as e:
            return {"success": False, "error": f"Failed to load model: {str(e)}", "context": context}

        print("[INFO] Compilation pipeline completed successfully")
        return {"success": True, "error": None, "context": context}

    except Exception as e:
        os.chdir(original_cwd)
        return {"success": False, "error": f"Unexpected error: {str(e)}", "context": context}
