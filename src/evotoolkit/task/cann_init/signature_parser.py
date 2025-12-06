# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Operator signature parser for Python reference code.

This module extracts operator signature (inputs, outputs, dtypes) from
Python reference implementations using AST parsing.

Supports MultiKernelBench reference format:
- Model class with __init__ and forward methods
- get_inputs() and get_init_inputs() functions
"""

import ast
import re
from typing import Any, Dict, List, Optional


class OperatorSignatureParser:
    """
    Parse Python reference code to extract operator signature.

    Extracts:
    - Model.__init__ parameters (init_params)
    - Model.forward parameters (inputs)
    - Return values (outputs)
    - Type hints (if available)
    """

    def parse(self, python_code: str, op_name: str) -> Dict[str, Any]:
        """
        Parse Python code and extract operator signature.

        Args:
            python_code: Python reference implementation (MultiKernelBench format)
            op_name: Operator name (typically from filename, e.g., "elu", "add")

        Returns:
            Signature dict containing:
                - op_name: Operator name (passed in, not parsed)
                - inputs: List of forward() input info [{name, dtype}]
                - outputs: List of output info [{name, dtype}]
                - init_params: List of __init__() param info [{name, dtype, default}]
                - dtypes: Supported data types

        Note:
            In MultiKernelBench, op_name comes from the dataset/filename, not from
            parsing the code. The code only needs Model class with __init__/forward,
            plus get_inputs() and get_init_inputs() functions.
        """
        try:
            tree = ast.parse(python_code)
        except SyntaxError:
            # Fallback to regex parsing
            return self._parse_with_regex(python_code, op_name)

        # Find Model class and extract both __init__ and forward
        model_info = self._find_model_class(tree)

        if model_info is None:
            return self._parse_with_regex(python_code, op_name)

        return {
            "op_name": op_name,
            "inputs": model_info["inputs"],
            "outputs": model_info["outputs"],
            "init_params": model_info.get("init_params", []),
            "dtypes": model_info.get("dtypes", ["float16", "float32"]),
        }

    def _find_model_class(self, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """
        Find Model class and extract __init__ and forward info.

        Returns:
            Dict with inputs, outputs, init_params, dtypes
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                init_method = None
                forward_method = None

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__init__":
                            init_method = item
                        elif item.name == "forward":
                            forward_method = item

                if forward_method is None:
                    return None

                # Extract forward info (inputs/outputs)
                forward_info = self._extract_forward_info(forward_method)

                # Extract __init__ params
                init_params = []
                if init_method is not None:
                    init_params = self._extract_init_params(init_method)

                return {
                    "inputs": forward_info["inputs"],
                    "outputs": forward_info["outputs"],
                    "init_params": init_params,
                    "dtypes": forward_info.get("dtypes", ["float16", "float32"]),
                }

        return None

    def _find_main_function(self, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """
        Find the main function in AST (legacy, for backward compatibility).

        Priority:
        1. Model.forward method
        2. First function definition
        """
        # Look for Model class with forward method
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "forward":
                        return self._extract_func_info(item)

        # Look for standalone function
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                return self._extract_func_info(node)

        return None

    def _extract_init_params(self, init_method: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Extract __init__ parameters (excluding self).

        Returns list of {name, dtype, default} dicts.
        """
        params = []
        args = init_method.args

        # Get defaults - they align to the end of args
        num_defaults = len(args.defaults)
        num_args = len(args.args)

        for i, arg in enumerate(args.args):
            if arg.arg == "self":
                continue

            param_info = {
                "name": arg.arg,
                "dtype": self._extract_type_hint(arg.annotation),
            }

            # Check if this arg has a default value
            default_idx = i - (num_args - num_defaults)
            if default_idx >= 0 and default_idx < len(args.defaults):
                default_node = args.defaults[default_idx]
                param_info["default"] = self._extract_default_value(default_node)

            params.append(param_info)

        return params

    def _extract_default_value(self, node: ast.AST) -> Any:
        """Extract default value from AST node."""
        # ast.Constant covers all literal values in Python 3.8+
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id  # Return variable name as string
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers like -1.0
            inner = self._extract_default_value(node.operand)
            if isinstance(inner, (int, float)):
                return -inner
        return None

    def _extract_forward_info(self, func: ast.FunctionDef) -> Dict[str, Any]:
        """Extract forward method info (inputs and outputs)."""
        inputs = []

        # Extract parameters (skip 'self')
        for arg in func.args.args:
            if arg.arg == "self":
                continue
            dtype = self._extract_type_hint(arg.annotation)
            inputs.append({"name": arg.arg, "dtype": dtype})

        # Extract return type if available
        outputs = []
        if func.returns:
            return_type = self._extract_type_hint(func.returns)
            outputs.append({"name": "z", "dtype": return_type})
        else:
            outputs.append({"name": "z", "dtype": "float"})

        return {
            "inputs": inputs,
            "outputs": outputs,
            "dtypes": ["float16", "float32"],
        }

    def _extract_func_info(self, func: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function info from AST FunctionDef node."""
        inputs = []
        outputs = []

        # Get default argument names to skip
        num_defaults = len(func.args.defaults)
        num_args = len(func.args.args)
        default_arg_indices = set(range(num_args - num_defaults, num_args))

        # Extract parameters (skip 'self' and arguments with defaults)
        for i, arg in enumerate(func.args.args):
            arg_name = arg.arg
            if arg_name == "self":
                continue
            # Skip arguments with default values (like fn=module_fn)
            if i in default_arg_indices:
                continue

            dtype = self._extract_type_hint(arg.annotation)
            inputs.append({"name": arg_name, "dtype": dtype})

        # Extract return type if available
        if func.returns:
            return_type = self._extract_type_hint(func.returns)
            outputs.append({"name": "z", "dtype": return_type})
        else:
            # Default single output
            outputs.append({"name": "z", "dtype": "float"})

        # Try to infer dtypes from type hints
        dtypes = self._infer_dtypes(inputs, outputs)

        return {
            "func_name": func.name,
            "inputs": inputs,
            "outputs": outputs,
            "dtypes": dtypes,
        }

    def _extract_type_hint(self, annotation: Optional[ast.AST]) -> str:
        """Extract type hint string from AST annotation."""
        if annotation is None:
            return "float"

        if isinstance(annotation, ast.Name):
            name = annotation.id.lower()
            if "tensor" in name:
                return "float"
            return name

        if isinstance(annotation, ast.Subscript):
            # Handle generic types like Tensor[float32]
            return "float"

        if isinstance(annotation, ast.Attribute):
            # Handle torch.Tensor, np.ndarray, etc.
            return "float"

        return "float"

    def _parse_with_regex(self, python_code: str, op_name: str) -> Dict[str, Any]:
        """
        Fallback regex-based parsing for malformed Python code.

        Args:
            python_code: Python code string
            op_name: Operator name

        Returns:
            Basic signature dict
        """
        inputs = []

        # Try to find forward method or any function definition
        func_match = re.search(
            r"def\s+forward\s*\((.*?)\)", python_code, re.DOTALL
        )
        if not func_match:
            func_match = re.search(
                r"def\s+\w+\s*\((.*?)\)", python_code, re.DOTALL
            )

        if func_match:
            params_str = func_match.group(1)

            # Parse parameters
            params = [p.strip() for p in params_str.split(",") if p.strip()]
            for param in params:
                # Handle "self" parameter
                param_name = param.split(":")[0].strip().split("=")[0].strip()
                if param_name and param_name != "self":
                    inputs.append({"name": param_name, "dtype": "float"})
        else:
            # Default inputs if no function found
            inputs = [
                {"name": "x", "dtype": "float"},
                {"name": "y", "dtype": "float"},
            ]

        return {
            "op_name": op_name,
            "inputs": inputs,
            "outputs": [{"name": "z", "dtype": "float"}],
            "init_params": [],
            "dtypes": ["float16", "float32"],
        }
