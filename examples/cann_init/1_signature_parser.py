# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Test: Signature Parser

Demonstrates how OperatorSignatureParser extracts operator signature
from Python reference code.

Usage:
    python 1_signature_parser.py
"""

from evotoolkit.task.cann_init import OperatorSignatureParser
from _config import PYTHON_REFERENCE


def main():
    print("=" * 50)
    print("Signature Parser Test")
    print("=" * 50)

    parser = OperatorSignatureParser()
    signature = parser.parse(PYTHON_REFERENCE, "add")

    print(f"Op Name: {signature['op_name']}")
    print(f"Inputs: {signature['inputs']}")
    print(f"Outputs: {signature['outputs']}")
    print(f"Init Params: {signature['init_params']}")
    print(f"Dtypes: {signature['dtypes']}")


if __name__ == "__main__":
    main()
