# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Phase 0: Signature parsing + Compute pattern analysis."""

import re
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..run_config import CANNIniterConfig
    from ..run_state_dict import CANNIniterRunStateDict


def _parse_response(response: str) -> Dict[str, Any]:
    """
    Parse Phase 0 LLM response with <response> tags and ## sections.

    Expected format:
        <response>
        ## Shape Inference
        input: [B, D]
        output: same as input
        formula: auto output_shape = x.sizes();

        ## Strategies
        kernel: generate
        tiling: default
        pybind: default

        ## Functionality
        Brief description of what this operator does.
        </response>

    Strategy values:
        - "default": Use pre-defined template, no LLM generation needed
        - "generate": LLM must generate custom code for this component
        - Any other value will fallback to "generate" for safety

    Returns:
        dict with keys: shape_inference, strategies, functionality
    """
    # Extract content inside <response> tags
    response_match = re.search(r"<response>(.*?)</response>", response, re.DOTALL)
    content = response_match.group(1) if response_match else response

    result = {
        "shape_inference": {
            "input": "",
            "output": "",
            "formula": "auto output_shape = x.sizes();",  # safe default
        },
        "strategies": {
            "kernel": "generate",
            "tiling": "generate",
            "pybind": "generate",
        },
        "functionality": "",
    }

    # Parse ## sections
    sections = re.split(r"^## ", content, flags=re.MULTILINE)

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split("\n", 1)
        header = lines[0].strip().lower()
        body = lines[1].strip() if len(lines) > 1 else ""

        if header == "shape inference":
            # Parse shape inference section
            for line in body.split("\n"):
                line = line.strip()
                if line.startswith("input:"):
                    result["shape_inference"]["input"] = line[6:].strip()
                elif line.startswith("output:"):
                    result["shape_inference"]["output"] = line[7:].strip()
                elif line.startswith("formula:"):
                    result["shape_inference"]["formula"] = line[8:].strip()

        elif header == "strategies":
            for line in body.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip().lower()
                    if key in ("kernel", "tiling", "pybind"):
                        # Only accept "default" or "generate", fallback to "generate"
                        result["strategies"][key] = value if value in ("default", "generate") else "generate"

        elif header == "functionality":
            result["functionality"] = body.strip()

    # kernel must always be "generate" regardless of LLM output
    result["strategies"]["kernel"] = "generate"

    return result


class Phase0Analyzer:
    """Phase 0: Signature parsing (deterministic) + Compute pattern analysis (LLM)."""

    def __init__(self, config: "CANNIniterConfig", run_state_dict: "CANNIniterRunStateDict"):
        self.config = config
        self.run_state_dict = run_state_dict

    def _verbose(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.config.verbose:
            print(msg)

    def analyze(self, op_name: str, python_ref: str):
        """
        Execute Phase 0 analysis.

        Args:
            op_name: Operator name
            python_ref: Python reference implementation code
        """
        # 0. Store op_name
        self.run_state_dict.op_name = op_name

        # 1. Signature parsing (reuse Evaluator's parser)
        self._verbose("Parsing signature...")
        self.run_state_dict.signature = self.config.task._parser.parse(python_ref, op_name)

        # 2. Shape and strategy analysis (LLM)
        self._verbose("Analyzing shape relationship and strategies with LLM...")
        prompt = self.config.interface.get_pattern_analysis_prompt(
            python_ref, self.run_state_dict.signature
        )
        response, _ = self.config.running_llm.get_response(prompt)
        result = _parse_response(response)

        self.run_state_dict.shape_inference = result["shape_inference"]
        self.run_state_dict.functionality = result["functionality"]
        self.run_state_dict.strategies = result["strategies"]

        self._verbose(f"Shape inference formula: {self.run_state_dict.shape_inference.get('formula', 'N/A')}")
        self._verbose(f"Functionality: {self.run_state_dict.functionality}")
        self._verbose(f"Strategies: {self.run_state_dict.strategies}")
