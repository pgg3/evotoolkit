# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Pybind independent branch for output shape inference code generation."""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..run_config import CANNIniterConfig
    from ..run_state_dict import CANNIniterRunStateDict


def _parse_response(response: str) -> str:
    """
    Parse pybind LLM response, extract code from <response> tags.

    Returns the function body code for shape inference.
    """
    # Extract content inside <response> tags
    response_match = re.search(r"<response>(.*?)</response>", response, re.DOTALL)
    if response_match:
        return response_match.group(1).strip()

    # Fallback: try to extract code block
    code_match = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Last resort: return as-is
    return response.strip()


class PybindBranch:
    """Pybind independent branch (simple context, can run in parallel)."""

    def __init__(self, config: "CANNIniterConfig", run_state_dict: "CANNIniterRunStateDict"):
        self.config = config
        self.run_state_dict = run_state_dict

    def _verbose(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.config.verbose:
            print(msg)

    def run(self):
        """Execute Pybind branch."""
        self._verbose("[Pybind] Starting...")

        signature = self.run_state_dict.signature

        if self.run_state_dict.strategies.get("pybind") == "default":
            # Use default shape inference: output shape = input shape
            self._verbose("[Pybind] Using default template (output shape = input shape)")
            shape_inference_code = "auto output_shape = x.sizes();"
        else:
            # Generate shape inference code with LLM
            self._verbose("[Pybind] Generating shape inference code with LLM...")
            prompt = self.config.interface.get_pybind_prompt(
                signature=signature,
                functionality=self.run_state_dict.functionality or "",
                shape_inference=self.run_state_dict.shape_inference or {},
                python_ref=self.config.task.python_reference or "",
            )
            response, _ = self.config.running_llm.get_response(prompt)
            shape_inference_code = _parse_response(response)
            self._verbose(f"[Pybind] Shape inference: {shape_inference_code}")

        # Save shape inference code for InferShape translation
        self.run_state_dict.shape_inference_code = shape_inference_code

        # Assemble complete pybind source code
        self.run_state_dict.pybind_src = self.config.interface.assemble_pybind_code(
            signature=signature,
            shape_inference_code=shape_inference_code,
        )
        self._verbose("[Pybind] Done")
