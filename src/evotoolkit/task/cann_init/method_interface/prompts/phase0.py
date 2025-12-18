# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Phase 0: Compute pattern analysis prompt for Ascend C operator generation."""

from typing import Any, Dict, List


def _format_params(params: List[Dict], label: str) -> str:
    """Format parameter list for display."""
    if not params:
        return f"{label}: (none)"
    lines = [f"{label}:"]
    for p in params:
        tensor_mark = " [tensor]" if p.get("is_tensor") else ""
        lines.append(f"  - {p['name']}: {p['dtype']}{tensor_mark}")
    return "\n".join(lines)


def _format_signature(signature: Any) -> str:
    """Format operator signature for clear display in prompt."""
    if isinstance(signature, dict):
        sig = signature
    elif hasattr(signature, "to_dict"):
        sig = signature.to_dict()
    else:
        sig = {"op_name": "Unknown", "inputs": [], "outputs": [], "init_params": []}

    parts = [
        f"Operator: {sig.get('op_name', 'Unknown')}",
        _format_params(sig.get("inputs", []), "Inputs"),
        _format_params(sig.get("outputs", []), "Outputs"),
    ]

    init_params = sig.get("init_params", [])
    if init_params:
        parts.append(_format_params(init_params, "Init Params"))

    return "\n".join(parts)


class Phase0PromptMixin:
    """Phase 0: Shape analysis and strategy decision for Ascend C operator generation."""

    def get_pattern_analysis_prompt(self, python_ref: str, signature: Any) -> str:
        """
        Generate prompt for operator analysis.

        Returns a prompt that asks LLM to analyze the Python reference code
        and determine shape relationships and generation strategies for
        Ascend C operator implementation.
        """
        formatted_sig = _format_signature(signature)

        return f"""## Your Role

You are the **analysis agent** in a multi-agent Ascend C code generation pipeline.

Your job is to:
1. Analyze input/output shape relationship and provide shape inference formula
2. Decide which components need custom generation vs default templates
3. Provide a clear functionality description for downstream agents

**Important:** You do NOT generate any code. Downstream agents will read your output and generate the actual Ascend C code based on your analysis.

## Input

### Python Reference
```python
{python_ref}
```

### Operator Signature
{formatted_sig}

## Rules

### Strategy Decision

**Choose `default` when ALL of the following conditions are met:**
1. Input has exactly ONE tensor with NO additional scalar/attribute parameters
2. Output has exactly ONE tensor with the SAME shape as input
3. Data partitioning/tiling strategy does NOT need to change

**Examples of `default`:** relu, sigmoid, tanh, abs, sqrt, exp, log, add (element-wise), mul (element-wise)

**Choose `generate` for all other cases:**
- Multiple input/output tensors
- Output shape differs from input shape
- Requires reduction, aggregation, or shape transformation
- Data partitioning/tiling needs to be customized

**Note:** If you think data partitioning needs to change, use `generate` (but be aware this increases compilation failure risk)

**Examples of `generate`:** softmax, sum, matmul, conv2d, pooling, normalization, broadcast operations

**Strategy meanings:**
- `default`: Downstream agent will use pre-defined template (no custom code generation needed)
- `generate`: Downstream agent must generate custom code for this component

## Response Format

Respond inside `<response>` tags using the exact section headers:

<response>
## Shape Inference
input: <describe input shape, e.g., "[B, M, K]" or "[N]">
output: <describe output shape, e.g., "[B, M, N]" or "same as input">
formula: <C++ code to compute output_shape from input tensors>

## Strategies
kernel: generate
tiling: <default | generate>
pybind: <default | generate>

## Functionality
<1-2 sentences describing what this operator does mathematically>
</response>

## Examples

### Example 1: Element-wise (ReLU)
<response>
## Shape Inference
input: [*] (any shape)
output: same as input
formula: auto output_shape = x.sizes();

## Strategies
kernel: generate
tiling: default
pybind: default

## Functionality
Applies ReLU activation max(0, x) element-wise to the input tensor.
</response>

### Example 2: Reduction with shape preservation (Softmax)
<response>
## Shape Inference
input: [B, D] where B=batch, D=features
output: same as input (softmax preserves shape)
formula: auto output_shape = x.sizes();

## Strategies
kernel: generate
tiling: generate
pybind: generate

## Functionality
Applies softmax along dimension 1: exp(x_i) / sum(exp(x_j)), normalizing to probability distribution.
</response>

### Example 3: Shape transformation (MatMul)
<response>
## Shape Inference
input: a=[M, K], b=[K, N]
output: [M, N]
formula: auto output_shape = {{a.size(0), b.size(1)}};

## Strategies
kernel: generate
tiling: generate
pybind: generate

## Functionality
Performs matrix multiplication C = A @ B where A is [M,K] and B is [K,N].
</response>

Now analyze the given operator. Output ONLY the `<response>` block, nothing else:
"""
