# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from jinja2 import Template

translation_template = Template("""
You are a CUDA engineer tasked with translating PyTorch code into CUDA kernel code.

The CUDA code you generate will be saved in `cuda_fname` and loaded using torch.utils.cpp_extension.load():
```python
cuda_fn = load(
name=task_name,
sources=[cuda_fname],
extra_cuda_cflags=["-O3", "--use_fast_math"],
with_cuda=True,
verbose=True,
)
```
Later, the function will be called via `cuda_fn = load(name=task_name, ...).forward` and thoroughly tested.

Translate the PyTorch code (in <pytorch> tags) into CUDA kernel code.

<instructions>
- Write CUDA code that performs the **exact same operation** as the PyTorch code.
- Include the required pybind11 cuda module name in the code.
- Return the code between <cuda></cuda> tags.
</instructions>

<pytorch>
```Python
{{code_to_translate}}
```
</pytorch>

========Output Example========
<cuda>
```C++
// Your CUDA kernel code here
```
</cuda>

Now, translate the PyTorch code into CUDA kernel code.
Return only CUDA kernel code, no other text

"""
)
translation_error_summary_template= Template("""
The above CUDA kernel code does not work as expected. Error message:

{{error_msg}}

Please provide a summary of this error. Return only the summary text and no other information.
""")

translation_error_template= Template("""
The above CUDA kernel code does not work as expected. Error message:

{{error_summary}}

Please provide the CORRECT CUDA kernel code.
Return only CUDA kernel code, no other text.
""")