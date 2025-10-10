# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from jinja2 import Template

rag_sys_prompt_template = Template("""
You are a Machine Learning Engineer trying to reduce the runtime of a kernel in CUDA. Make sure the kernel returns the correct result. Do not use any alternative precision that could result in an incorrect result. The kernel will be run on a {{gpu_type}} GPU with CUDA {{cuda_version}}.

Answer using the following schema:

name: A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
code: The proposed cuda script in code.
thought: The rationale for the improvement idea.

The pybind11 cuda module name has to be the same as in the example.
MAKE SURE THE PROPOSAL CODE IS VALID CUDA CODE.
FOLLOW EXACTLY THIS FORMAT. DO NOT ADD ANYTHING ELSE.

{% if optimization_history %}

Here are some examples of previous successful proposals for OTHER operations:

{% for entry in optimization_history %}

=== Proposal {{ loop.index }} ===

Python functional code: 
```Python
{{entry.task_info.func_py_code}}
```

name: {{ entry.name }}
code: 
```C++
{{ entry.best_kernel.cuda_code }}
```
thought: {{ entry.best_kernel.thought }}

runtime: {{ "%.5f"|format(entry.best_kernel.runtime) }} milliseconds
speedup over torch: {{ "%.2f"|format(func_runtime / entry.best_kernel.runtime) }}x

{% endfor %}
===================
You have made {{ optimization_history|length }} proposals so far.
{% endif %}
""")

evo_propose_template = Template("""
Propose a new CUDA kernel (including name, code, thought) which aims to reduce the runtime of the operation, while ensuring the kernel returns the correct result.
Here is the CUDA kernel code you need to optimize:

```C++
{{code}}
```

runtime of this code: {{ "%.5f"|format(runtime) }} milliseconds

speedup over torch of this code: {{ "%.2f"|format(func_runtime/runtime) }}x

profile_info of this code: 

{{profile_string}}

""")


