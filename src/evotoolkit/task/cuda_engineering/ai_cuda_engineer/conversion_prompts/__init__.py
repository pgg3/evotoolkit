# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from jinja2 import Template
from .sys_prompt import SYS_PROMPT
from .example_1 import EG_1
from .example_2 import EG_2
from .example_3 import EG_3
from .example_4 import EG_4

conversion_template = Template("""
{{sytem_prompt}}

=== Example 1 ===

{{example_1}}

=== Example 2 ===

{{example_2}}

=== Example 3 ===

{{example_3}}

=== Example 4 ===

{{example_4}}

=================
Here is the code you need to convert:

```python
{{code_to_convert}}
```

"""
)

conversion_error_template = Template("""
The above functional code does not work as expected. Error message:

{{error_msg}}

Please provide the correct functional code.
Your returned functional version of the code should be a valid python file, and it will be checked against the original code. Their outputs should be identical.
Return only python code, no other text.
""")

