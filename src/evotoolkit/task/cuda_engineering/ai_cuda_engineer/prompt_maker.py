# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


from . import conversion_prompts, translation_prompts, evo_prompts, rag_prompts


class PromptMaker:
    @classmethod
    def make_convert_prompt(
        cls, code_to_convert: str, prev_response: str = None, error_msg: str = None
    ) -> list:
        """
        Convert original code to func code
        """
        base_prompt = conversion_prompts.conversion_template.render(
            sytem_prompt=conversion_prompts.SYS_PROMPT,
            example_1=conversion_prompts.EG_1,
            example_2=conversion_prompts.EG_2,
            example_3=conversion_prompts.EG_3,
            example_4=conversion_prompts.EG_4,
            code_to_convert=code_to_convert,
        )
        prompt = [{"role": "user", "content": base_prompt}]

        if error_msg is not None:
            content_new = conversion_prompts.conversion_error_template.render(
                error_msg=error_msg
            )
            prompt.append(
                {"role": "assistant", "content": f"```python\n{prev_response}\n```"}
            )
            prompt.append({"role": "user", "content": content_new})

        return prompt

    @classmethod
    def make_translate_prompt(
        cls, code_to_translate, prev_response: str, error_msg: str, error_summary: str
    ):
        base_prompt = translation_prompts.translation_template.render(
            code_to_translate=code_to_translate
        )
        prompt = [{"role": "user", "content": base_prompt}]

        if error_msg is not None:
            error_prompt = translation_prompts.translation_error_template.render(
                error_summary=error_summary
            )
            prompt.append(
                {
                    "role": "assistant",
                    "content": f"<cuda>\n\n```c++\n\n{prev_response}\n\n```\n\n</cuda>",
                }
            )
            prompt.append({"role": "user", "content": error_prompt})

        return prompt

    @classmethod
    def make_translate_error_summary_prompt(
        cls, code_to_translate, prev_response: str = None, error_msg: str = None
    ):
        base_prompt = translation_prompts.translation_template.render(
            code_to_translate=code_to_translate
        )
        error_summary_prompt = (
            translation_prompts.translation_error_summary_template.render(
                error_msg=error_msg
            )
        )

        prompt = [
            {"role": "user", "content": base_prompt},
            {
                "role": "assistant",
                "content": f"<cuda>\n\n```c++\n\n{prev_response}\n\n```\n\n</cuda>",
            },
            {"role": "user", "content": error_summary_prompt},
        ]
        return prompt

    @classmethod
    def make_evo_prompt(
        cls,
        gpu_type: str,
        cuda_version: str,
        top_5_kernel: list,
        func_runtime: float,
        cuda_indiv: dict,
    ):
        base_prompt = evo_prompts.evo_sys_prompt_template.render(
            gpu_type=gpu_type,
            cuda_version=cuda_version,
            optimization_history=top_5_kernel,
            func_runtime=func_runtime,
        )
        propose_prompt = evo_prompts.evo_propose_template.render(
            code=cuda_indiv["code"],
            runtime=cuda_indiv["runtime"],
            func_runtime=func_runtime,
            profile_string=cuda_indiv["prof_string"],
        )
        prompt = [
            {"role": "user", "content": base_prompt},
            {"role": "user", "content": propose_prompt},
        ]
        return prompt

    @classmethod
    def make_rag_prompt(
        cls,
        gpu_type: str,
        cuda_version: str,
        optimization_history: list,
        func_runtime: float,
        cuda_indiv: dict,
    ):
        base_prompt = rag_prompts.rag_sys_prompt_template.render(
            gpu_type=gpu_type,
            cuda_version=cuda_version,
            optimization_history=optimization_history,
            func_runtime=func_runtime,
        )
        propose_prompt = evo_prompts.evo_propose_template.render(
            code=cuda_indiv["code"],
            runtime=cuda_indiv["runtime"],
            func_runtime=func_runtime,
            profile_string=cuda_indiv["prof_string"],
        )
        prompt = [
            {"role": "user", "content": base_prompt},
            {"role": "user", "content": propose_prompt},
        ]
        return prompt
