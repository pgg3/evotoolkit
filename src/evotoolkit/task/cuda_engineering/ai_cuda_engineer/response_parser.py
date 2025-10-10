# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import re
import json
class ResponseParser:
    @classmethod
    def parse_convert_response(cls, response_to_parse=None) -> str:
        """
        parse the convert response
        """
        code_block_pattern = re.compile(
            r'\s*```(?:\w+)?\s*\n?(.*?)\n?\s*```\s*$',
            re.DOTALL
        )

        # find match
        match_group = code_block_pattern.findall(response_to_parse)
        if len(match_group) > 0:
            return match_group[0]
        else:
            return response_to_parse

    @classmethod
    def parse_translate_response(cls, response_to_parse=None) -> str:
        """
        parse the translate response
        """
        # match the content between <cuda> and </cuda> tags
        code_pattern = re.compile(r'<cuda>(.*?)</cuda>', re.DOTALL)

        # find the first match
        match = code_pattern.search(response_to_parse)
        if match is None:
            return response_to_parse
        else:
            code = match.group(1)
            code_block_pattern = re.compile(
                r'\s*```(?:[\w+]+)?\s*\n?(.*?)\n?\s*```\s*$',
                re.DOTALL
            )
            # find match
            match_group = code_block_pattern.findall(code)
            if match_group:
                return match_group[0]
            else:
                return code

    @classmethod
    def parse_evo_response(cls, response_to_parse=None) -> dict:
        if response_to_parse is None:
            return {"name": "", "thought": "", "code": ""}
        
        proposed_content = response_to_parse
        
        # Initialize variables to prevent undefined errors
        name = ""
        thought = ""
        cleaned_code = ""
        
        # if surrounded by ```json ```
        if proposed_content.strip().startswith("```json") and proposed_content.endswith("```"):
            content_inside = proposed_content.strip()[7:-3]
            try:
                json_dict = json.loads(content_inside)
                code_block_pattern = re.compile(
                    r'\s*(?:```[^\n]*)?\n?(.*?)(```|$)',
                    re.DOTALL
                )

                cleaned_code = code_block_pattern.search(json_dict["code"]).group(1)
                json_dict["code"] = cleaned_code
                del json_dict["code"]  # Fix: use del instead of remove

                return json_dict
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                # Fall through to regex parsing if JSON parsing fails
                pass

        name_heading = r"(?:name|Name|NAME)\s*:?"
        thought_heading = r"(?:thought|Thought|THOUGHT)\s*:?"
        code_heading = r"(?:code|Code|CODE)\s*:?"

        # Extract name
        name_pattern = re.compile(r"" + name_heading + r"\s*(.*?)" + code_heading, re.DOTALL)
        match = name_pattern.search(proposed_content)
        if match:
            name = match.group(1).strip()

        # Extract code
        code_pattern = re.compile(r"" + code_heading + r"\s*(.*)" + thought_heading, re.DOTALL)
        match = code_pattern.search(proposed_content)
        if match:
            code = match.group(1).strip()
            code_block_pattern = re.compile(
                r'\s*(?:```[^\n]*)?\n?(.*?)(```|$)',
                re.DOTALL
            )
            code_match = code_block_pattern.search(code)
            if code_match:
                cleaned_code = code_match.group(1)
            else:
                cleaned_code = code

        # Extract thought
        thought_pattern = re.compile(r"" + thought_heading + r"\s*(.*?)$", re.DOTALL)
        match = thought_pattern.search(proposed_content)
        if match:
            thought = match.group(1).strip()

        result = {
            "name": name,
            "thought": thought,
            "code": cleaned_code
        }
        return result
