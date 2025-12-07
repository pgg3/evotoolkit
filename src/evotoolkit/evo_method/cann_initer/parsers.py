# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANNIniter 解析工具"""

import json
import re


def parse_json(response: str) -> dict:
    """从 LLM 响应中解析 JSON"""
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


def parse_code(response: str) -> str:
    """从 LLM 响应中解析代码"""
    code_match = re.search(r"```(?:cpp|c\+\+|python)?\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return response.strip()
