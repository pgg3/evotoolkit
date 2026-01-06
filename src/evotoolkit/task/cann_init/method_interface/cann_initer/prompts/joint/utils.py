# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Utility functions for Joint Branch conversation extraction"""

from typing import List, Optional


def extract_current_plan(conversation: List[dict]) -> Optional[str]:
    """Extract current best plan from conversation (avoid passing full history).

    Finds the last tiling proposal with a <response> block.
    """
    if not conversation:
        return None
    # Find the last tiling proposal
    for msg in reversed(conversation):
        if msg.get('role') == 'tiling' and '<response>' in msg.get('content', ''):
            content = msg['content']
            start = content.find('<response>')
            end = content.find('</response>')
            if start != -1 and end != -1:
                return content[start:end + len('</response>')]
    return None


def extract_kernel_feedback(conversation: List[dict]) -> Optional[str]:
    """Extract latest kernel feedback from conversation."""
    if not conversation:
        return None
    for msg in reversed(conversation):
        if msg.get('role') == 'kernel':
            return msg.get('content', '')
    return None


def extract_kernel_design(conversation: List[dict]) -> Optional[str]:
    """Extract kernel design (including pseudocode) from conversation.

    Looks for accepted kernel responses containing '## Kernel Design'.
    """
    for msg in reversed(conversation):
        if msg.get('role') == 'kernel' and 'accepted: true' in msg.get('content', '').lower():
            content = msg['content']
            # Extract from ## Kernel Design to end of response
            start = content.find('## Kernel Design')
            if start != -1:
                end = content.find('</response>')
                if end != -1:
                    return content[start:end].strip()
    return None


def extract_tiling_strategy(conversation: List[dict]) -> dict:
    """Extract tiling strategy info from the current plan.

    Returns:
        dict with 'strategy' ('default' or 'custom') and 'paradigm' ('vector' or 'cube')
    """
    current_plan = extract_current_plan(conversation)
    if not current_plan:
        return {"strategy": "unknown", "paradigm": "vector"}

    plan_lower = current_plan.lower()
    strategy = "default" if "strategy: default" in plan_lower else "custom"
    paradigm = "cube" if "paradigm: cube" in plan_lower else "vector"
    return {"strategy": strategy, "paradigm": paradigm}
