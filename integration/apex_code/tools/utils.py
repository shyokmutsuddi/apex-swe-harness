"""Tool utilities and helpers."""

import json
import re
from typing import Any


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    """
    Parse tool calls from agent response content.

    Supports xml tool use format:
    - <tool_use>{"tool": "name", ...}</tool_use>

    Args:
        content: Agent response content to parse

    Returns:
        List of parsed tool calls
    """
    tool_calls = []

    # XML tool use format: <tool_use>{...}</tool_use>
    xml_patterns = [
        r"<tool_use>\s*(\{.*?\})\s*</tool_use>",  # Direct JSON
        r"<tool_use>\s*```json\s*(\{.*?\})\s*```\s*</tool_use>",  # Nested markdown
    ]

    for pattern in xml_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                tool_call = json.loads(match)
                if isinstance(tool_call, dict) and "tool" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

    return tool_calls


def format_tool_call(tool_name: str, **kwargs) -> dict[str, Any]:
    """
    Format a tool call dictionary.

    Args:
        tool_name: Name of the tool
        **kwargs: Tool parameters

    Returns:
        Formatted tool call dictionary
    """
    return {"tool": tool_name, **kwargs}


def validate_tool_call(tool_call: dict[str, Any]) -> bool:
    """
    Validate a tool call dictionary.

    Args:
        tool_call: Tool call to validate

    Returns:
        True if valid, False otherwise
    """
    return (
        isinstance(tool_call, dict)
        and "tool" in tool_call
        and isinstance(tool_call["tool"], str)
        and tool_call["tool"].strip() != ""
    )
