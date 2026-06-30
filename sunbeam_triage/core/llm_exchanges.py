from __future__ import annotations

import json
from typing import Any

from .redaction import redact_data, redact_text


class ExchangeRecorder:
    def __init__(self) -> None:
        self.exchanges: list[dict[str, Any]] = []

    def record(self, request: dict[str, Any], response: Any) -> None:
        visible_request = {
            key: value
            for key, value in request.items()
            if key
            in {
                "model",
                "messages",
                "session_id",
                "cache_control",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
            }
        }
        visible_response = {
            "content": response_content(response),
            "usage": usage_dict(getattr(response, "usage", None)),
        }
        calls = tool_calls(response)
        if calls:
            visible_response["tool_calls"] = [tool_call_dict(call) for call in calls]
        self.exchanges.append({
            "request": redact_data(json_safe(visible_request)),
            "response": redact_data(json_safe(visible_response)),
        })


def response_content(response: Any) -> str:
    choice = response.choices[0]
    message = choice.message
    if message.content is None:
        return ""
    return redact_text(str(message.content))


def response_json(response: Any) -> dict[str, Any]:
    content = response_content(response)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "LLM response was not valid JSON. "
            f"Response preview: {response_preview(content)}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            "LLM response JSON was not an object. "
            f"Response preview: {response_preview(content)}"
        )
    return data


def response_preview(content: str, max_chars: int = 4000) -> str:
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 16] + "\n[truncated]\n"


def tool_calls(response: Any) -> list[Any]:
    choice = response.choices[0]
    message = choice.message
    return list(getattr(message, "tool_calls", None) or [])


def assistant_tool_message(response: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response_content(response),
        "tool_calls": [tool_call_dict(call) for call in tool_calls(response)],
    }


def tool_call_dict(tool_call: Any) -> dict[str, Any]:
    name, arguments = tool_call_name_and_arguments(tool_call)
    return {
        "id": tool_call_id(tool_call),
        "type": tool_call_type(tool_call),
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True),
        },
    }


def tool_call_name_and_arguments(tool_call: Any) -> tuple[str, dict[str, Any]]:
    function = get_value(tool_call, "function", {})
    name = str(get_value(function, "name", ""))
    raw_arguments = get_value(function, "arguments", "{}")
    if isinstance(raw_arguments, dict):
        return name, raw_arguments
    try:
        parsed = json.loads(str(raw_arguments or "{}"))
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return name, parsed


def tool_call_id(tool_call: Any) -> str:
    return str(get_value(tool_call, "id", ""))


def tool_call_type(tool_call: Any) -> str:
    return str(get_value(tool_call, "type", "function"))


def get_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    data: dict[str, Any] = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_details",
        "prompt_tokens_details",
        "cache_write_tokens",
        "cost",
        "cost_details",
        "is_byok",
    ):
        if hasattr(usage, name):
            data[name] = json_safe(getattr(usage, name))
    return data


def coerce_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {
            key: json_safe(item)
            for key, item in attrs.items()
            if not key.startswith("_")
        }
    return str(value)
