from __future__ import annotations

import json
from collections import Counter
from typing import Any


def analyze_tool_activity(session: dict[str, Any]) -> dict[str, Any]:
    exchanges = [
        exchange
        for exchange in session.get("exchanges", [])
        if isinstance(exchange, dict)
    ]
    rows = []
    tool_result_chars = 0
    tool_result_count = 0
    total_tokens = 0
    total_cost = 0.0
    budget_truncated = False
    fallback = False
    targets: Counter[tuple[str, str]] = Counter()
    tool_result_chars_by_id: dict[str, int] = {}
    tool_result_count_by_id: Counter[str] = Counter()

    for exchange in exchanges:
        request = exchange.get("request", {})
        messages = request.get("messages", []) if isinstance(request, dict) else []
        tool_results = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        result_chars = sum(
            len(str(message.get("content", ""))) for message in tool_results
        )
        tool_result_chars += result_chars
        tool_result_count += len(tool_results)
        for message in tool_results:
            tool_call_id = str(message.get("tool_call_id", ""))
            content_len = len(str(message.get("content", "")))
            tool_result_chars_by_id[tool_call_id] = content_len
            tool_result_count_by_id[tool_call_id] += 1
            parsed = _parse_json_object(message.get("content"))
            if parsed.get("tool_result_truncated_by_budget"):
                budget_truncated = True

        if _is_fallback_request(messages):
            fallback = True

    for index, exchange in enumerate(exchanges, start=1):
        response = exchange.get("response", {})

        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if isinstance(usage, dict):
            total_tokens += int(usage.get("total_tokens") or 0)
            total_cost += _coerce_float(usage.get("cost"))

        tool_calls = (
            response.get("tool_calls", []) if isinstance(response, dict) else []
        )
        for call in tool_calls or []:
            tool_name, target = _tool_call_name_and_target(call)
            call_id = str(call.get("id", "")) if isinstance(call, dict) else ""
            targets[tool_name, target] += 1
            rows.append({
                "exchange": index,
                "tool_name": tool_name,
                "target": target,
                "result_count": tool_result_count_by_id[call_id],
                "result_chars": tool_result_chars_by_id.get(call_id, 0),
                "total_tokens": (
                    usage.get("total_tokens") if isinstance(usage, dict) else None
                ),
                "cost": _coerce_float(usage.get("cost")),
            })

    repeated_reads = [
        {"tool": tool, "target": target, "count": count}
        for (tool, target), count in sorted(targets.items())
        if tool in {"get_artifact_file", "get_sosreport_file"} and target and count > 1
    ]
    warnings = []
    if budget_truncated:
        warnings.append("budget_truncated_tool_result")
    if repeated_reads:
        warnings.append("repeated_read")
    if fallback:
        warnings.append("tool_budget_fallback")
    return {
        "uuid": str(session.get("uuid", "")),
        "model": str(session.get("model", "")),
        "exchange_count": len(exchanges),
        "tool_call_count": len(rows),
        "tool_result_count": tool_result_count,
        "tool_result_chars": tool_result_chars,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "repeated_reads": repeated_reads,
        "warnings": warnings,
        "rows": rows,
    }


def _tool_call_name_and_target(call: dict[str, Any]) -> tuple[str, str]:
    function = call.get("function", {}) if isinstance(call, dict) else {}
    if not isinstance(function, dict):
        return "", ""
    name = str(function.get("name", ""))
    arguments = _parse_json_object(function.get("arguments"))
    target = str(
        arguments.get("path")
        or arguments.get("member_path")
        or arguments.get("archive_path")
        or arguments.get("path_prefix")
        or ""
    )
    if arguments.get("archive_path") and arguments.get("member_path"):
        target = f"{arguments['archive_path']}::{arguments['member_path']}"
    return name, target


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _is_fallback_request(messages: list[Any]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "artifact tool budget is exhausted" in str(message.get("content", "")):
            return True
    return False
