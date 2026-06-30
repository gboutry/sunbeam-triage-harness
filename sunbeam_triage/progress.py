from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable


ProgressSink = Callable[["ProgressEvent"], None]


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    run_type: str
    phase: str
    status: str
    message: str
    contender_id: str | None = None
    round_number: int | None = None
    tool_name: str | None = None
    target: str | None = None
    result_chars: int | None = None
    total_tokens: int | None = None
    total_cost: float | None = None
    warning: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )

    def to_trace(self) -> dict[str, Any]:
        trace: dict[str, Any] = {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "phase": self.phase,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
        }
        for key in (
            "contender_id",
            "round_number",
            "tool_name",
            "target",
            "result_chars",
            "total_tokens",
            "total_cost",
            "warning",
        ):
            value = getattr(self, key)
            if value is not None:
                trace[key] = value
        return trace


def emit_progress(progress: ProgressSink | None, event: ProgressEvent) -> None:
    if progress is not None:
        progress(event)


def event_from_tool_call(
    tool_call: dict[str, Any],
    *,
    run_id: str,
    run_type: str,
    contender_id: str | None = None,
    round_number: int | None = None,
) -> ProgressEvent:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    if not isinstance(function, dict):
        function = {}
    name = str(function.get("name", ""))
    arguments = _parse_json_object(function.get("arguments"))
    target = _tool_target(arguments)
    subject = f"Contender {contender_id}" if contender_id else "Model"
    return ProgressEvent(
        run_id=run_id,
        run_type=run_type,
        phase="tool_call",
        status="running",
        message=f"{subject} requested {name}",
        contender_id=contender_id,
        round_number=round_number,
        tool_name=name,
        target=target,
        raw={"tool_call": tool_call},
    )


def summarize_progress_events(events: list[ProgressEvent | dict[str, Any]]) -> dict[str, Any]:
    traces = [event.to_trace() if isinstance(event, ProgressEvent) else event for event in events]
    warnings = sorted(
        {
            str(event.get("warning"))
            for event in traces
            if isinstance(event, dict) and event.get("warning")
        }
    )
    return {
        "event_count": len(traces),
        "tool_call_count": sum(1 for event in traces if event.get("phase") == "tool_call"),
        "tool_result_count": sum(
            1 for event in traces if event.get("phase") == "tool_result"
        ),
        "tool_result_chars": sum(
            int(event.get("result_chars") or 0) for event in traces
        ),
        "total_tokens": sum(int(event.get("total_tokens") or 0) for event in traces),
        "total_cost": sum(float(event.get("total_cost") or 0) for event in traces),
        "warnings": warnings,
    }


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


def _tool_target(arguments: dict[str, Any]) -> str:
    if arguments.get("archive_path") and arguments.get("member_path"):
        return f"{arguments['archive_path']}::{arguments['member_path']}"
    return str(
        arguments.get("path")
        or arguments.get("member_path")
        or arguments.get("archive_path")
        or arguments.get("path_prefix")
        or arguments.get("pattern")
        or ""
    )
