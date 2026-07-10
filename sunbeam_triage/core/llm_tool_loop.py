from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .artifact_tools import artifact_tool_definitions, execute_artifact_tool
from .llm_exchanges import (
    assistant_tool_message,
    coerce_float,
    tool_call_dict,
    tool_call_id,
    tool_call_name_and_arguments,
    tool_calls,
    usage_dict,
)
from .llm_policy import tool_calls_need_targeted_read_nudge
from .llm_prompts import (
    finalization_message,
    investigation_state_message,
    targeted_read_nudge_message,
)
from .progress import (
    ProgressEvent,
    ProgressSink,
    emit_progress,
    event_from_tool_call,
)
from .triage_state import (
    BudgetProfile,
    InvestigationState,
    TriageLoopOptions,
    observe_tool_result,
    resolve_triage_budget,
)

TOOL_RESULT_TRUNCATED_MARKER = json.dumps(
    {"tool_result_truncated_by_budget": True},
    sort_keys=True,
)
MAX_TOOL_CALLS_PER_ROUND = 4
MAX_ACCUMULATED_MESSAGE_CHARS = 120_000


class ModelToolProtocolError(RuntimeError):
    code = "model_tool_protocol"

    def __init__(self, model: str, tool_choice: str, exchange_count: int):
        self.model = model
        self.tool_choice = tool_choice
        self.exchange_count = exchange_count
        super().__init__(
            f"Model {model} returned no tool call while tool_choice={tool_choice}"
        )


class ArtifactToolLoop:
    def __init__(
        self,
        *,
        send: Callable[[dict[str, Any]], Any],
        record_exchange: Callable[[dict[str, Any], Any], None],
    ):
        self.send = send
        self.record_exchange = record_exchange

    def run(
        self,
        request: dict[str, Any],
        *,
        artifact_root: Path | None,
        max_tool_rounds: int | None,
        max_tool_result_chars: int,
        triage_options: TriageLoopOptions | None = None,
        tool_choice: str = "auto",
        progress: ProgressSink | None = None,
        run_id: str = "",
        run_type: str = "diagnosis",
        contender_id: str | None = None,
    ) -> Any:
        if artifact_root is None:
            emit_model_request(
                progress,
                run_id=run_id,
                run_type=run_type,
                contender_id=contender_id,
                round_number=None,
            )
            response = self.send(request)
            self.record_exchange(request, response)
            emit_completed(
                progress,
                response,
                run_id=run_id,
                run_type=run_type,
                contender_id=contender_id,
            )
            return response

        options = triage_options or resolve_triage_budget(
            BudgetProfile(max_tool_result_chars=max_tool_result_chars),
            max_tool_rounds=max_tool_rounds,
        )
        state = InvestigationState(options=options)
        seen_tool_call_keys: set[str] = set()
        required_tool_observed = False
        request = {
            **request,
            "tools": artifact_tool_definitions(),
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
        }
        for round_number in range(1, options.max_rounds + 1):
            emit_model_request(
                progress,
                run_id=run_id,
                run_type=run_type,
                contender_id=contender_id,
                round_number=round_number,
            )
            response = self.send(request)
            self.record_exchange(request, response)
            calls = tool_calls(response)
            if not calls:
                if tool_choice == "required" and not required_tool_observed:
                    _raise_required_tool_error(
                        request=request,
                        progress=progress,
                        run_id=run_id,
                        run_type=run_type,
                        contender_id=contender_id,
                        round_number=round_number,
                    )
                emit_completed(
                    progress,
                    response,
                    run_id=run_id,
                    run_type=run_type,
                    contender_id=contender_id,
                )
                return response
            required_tool_observed = True
            for tool_call in calls:
                emit_progress(
                    progress,
                    event_from_tool_call(
                        tool_call_dict(tool_call),
                        run_id=run_id,
                        run_type=run_type,
                        contender_id=contender_id,
                        round_number=round_number,
                    ),
                )
            tool_result_messages = tool_result_messages_for_calls(
                artifact_root,
                calls,
                max_chars=options.max_tool_result_chars,
                seen_tool_call_keys=seen_tool_call_keys,
            )
            for tool_call, message in zip(calls, tool_result_messages, strict=False):
                name, arguments = tool_call_name_and_arguments(tool_call)
                emit_progress(
                    progress,
                    ProgressEvent(
                        run_id=run_id,
                        run_type=run_type,
                        phase="tool_result",
                        status="running",
                        message=f"Tool returned {name}",
                        contender_id=contender_id,
                        round_number=round_number,
                        tool_name=name,
                        target=progress_tool_target(arguments),
                        result_chars=len(str(message.get("content", ""))),
                    ),
                )
            state.record_round()
            for tool_call, message in zip(calls, tool_result_messages, strict=False):
                name, arguments = tool_call_name_and_arguments(tool_call)
                state.apply_observation(
                    observe_tool_result(
                        name, arguments, str(message.get("content", ""))
                    )
                )
            previous_messages = request["messages"]
            current_round_messages = [
                assistant_tool_message(response),
                *tool_result_messages,
            ]
            request["messages"] = [*previous_messages, *current_round_messages]
            if tool_calls_need_targeted_read_nudge(calls):
                request["messages"].append(targeted_read_nudge_message())
            request["messages"].append(investigation_state_message(state))
            request["messages"] = compact_investigation_messages(
                request["messages"],
                state,
                current_round_size=len(current_round_messages) + 2,
            )
            if state.should_finalize():
                final_request = final_request_with_state(request, state)
                emit_progress(
                    progress,
                    ProgressEvent(
                        run_id=run_id,
                        run_type=run_type,
                        phase="finalizing",
                        status="running",
                        message=f"Finalizing after {state.stop_reason}",
                        contender_id=contender_id,
                        round_number=round_number,
                    ),
                )
                emit_model_request(
                    progress,
                    run_id=run_id,
                    run_type=run_type,
                    contender_id=contender_id,
                    round_number=round_number + 1,
                )
                response = self.send(final_request)
                self.record_exchange(final_request, response)
                emit_completed(
                    progress,
                    response,
                    run_id=run_id,
                    run_type=run_type,
                    contender_id=contender_id,
                )
                return response

        if not state.stop_reason:
            state.stop_reason = "budget_exhausted"
            state.phase = "finalisation"
        final_request = final_request_with_state(request, state)
        emit_progress(
            progress,
            ProgressEvent(
                run_id=run_id,
                run_type=run_type,
                phase="finalizing",
                status="running",
                message=f"Finalizing after {state.stop_reason}",
                contender_id=contender_id,
                round_number=options.max_rounds,
                warning=state.stop_reason,
            ),
        )
        emit_model_request(
            progress,
            run_id=run_id,
            run_type=run_type,
            contender_id=contender_id,
            round_number=options.max_rounds + 1,
        )
        response = self.send(final_request)
        self.record_exchange(final_request, response)
        emit_completed(
            progress,
            response,
            run_id=run_id,
            run_type=run_type,
            contender_id=contender_id,
        )
        return response


def _raise_required_tool_error(
    *,
    request: dict[str, Any],
    progress: ProgressSink | None,
    run_id: str,
    run_type: str,
    contender_id: str | None,
    round_number: int,
) -> None:
    error = ModelToolProtocolError(
        str(request.get("model", "unknown")),
        "required",
        round_number,
    )
    emit_progress(
        progress,
        ProgressEvent(
            run_id=run_id,
            run_type=run_type,
            phase="model_protocol_error",
            status="failed",
            message=str(error),
            contender_id=contender_id,
            round_number=round_number,
            warning=error.code,
        ),
    )
    raise error


def emit_model_request(
    progress: ProgressSink | None,
    *,
    run_id: str,
    run_type: str,
    contender_id: str | None,
    round_number: int | None,
) -> None:
    subject = f"Contender {contender_id}" if contender_id else "Model"
    emit_progress(
        progress,
        ProgressEvent(
            run_id=run_id,
            run_type=run_type,
            phase="model_request",
            status="running",
            message=f"{subject} request sent",
            contender_id=contender_id,
            round_number=round_number,
        ),
    )


def emit_completed(
    progress: ProgressSink | None,
    response: Any,
    *,
    run_id: str,
    run_type: str,
    contender_id: str | None,
) -> None:
    usage = usage_dict(getattr(response, "usage", None))
    subject = f"Contender {contender_id}" if contender_id else "Model"
    emit_progress(
        progress,
        ProgressEvent(
            run_id=run_id,
            run_type=run_type,
            phase="completed",
            status="completed",
            message=f"{subject} response complete",
            contender_id=contender_id,
            total_tokens=int(usage.get("total_tokens") or 0),
            total_cost=coerce_float(usage.get("cost")),
        ),
    )


def progress_tool_target(arguments: dict[str, Any]) -> str:
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


def tool_result_message(
    root: Path,
    tool_call: Any,
    *,
    max_chars: int,
) -> dict[str, Any]:
    name, arguments = tool_call_name_and_arguments(tool_call)
    result = execute_artifact_tool(root, name, arguments)
    content = tool_result_content(result, max_chars=max_chars)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id(tool_call),
        "content": content,
    }


def tool_result_messages_for_calls(
    root: Path,
    calls: list[Any],
    *,
    max_chars: int,
    seen_tool_call_keys: set[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    remaining_budget = max(max_chars, 0)
    for index, tool_call in enumerate(calls):
        remaining_calls = len(calls) - index
        call_budget = remaining_budget // remaining_calls if remaining_calls else 0
        message = bounded_tool_result_message(
            root,
            tool_call,
            index=index,
            max_chars=call_budget,
            seen_tool_call_keys=seen_tool_call_keys,
        )
        remaining_budget = max(remaining_budget - len(message["content"]), 0)
        messages.append(message)
    return messages


def bounded_tool_result_message(
    root: Path,
    tool_call: Any,
    *,
    index: int,
    max_chars: int,
    seen_tool_call_keys: set[str],
) -> dict[str, Any]:
    if index >= MAX_TOOL_CALLS_PER_ROUND:
        return synthetic_tool_result_message(
            tool_call,
            {
                "ok": False,
                "round_tool_limit_reached": True,
                "reason": (
                    f"Tool call skipped because the per-round limit is "
                    f"{MAX_TOOL_CALLS_PER_ROUND}."
                ),
            },
            max_chars=max_chars,
        )
    key = tool_call_cache_key(tool_call)
    if key in seen_tool_call_keys:
        return synthetic_tool_result_message(
            tool_call,
            {
                "ok": False,
                "duplicate_tool_call": True,
                "reason": (
                    "Duplicate tool call skipped; use earlier result for same "
                    "arguments."
                ),
            },
            max_chars=max_chars,
        )
    seen_tool_call_keys.add(key)
    return tool_result_message(root, tool_call, max_chars=max_chars)


def synthetic_tool_result_message(
    tool_call: Any,
    result: dict[str, Any],
    *,
    max_chars: int,
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id(tool_call),
        "content": tool_result_content(result, max_chars=max_chars),
    }


def tool_result_content(result: dict[str, Any], *, max_chars: int) -> str:
    content = json.dumps(result, sort_keys=True)
    if len(content) <= max_chars:
        return content

    trimmed = dict(result)
    if isinstance(trimmed.get("content"), str):
        trimmed["tool_result_truncated_by_budget"] = True
        overhead = len(json.dumps({**trimmed, "content": ""}, sort_keys=True))
        budget = max(max_chars - overhead - 16, 0)
        trimmed["content"] = trimmed["content"][:budget]
    elif isinstance(trimmed.get("files"), list):
        trimmed["files"] = []
        trimmed["tool_result_truncated_by_budget"] = True
        trimmed["error"] = "Tool result exceeded budget; use a narrower prefix."
    elif isinstance(trimmed.get("matches"), list):
        trimmed["matches"] = []
        trimmed["tool_result_truncated_by_budget"] = True
        trimmed["error"] = "Tool result exceeded budget; use a narrower search."
    else:
        trimmed = {
            "ok": False,
            "tool_result_truncated_by_budget": True,
            "error": "Tool result exceeded budget; narrow the request.",
        }

    content = json.dumps(trimmed, sort_keys=True)
    if len(content) <= max_chars:
        return content
    compact = json.dumps(
        {
            "ok": False,
            "tool_result_truncated_by_budget": True,
            "error": "Tool result exceeded budget; narrow the request.",
        },
        sort_keys=True,
    )
    if len(compact) <= max_chars:
        return compact
    return TOOL_RESULT_TRUNCATED_MARKER


def final_request_with_state(
    request: dict[str, Any],
    state: InvestigationState,
) -> dict[str, Any]:
    final_request = dict(request)
    final_request["tool_choice"] = "none"
    final_request["messages"] = [
        *final_request["messages"],
        finalization_message(state),
    ]
    return final_request


def tool_call_cache_key(tool_call: Any) -> str:
    name, arguments = tool_call_name_and_arguments(tool_call)
    return json.dumps(
        {"name": name, "arguments": arguments},
        sort_keys=True,
        default=str,
    )


def compact_investigation_messages(
    messages: list[dict[str, Any]],
    state: InvestigationState,
    *,
    current_round_size: int,
) -> list[dict[str, Any]]:
    total_chars = sum(len(str(message.get("content", ""))) for message in messages)
    if total_chars <= MAX_ACCUMULATED_MESSAGE_CHARS:
        return messages
    prefix = messages[:2]
    suffix = messages[-current_round_size:]
    return [
        *prefix,
        {
            "role": "user",
            "content": (
                "The harness compacted earlier raw tool results to control context "
                "growth. The following evidence ledger preserves their stable "
                f"findings and unresolved checks:\n{state.to_prompt_summary()}"
            ),
        },
        *suffix,
    ]
