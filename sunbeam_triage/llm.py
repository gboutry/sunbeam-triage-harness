from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openrouter import OpenRouter

from .artifact_tools import artifact_tool_definitions, execute_artifact_tool
from .config import LlmConfig


EVIDENCE_PRODUCING_TOOLS = {
    "search_artifacts",
    "get_artifact_file",
    "search_sosreport",
    "get_sosreport_file",
}
DISCOVERY_TOOLS = {
    "list_artifact_files",
    "list_sosreports",
    "list_sosreport_files",
}
TOOL_RESULT_TRUNCATED_MARKER = json.dumps(
    {"tool_result_truncated_by_budget": True},
    sort_keys=True,
)


@dataclass(frozen=True)
class ReportEvidence:
    path: str
    line: int | None
    excerpt: str


@dataclass(frozen=True)
class CandidateMechanism:
    name: str
    status: str
    rationale: str


@dataclass(frozen=True)
class DiagnosisReport:
    summary: str
    failure_surface: str
    confidence: str
    root_cause: str
    needs_more_evidence: bool = False
    evidence: list[ReportEvidence] = field(default_factory=list)
    candidate_mechanisms: list[CandidateMechanism] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiagnosisReport":
        return cls(
            summary=str(data.get("summary", "")),
            failure_surface=str(data.get("failure_surface", "")),
            confidence=str(data.get("confidence", "speculative")),
            root_cause=str(data.get("root_cause", "")),
            needs_more_evidence=bool(data.get("needs_more_evidence", False)),
            evidence=[
                ReportEvidence(
                    path=str(item.get("path", "")),
                    line=item.get("line"),
                    excerpt=str(item.get("excerpt", "")),
                )
                for item in data.get("evidence", [])
                if isinstance(item, dict)
            ],
            candidate_mechanisms=[
                CandidateMechanism(
                    name=str(item.get("name", "")),
                    status=str(item.get("status", "")),
                    rationale=str(item.get("rationale", "")),
                )
                for item in data.get("candidate_mechanisms", [])
                if isinstance(item, dict)
            ],
            recommendations=[
                str(item) for item in data.get("recommendations", []) if item is not None
            ],
            unknowns=[str(item) for item in data.get("unknowns", []) if item is not None],
        )


REPORT_SCHEMA = {
    "name": "sunbeam_ci_diagnosis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "failure_surface",
            "confidence",
            "root_cause",
            "needs_more_evidence",
            "evidence",
            "candidate_mechanisms",
            "recommendations",
            "unknowns",
        ],
        "properties": {
            "summary": {"type": "string"},
            "failure_surface": {"type": "string"},
            "confidence": {
                "type": "string",
                "enum": ["confirmed", "supported", "speculative", "unknown"],
            },
            "root_cause": {"type": "string"},
            "needs_more_evidence": {"type": "boolean"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "line", "excerpt"],
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": ["integer", "null"]},
                        "excerpt": {"type": "string"},
                    },
                },
            },
            "candidate_mechanisms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "status", "rationale"],
                    "properties": {
                        "name": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["confirmed", "supported", "speculative", "rejected"],
                        },
                        "rationale": {"type": "string"},
                    },
                },
            },
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "unknowns": {"type": "array", "items": {"type": "string"}},
        },
    },
}


class OpenRouterClient:
    def __init__(self, config: LlmConfig, sdk_client=None):
        self.config = config
        self.sdk_client = sdk_client
        self.exchanges: list[dict[str, Any]] = []

    def diagnose(
        self,
        evidence_text: str,
        *,
        session_id: str | None = None,
        artifact_root: Path | None = None,
        max_tool_rounds: int = 4,
        max_tool_result_chars: int = 60_000,
    ) -> DiagnosisReport:
        request = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a CI failure diagnostician. Use only the provided "
                        "evidence. Do not invent files, timestamps, or causes. "
                        "Actively inspect the next likely decisive artifact when "
                        "the current evidence only identifies a wrapper failure. "
                        "Use search_artifacts before reading whole files. For "
                        "sosreport archives, use search_sosreport before reading "
                        "a specific member. Prefer line windows over full file "
                        "reads. Read a specific file only when it is likely to "
                        "materially improve the diagnosis, because file reads can "
                        "be costly. Set needs_more_evidence to true only when a "
                        "likely decisive artifact remains uninspected; if tools "
                        "are available, use them before finalizing that response."
                    ),
                },
                {"role": "user", "content": evidence_text},
            ],
            "response_format": {"type": "json_schema", "json_schema": REPORT_SCHEMA},
        }
        if session_id:
            request["session_id"] = session_id
        request.update(_cache_kwargs(self.config.model))
        exchange_start = len(self.exchanges)
        response = self._send_with_artifact_tools(
            request,
            artifact_root=artifact_root,
            max_tool_rounds=max_tool_rounds,
            max_tool_result_chars=max_tool_result_chars,
        )
        data = _response_json(response)
        has_tool_budget_fallback = _exchange_range_has_tool_budget_fallback(
            self.exchanges,
            exchange_start,
        )
        if has_tool_budget_fallback:
            data = _downgrade_tool_budget_diagnosis(data)
        if (
            not has_tool_budget_fallback
            and artifact_root is not None
            and _diagnosis_needs_required_tool_retry(
                data,
                self.exchanges,
                exchange_start,
            )
        ):
            retry_request = _request_with_evidence_retry(
                request,
                response,
                data=data,
            )
            response = self._send_with_artifact_tools(
                retry_request,
                artifact_root=artifact_root,
                max_tool_rounds=max_tool_rounds,
                max_tool_result_chars=max_tool_result_chars,
                tool_choice="required",
            )
            data = _response_json(response)
            if _exchange_range_has_tool_budget_fallback(
                self.exchanges,
                exchange_start,
            ):
                has_tool_budget_fallback = True
                data = _downgrade_tool_budget_diagnosis(data)
        if (
            not has_tool_budget_fallback
            and artifact_root is not None
            and _diagnosis_confidence_requires_artifact_evidence(data)
            and not _exchange_range_has_evidence_tool_calls(
                self.exchanges,
                exchange_start,
            )
        ):
            data = _downgrade_missing_tool_evidence_diagnosis(data)
        return DiagnosisReport.from_dict(data)

    def chat(
        self,
        context_text: str,
        messages: list[dict[str, str]],
        *,
        session_id: str | None = None,
        artifact_root: Path | None = None,
        max_tool_rounds: int = 4,
        max_tool_result_chars: int = 60_000,
    ) -> str:
        request = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are continuing a Sunbeam CI failure diagnosis. "
                        "Use the provided diagnosis context and evidence. "
                        "Separate evidence from inference. Actively inspect the "
                        "next likely decisive artifact when the current context "
                        "only identifies a wrapper failure. Use search_artifacts "
                        "before reading whole files. For sosreport archives, use "
                        "search_sosreport before reading a specific member. "
                        "Prefer line windows over full file reads. Read a "
                        "specific file only when it is likely to materially "
                        "improve the answer, because file reads can be costly."
                    ),
                },
                {"role": "user", "content": context_text},
                *messages,
            ],
        }
        if session_id:
            request["session_id"] = session_id
        request.update(_cache_kwargs(self.config.model))
        response = self._send_with_artifact_tools(
            request,
            artifact_root=artifact_root,
            max_tool_rounds=max_tool_rounds,
            max_tool_result_chars=max_tool_result_chars,
        )
        return _response_content(response)

    def _send_with_artifact_tools(
        self,
        request: dict[str, Any],
        *,
        artifact_root: Path | None,
        max_tool_rounds: int,
        max_tool_result_chars: int,
        tool_choice: str = "auto",
    ):
        if artifact_root is None:
            response = self._send(request)
            self._record_exchange(request, response)
            return response

        request = {
            **request,
            "tools": artifact_tool_definitions(),
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
        }
        for _ in range(max_tool_rounds):
            response = self._send(request)
            self._record_exchange(request, response)
            tool_calls = _tool_calls(response)
            if not tool_calls:
                return response
            request["messages"] = [
                *request["messages"],
                _assistant_tool_message(response),
                *_tool_result_messages(
                    artifact_root,
                    tool_calls,
                    max_chars=max_tool_result_chars,
                ),
            ]
            if _tool_calls_need_targeted_read_nudge(tool_calls):
                request["messages"].append(_targeted_read_nudge_message())

        final_request = _final_request_without_artifact_tools(request)
        response = self._send(final_request)
        self._record_exchange(final_request, response)
        return response

    def _send(self, request: dict[str, Any]):
        return self._sdk().chat.send(**request)

    def _sdk(self):
        if self.sdk_client is not None:
            return self.sdk_client
        if not self.config.api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key. Set {self.config.api_key_env}."
            )
        self.sdk_client = OpenRouter(
            api_key=self.config.api_key,
            server_url=self.config.base_url,
            timeout_ms=self.config.timeout_seconds * 1000,
        )
        return self.sdk_client

    def _record_exchange(self, request: dict[str, Any], response: Any) -> None:
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
            "content": _response_content(response),
            "usage": _usage_dict(getattr(response, "usage", None)),
        }
        tool_calls = _tool_calls(response)
        if tool_calls:
            visible_response["tool_calls"] = [
                _tool_call_dict(call) for call in tool_calls
            ]
        self.exchanges.append(
            {
                "request": _json_safe(visible_request),
                "response": _json_safe(visible_response),
            }
        )


def _cache_kwargs(model: str) -> dict[str, Any]:
    if model.startswith("anthropic/"):
        return {"cache_control": {"type": "ephemeral"}}
    return {}


def _response_content(response: Any) -> str:
    choice = response.choices[0]
    message = choice.message
    if message.content is None:
        return ""
    return str(message.content)


def _response_json(response: Any) -> dict[str, Any]:
    content = _response_content(response)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("LLM response JSON was not an object")
    return data


def _exchange_range_has_tool_calls(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    for exchange in exchanges[start_index:]:
        response = exchange.get("response", {})
        if isinstance(response, dict) and response.get("tool_calls"):
            return True
    return False


def _exchange_range_has_evidence_tool_calls(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    return bool(
        _exchange_range_tool_names(exchanges, start_index) & EVIDENCE_PRODUCING_TOOLS
    )


def _exchange_range_tool_names(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> set[str]:
    names = set()
    for exchange in exchanges[start_index:]:
        response = exchange.get("response", {})
        if not isinstance(response, dict):
            continue
        for tool_call in response.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if isinstance(function, dict):
                names.add(str(function.get("name", "")))
    return names


def _exchange_range_has_tool_budget_fallback(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    for exchange in exchanges[start_index:]:
        request = exchange.get("request", {})
        if not isinstance(request, dict):
            continue
        if "tools" in request:
            continue
        for message in request.get("messages", []):
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", ""))
            if "artifact tool budget is exhausted" in content:
                return True
    return False


def _diagnosis_needs_required_tool_retry(
    data: dict[str, Any],
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    if data.get("needs_more_evidence") is True and not _exchange_range_has_tool_calls(
        exchanges,
        start_index,
    ):
        return True
    return (
        _diagnosis_confidence_requires_artifact_evidence(data)
        and not _exchange_range_has_evidence_tool_calls(exchanges, start_index)
    )


def _diagnosis_confidence_requires_artifact_evidence(data: dict[str, Any]) -> bool:
    return data.get("confidence") in {"supported", "confirmed"}


def _downgrade_tool_budget_diagnosis(data: dict[str, Any]) -> dict[str, Any]:
    downgraded = dict(data)
    if downgraded.get("confidence") == "confirmed":
        downgraded["confidence"] = "supported"
    downgraded["needs_more_evidence"] = True
    unknowns = [str(item) for item in downgraded.get("unknowns", []) if item is not None]
    budget_unknown = (
        "The artifact tool budget was exhausted before targeted artifact reads "
        "could fully validate the mechanism."
    )
    if budget_unknown not in unknowns:
        unknowns.append(budget_unknown)
    downgraded["unknowns"] = unknowns
    mechanisms = []
    for item in downgraded.get("candidate_mechanisms", []):
        if not isinstance(item, dict):
            continue
        mechanism = dict(item)
        if mechanism.get("status") == "confirmed":
            mechanism["status"] = "supported"
        mechanisms.append(mechanism)
    downgraded["candidate_mechanisms"] = mechanisms
    return downgraded


def _downgrade_missing_tool_evidence_diagnosis(data: dict[str, Any]) -> dict[str, Any]:
    downgraded = dict(data)
    if downgraded.get("confidence") in {"supported", "confirmed"}:
        downgraded["confidence"] = "speculative"
    downgraded["needs_more_evidence"] = True
    unknowns = [str(item) for item in downgraded.get("unknowns", []) if item is not None]
    missing_unknown = (
        "No evidence-producing artifact tools were used to validate the "
        "diagnosis against the downloaded artifacts."
    )
    if missing_unknown not in unknowns:
        unknowns.append(missing_unknown)
    downgraded["unknowns"] = unknowns
    mechanisms = []
    for item in downgraded.get("candidate_mechanisms", []):
        if not isinstance(item, dict):
            continue
        mechanism = dict(item)
        if mechanism.get("status") in {"supported", "confirmed"}:
            mechanism["status"] = "speculative"
        mechanisms.append(mechanism)
    downgraded["candidate_mechanisms"] = mechanisms
    return downgraded


def _request_with_evidence_retry(
    request: dict[str, Any],
    response: Any,
    *,
    data: dict[str, Any],
) -> dict[str, Any]:
    if data.get("needs_more_evidence") is True:
        retry_content = (
            "You marked needs_more_evidence=true without using artifact "
            "tools. Do not answer yet. First use the artifact tools to "
            "search for the most likely decisive log or read a narrow "
            "line window, then finalize the diagnosis. Keep tool usage "
            "targeted."
        )
    else:
        retry_content = (
            "You returned a supported or confirmed diagnosis without using "
            "an evidence-producing artifact tool. Do not answer yet. First "
            "use an evidence-producing artifact tool such as search_artifacts, "
            "get_artifact_file, search_sosreport, or get_sosreport_file to "
            "validate the most likely decisive evidence, then finalize the "
            "diagnosis. Keep tool usage targeted."
        )
    return {
        **request,
        "messages": [
            *request["messages"],
            {"role": "assistant", "content": _response_content(response)},
            {"role": "user", "content": retry_content},
        ],
    }


def _tool_calls_need_targeted_read_nudge(tool_calls: list[Any]) -> bool:
    names = {_tool_call_name_and_arguments(call)[0] for call in tool_calls}
    broad_tools = DISCOVERY_TOOLS | {"search_artifacts"}
    targeted_tools = EVIDENCE_PRODUCING_TOOLS - {"search_artifacts"}
    return bool(names & broad_tools) and not bool(names & targeted_tools)


def _targeted_read_nudge_message() -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "The previous tool round only discovered candidate artifacts or "
            "broad matches. If the evidence is still insufficient, make the "
            "next tool call a targeted read or targeted archive search, such "
            "as get_artifact_file, search_sosreport, or get_sosreport_file. "
            "If no targeted read is justified, keep the final diagnosis "
            "conservative and leave needs_more_evidence=true."
        ),
    }


def _tool_calls(response: Any) -> list[Any]:
    choice = response.choices[0]
    message = choice.message
    return list(getattr(message, "tool_calls", None) or [])


def _assistant_tool_message(response: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": _response_content(response),
        "tool_calls": [_tool_call_dict(call) for call in _tool_calls(response)],
    }


def _tool_result_message(
    root: Path,
    tool_call: Any,
    *,
    max_chars: int,
) -> dict[str, Any]:
    name, arguments = _tool_call_name_and_arguments(tool_call)
    result = execute_artifact_tool(root, name, arguments)
    content = _tool_result_content(result, max_chars=max_chars)
    return {
        "role": "tool",
        "tool_call_id": _tool_call_id(tool_call),
        "content": content,
    }


def _tool_result_messages(
    root: Path,
    tool_calls: list[Any],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    remaining_budget = max(max_chars, 0)
    for index, tool_call in enumerate(tool_calls):
        remaining_calls = len(tool_calls) - index
        call_budget = remaining_budget // remaining_calls if remaining_calls else 0
        message = _tool_result_message(root, tool_call, max_chars=call_budget)
        remaining_budget = max(remaining_budget - len(message["content"]), 0)
        messages.append(message)
    return messages


def _tool_result_content(result: dict[str, Any], *, max_chars: int) -> str:
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


def _final_request_without_artifact_tools(request: dict[str, Any]) -> dict[str, Any]:
    final_request = {
        key: value
        for key, value in request.items()
        if key not in {"tools", "tool_choice", "parallel_tool_calls"}
    }
    final_request["messages"] = [
        *final_request["messages"],
        {
            "role": "user",
            "content": (
                "The artifact tool budget is exhausted. Answer now using the "
                "initial evidence and any artifact tool results already "
                "provided. Do not request more files. Do not mark a diagnosis "
                "confirmed solely from discovery-only tool results; preserve "
                "uncertainty when targeted files or sosreport members were not "
                "read."
            ),
        },
    ]
    return final_request


def _tool_call_dict(tool_call: Any) -> dict[str, Any]:
    name, arguments = _tool_call_name_and_arguments(tool_call)
    return {
        "id": _tool_call_id(tool_call),
        "type": _tool_call_type(tool_call),
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True),
        },
    }


def _tool_call_name_and_arguments(tool_call: Any) -> tuple[str, dict[str, Any]]:
    function = _get_value(tool_call, "function", {})
    name = str(_get_value(function, "name", ""))
    raw_arguments = _get_value(function, "arguments", "{}")
    if isinstance(raw_arguments, dict):
        return name, raw_arguments
    try:
        parsed = json.loads(str(raw_arguments or "{}"))
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return name, parsed


def _tool_call_id(tool_call: Any) -> str:
    return str(_get_value(tool_call, "id", ""))


def _tool_call_type(tool_call: Any) -> str:
    return str(_get_value(tool_call, "type", "function"))


def _get_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _usage_dict(usage: Any) -> dict[str, Any]:
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
            data[name] = _json_safe(getattr(usage, name))
    return data


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {
            key: _json_safe(item)
            for key, item in attrs.items()
            if not key.startswith("_")
        }
    return str(value)
