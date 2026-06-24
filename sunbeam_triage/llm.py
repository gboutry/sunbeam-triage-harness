from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openrouter import OpenRouter

from .artifact_tools import artifact_tool_definitions, execute_artifact_tool
from .config import LlmConfig


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
                        "If more log context is needed, list artifact files first "
                        "and search artifacts before reading whole files. For "
                        "sosreport archives, search inside the archive before "
                        "reading a specific member. Prefer line windows over full "
                        "file reads. "
                        "Read a specific file only when it is likely to materially "
                        "improve the diagnosis, because file reads can be costly."
                    ),
                },
                {"role": "user", "content": evidence_text},
            ],
            "response_format": {"type": "json_schema", "json_schema": REPORT_SCHEMA},
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
        content = _response_content(response)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM response was not valid JSON") from exc
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
                        "Separate evidence from inference. If more log context "
                        "is needed, list artifact files first and search artifacts "
                        "before reading whole files. For sosreport archives, search "
                        "inside the archive before reading a specific member. "
                        "Prefer line windows over full file reads. Read a specific "
                        "file only when it is likely to materially improve the "
                        "answer, because file reads can be costly."
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
    ):
        if artifact_root is None:
            response = self._send(request)
            self._record_exchange(request, response)
            return response

        request = {
            **request,
            "tools": artifact_tool_definitions(),
            "tool_choice": "auto",
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
    minimal = json.dumps({"tool_result_truncated_by_budget": True}, sort_keys=True)
    if len(minimal) <= max_chars:
        return minimal
    if max_chars >= 2:
        return "{}"
    return ""


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
                "provided. Do not request more files."
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
        "prompt_tokens_details",
        "cache_write_tokens",
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
