from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openrouter import OpenRouter

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
    ) -> DiagnosisReport:
        request = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a CI failure diagnostician. Use only the provided "
                        "evidence. Do not invent files, timestamps, or causes."
                    ),
                },
                {"role": "user", "content": evidence_text},
            ],
            "response_format": {"type": "json_schema", "json_schema": REPORT_SCHEMA},
        }
        if session_id:
            request["session_id"] = session_id
        request.update(_cache_kwargs(self.config.model))
        response = self._send(request)
        self._record_exchange(request, response)
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
    ) -> str:
        request = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are continuing a Sunbeam CI failure diagnosis. "
                        "Use the provided diagnosis context and evidence. "
                        "Separate evidence from inference."
                    ),
                },
                {"role": "user", "content": context_text},
                *messages,
            ],
        }
        if session_id:
            request["session_id"] = session_id
        request.update(_cache_kwargs(self.config.model))
        response = self._send(request)
        self._record_exchange(request, response)
        return _response_content(response)

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
            if key in {"model", "messages", "session_id", "cache_control"}
        }
        self.exchanges.append(
            {
                "request": visible_request,
                "response": {
                    "content": _response_content(response),
                    "usage": _usage_dict(getattr(response, "usage", None)),
                },
            }
        )


def _cache_kwargs(model: str) -> dict[str, Any]:
    if model.startswith("anthropic/"):
        return {"cache_control": {"type": "ephemeral"}}
    return {}


def _response_content(response: Any) -> str:
    choice = response.choices[0]
    message = choice.message
    return str(message.content)


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
            data[name] = getattr(usage, name)
    return data
