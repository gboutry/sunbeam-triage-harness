from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import LlmConfig
from .http import UrlLibHttp


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
    def __init__(self, config: LlmConfig, http=None):
        self.config = config
        self.http = http or UrlLibHttp(timeout=config.timeout_seconds)

    def diagnose(self, evidence_text: str) -> DiagnosisReport:
        if not self.config.api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key. Set {self.config.api_key_env}."
            )
        payload = {
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
        response = self.http.post_json(
            f"{self.config.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        content = response["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM response was not valid JSON") from exc
        return DiagnosisReport.from_dict(data)

    def chat(self, context_text: str, messages: list[dict[str, str]]) -> str:
        if not self.config.api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key. Set {self.config.api_key_env}."
            )
        payload = {
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
        response = self.http.post_json(
            f"{self.config.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        return str(response["choices"][0]["message"]["content"])
