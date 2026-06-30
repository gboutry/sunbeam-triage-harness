from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class TimelineEvent:
    timestamp: str
    source: str
    location: str
    event: str


@dataclass(frozen=True)
class AlternativeConsidered:
    hypothesis: str
    status: str
    reason: str


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
    triage_confidence: str = "unknown"
    failure_timeline: list[TimelineEvent] = field(default_factory=list)
    cascading_errors: list[ReportEvidence] = field(default_factory=list)
    alternatives_considered: list[AlternativeConsidered] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    stop_reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosisReport:
        return cls(
            summary=str(data.get("summary", "")),
            failure_surface=str(data.get("failure_surface", "")),
            confidence=str(data.get("confidence", "speculative")),
            root_cause=str(data.get("root_cause", "")),
            needs_more_evidence=bool(data.get("needs_more_evidence")),
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
                str(item)
                for item in data.get("recommendations", [])
                if item is not None
            ],
            unknowns=[
                str(item) for item in data.get("unknowns", []) if item is not None
            ],
            triage_confidence=str(data.get("triage_confidence", "unknown")),
            failure_timeline=[
                TimelineEvent(
                    timestamp=str(item.get("timestamp", "")),
                    source=str(item.get("source", "")),
                    location=str(item.get("location", "")),
                    event=str(item.get("event", "")),
                )
                for item in data.get("failure_timeline", [])
                if isinstance(item, dict)
            ],
            cascading_errors=[
                ReportEvidence(
                    path=str(item.get("path", "")),
                    line=item.get("line"),
                    excerpt=str(item.get("excerpt", "")),
                )
                for item in data.get("cascading_errors", [])
                if isinstance(item, dict)
            ],
            alternatives_considered=[
                AlternativeConsidered(
                    hypothesis=str(item.get("hypothesis", "")),
                    status=str(item.get("status", "")),
                    reason=str(item.get("reason", "")),
                )
                for item in data.get("alternatives_considered", [])
                if isinstance(item, dict)
            ],
            missing_evidence=[
                str(item)
                for item in data.get("missing_evidence", [])
                if item is not None
            ],
            stop_reason=str(data.get("stop_reason", "")),
        )


REPORT_SCHEMA: dict[str, Any] = {
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
            "triage_confidence",
            "failure_timeline",
            "cascading_errors",
            "alternatives_considered",
            "missing_evidence",
            "stop_reason",
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
                            "enum": [
                                "confirmed",
                                "supported",
                                "speculative",
                                "rejected",
                            ],
                        },
                        "rationale": {"type": "string"},
                    },
                },
            },
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "unknowns": {"type": "array", "items": {"type": "string"}},
            "triage_confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "unknown"],
            },
            "failure_timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["timestamp", "source", "location", "event"],
                    "properties": {
                        "timestamp": {"type": "string"},
                        "source": {"type": "string"},
                        "location": {"type": "string"},
                        "event": {"type": "string"},
                    },
                },
            },
            "cascading_errors": {
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
            "alternatives_considered": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["hypothesis", "status", "reason"],
                    "properties": {
                        "hypothesis": {"type": "string"},
                        "status": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
            "missing_evidence": {"type": "array", "items": {"type": "string"}},
            "stop_reason": {"type": "string"},
        },
    },
}
