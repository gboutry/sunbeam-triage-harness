from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evidence_model import evidence_id


@dataclass(frozen=True)
class ReportEvidence:
    path: str
    line: int | None
    excerpt: str
    id: str = ""
    role: str = "observation"

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(
                self, "id", evidence_id(self.path, self.line, self.excerpt)
            )


@dataclass(frozen=True)
class CausalClaim:
    claim: str
    confidence: str = "unknown"
    evidence_ids: list[str] = field(default_factory=list)
    counterevidence_ids: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CausalAssessment:
    failure_trigger: CausalClaim
    symptoms: list[CausalClaim] = field(default_factory=list)
    contributing_factors: list[CausalClaim] = field(default_factory=list)
    root_cause: CausalClaim = field(
        default_factory=lambda: CausalClaim(
            claim="The underlying cause is not established.",
            confidence="unknown",
        )
    )
    post_failure_outcome: CausalClaim = field(
        default_factory=lambda: CausalClaim(claim="", confidence="unknown")
    )


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
    causal_assessment: CausalAssessment | None = None

    def __post_init__(self) -> None:
        if self.causal_assessment is None:
            object.__setattr__(
                self, "causal_assessment", _legacy_causal_assessment(self)
            )

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
                    id=str(item.get("id", "")),
                    role=str(item.get("role", "observation")),
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
                    id=str(item.get("id", "")),
                    role=str(item.get("role", "observation")),
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
            causal_assessment=_causal_assessment_from_dict(
                data.get("causal_assessment")
            ),
        )


def _claim_from_dict(data: Any, *, default: str = "") -> CausalClaim:
    if not isinstance(data, dict):
        return CausalClaim(claim=default, confidence="unknown")
    return CausalClaim(
        claim=str(data.get("claim", default)),
        confidence=str(data.get("confidence", "unknown")),
        evidence_ids=[str(item) for item in data.get("evidence_ids", [])],
        counterevidence_ids=[str(item) for item in data.get("counterevidence_ids", [])],
        missing_evidence=[str(item) for item in data.get("missing_evidence", [])],
    )


def _causal_assessment_from_dict(data: Any) -> CausalAssessment | None:
    if not isinstance(data, dict):
        return None
    return CausalAssessment(
        failure_trigger=_claim_from_dict(data.get("failure_trigger")),
        symptoms=[
            _claim_from_dict(item)
            for item in data.get("symptoms", [])
            if isinstance(item, dict)
        ],
        contributing_factors=[
            _claim_from_dict(item)
            for item in data.get("contributing_factors", [])
            if isinstance(item, dict)
        ],
        root_cause=_claim_from_dict(
            data.get("root_cause"),
            default="The underlying cause is not established.",
        ),
        post_failure_outcome=_claim_from_dict(data.get("post_failure_outcome")),
    )


def _legacy_causal_assessment(report: DiagnosisReport) -> CausalAssessment:
    root_text = report.root_cause.strip()
    lowered = root_text.lower()
    root_unknown = not root_text or any(
        marker in lowered
        for marker in (
            "does not establish",
            "not established",
            "cannot be confirmed",
            "not confirmed",
            "unknown",
        )
    )
    return CausalAssessment(
        failure_trigger=CausalClaim(
            claim=report.failure_surface,
            confidence=report.confidence if report.failure_surface else "unknown",
            evidence_ids=[item.id for item in report.evidence],
        ),
        contributing_factors=[
            CausalClaim(claim=item.name, confidence=item.status)
            for item in report.candidate_mechanisms
            if item.status != "rejected"
        ],
        root_cause=CausalClaim(
            claim=(root_text or "The underlying cause is not established."),
            confidence="unknown" if root_unknown else report.confidence,
            evidence_ids=[] if root_unknown else [item.id for item in report.evidence],
            missing_evidence=list(report.missing_evidence),
        ),
    )


_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim",
        "confidence",
        "evidence_ids",
        "counterevidence_ids",
        "missing_evidence",
    ],
    "properties": {
        "claim": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["confirmed", "supported", "speculative", "unknown"],
        },
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "counterevidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
    },
}


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
            "causal_assessment",
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
                    "required": ["path", "line", "excerpt", "id", "role"],
                    "properties": {
                        "path": {"type": "string"},
                        "line": {"type": ["integer", "null"]},
                        "excerpt": {"type": "string"},
                        "id": {"type": "string"},
                        "role": {
                            "type": "string",
                            "enum": [
                                "observation",
                                "failure_trigger",
                                "symptom",
                                "contributing_factor",
                                "root_cause",
                                "counterevidence",
                                "post_failure_outcome",
                            ],
                        },
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
            "causal_assessment": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "failure_trigger",
                    "symptoms",
                    "contributing_factors",
                    "root_cause",
                    "post_failure_outcome",
                ],
                "properties": {
                    "failure_trigger": _CLAIM_SCHEMA,
                    "symptoms": {"type": "array", "items": _CLAIM_SCHEMA},
                    "contributing_factors": {
                        "type": "array",
                        "items": _CLAIM_SCHEMA,
                    },
                    "root_cause": _CLAIM_SCHEMA,
                    "post_failure_outcome": _CLAIM_SCHEMA,
                },
            },
        },
    },
}
