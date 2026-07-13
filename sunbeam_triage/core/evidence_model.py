from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .redaction import redact_text

Confidence = str
ReadClass = str


def evidence_id(
    path: str,
    line: int | None,
    excerpt: str,
    *,
    member_path: str = "",
) -> str:
    """
    Build an identifier for one observation.

    Returns:
        A stable, content-addressed evidence identifier.

    """
    material = "\x1f".join((path, member_path, str(line or ""), excerpt.strip()))
    return f"ev-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


def confidence_for_read(read_class: ReadClass) -> Confidence:
    return {
        "discovery": "low",
        "broad_search": "low",
        "targeted_search": "medium",
        "targeted_read": "high",
    }.get(read_class, "low")


@dataclass(frozen=True)
class SourceRef:
    path: str
    line_start: int | None = None
    line_end: int | None = None
    member_path: str = ""
    role: str = ""

    def location(self) -> str:
        path = self.path
        if self.member_path:
            path = f"{path}:{self.member_path}"
        if self.line_start is None:
            return path
        if self.line_end is None or self.line_end == self.line_start:
            return f"{path}:{self.line_start}"
        return f"{path}:{self.line_start}-{self.line_end}"


@dataclass(frozen=True)
class EvidenceProvenance:
    origin: str
    selector: str = ""
    read_class: ReadClass = "broad_search"
    probe_id: str = ""
    tool_name: str = ""


@dataclass(frozen=True)
class EvidenceObservation:
    source: SourceRef
    category: str
    excerpt: str
    provenance: EvidenceProvenance
    timestamp: str = ""
    is_counterevidence: bool = False
    is_missing_evidence: bool = False
    confidence: Confidence | None = None

    def __post_init__(self) -> None:
        if self.confidence is None:
            object.__setattr__(
                self,
                "confidence",
                confidence_for_read(self.provenance.read_class),
            )

    @property
    def is_targeted(self) -> bool:
        return self.provenance.read_class in {"targeted_search", "targeted_read"}

    @property
    def resolved_confidence(self) -> Confidence:
        return self.confidence or confidence_for_read(self.provenance.read_class)

    def to_prompt_line(self) -> str:
        return (
            f"- [{self.category}/{self.resolved_confidence}/"
            f"{self.provenance.read_class}] {self.source.location()}: "
            f"{redact_text(self.excerpt)}"
        )
