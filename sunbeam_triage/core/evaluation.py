from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    uuid: str
    phase: str
    manifest_sha256: str
    accepted_root_causes: tuple[str, ...]
    required_evidence: tuple[str, ...] = ()
    counterevidence: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    exact_root_cause_known: bool = True
    evidence_sufficient: bool = True
    human_steering_required: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationCase:
        required = {"uuid", "phase", "manifest_sha256", "accepted_root_causes"}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"Missing evaluation case fields: {', '.join(missing)}")
        return cls(
            uuid=str(data["uuid"]),
            phase=str(data["phase"]),
            manifest_sha256=str(data["manifest_sha256"]),
            accepted_root_causes=tuple(map(str, data["accepted_root_causes"])),
            required_evidence=tuple(map(str, data.get("required_evidence", ()))),
            counterevidence=tuple(map(str, data.get("counterevidence", ()))),
            forbidden_claims=tuple(map(str, data.get("forbidden_claims", ()))),
            exact_root_cause_known=bool(data.get("exact_root_cause_known", True)),
            evidence_sufficient=bool(data.get("evidence_sufficient", True)),
            human_steering_required=bool(data.get("human_steering_required")),
        )


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation corpus must be a JSON array")
    cases = [EvaluationCase.from_dict(item) for item in data]
    duplicates = [
        case.uuid for case in cases if sum(c.uuid == case.uuid for c in cases) > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate evaluation UUID: {min(duplicates)}")
    return cases


def score_session(case: EvaluationCase, session: dict[str, Any]) -> dict[str, Any]:
    root_cause = str(session.get("root_cause", ""))
    evidence = "\n".join(
        f"{item.get('path', '')}:{item.get('line', '')} {item.get('excerpt', '')}"
        for item in session.get("evidence", [])
        if isinstance(item, dict)
    )
    report_text = "\n".join(
        str(session.get(field, ""))
        for field in ("summary", "failure_surface", "root_cause")
    )
    accepted = _matches_any(root_cause, case.accepted_root_causes)
    required_hits = [
        pattern for pattern in case.required_evidence if _matches(evidence, pattern)
    ]
    counter_hits = [
        pattern for pattern in case.counterevidence if _matches(evidence, pattern)
    ]
    forbidden_hits = [
        pattern for pattern in case.forbidden_claims if _matches(report_text, pattern)
    ]
    acknowledges_insufficiency = bool(session.get("needs_more_evidence")) or str(
        session.get("confidence", "")
    ) in {"unknown", "speculative"}
    accuracy = (
        accepted
        if case.exact_root_cause_known
        else (acknowledges_insufficiency if not case.evidence_sufficient else accepted)
    )
    evidence_coverage = (
        len(required_hits) / len(case.required_evidence)
        if case.required_evidence
        else 1.0
    )
    supported = bool(session.get("evidence")) and not forbidden_hits
    return {
        "uuid": case.uuid,
        "root_cause_accurate": accuracy,
        "directly_supported": supported,
        "required_evidence_coverage": evidence_coverage,
        "counterevidence_coverage": (
            len(counter_hits) / len(case.counterevidence)
            if case.counterevidence
            else 1.0
        ),
        "forbidden_claims": forbidden_hits,
        "acknowledges_insufficient_evidence": acknowledges_insufficiency,
        "passed": (
            accuracy
            and supported
            and len(required_hits) == len(case.required_evidence)
            and not forbidden_hits
        ),
    }


def manifest_sha256(artifact_root: Path) -> str:
    manifest = Path(artifact_root) / ".sunbeam-triage-manifest.json"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches(text, pattern) for pattern in patterns)


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
