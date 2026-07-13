from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_policy import TARGETED_READ_TOOLS
from .redaction import redact_data
from .tool_activity import analyze_tool_activity


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
    expected_failure_triggers: tuple[str, ...] = ()
    expected_contributing_factors: tuple[str, ...] = ()
    expected_post_failure_outcomes: tuple[str, ...] = ()

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
            expected_failure_triggers=tuple(
                map(str, data.get("expected_failure_triggers", ()))
            ),
            expected_contributing_factors=tuple(
                map(str, data.get("expected_contributing_factors", ()))
            ),
            expected_post_failure_outcomes=tuple(
                map(str, data.get("expected_post_failure_outcomes", ()))
            ),
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
    assessment = session.get("causal_assessment")
    root_claim = (
        assessment.get("root_cause", {}) if isinstance(assessment, dict) else {}
    )
    root_cause = str(root_claim.get("claim") or session.get("root_cause", ""))
    root_confidence = str(root_claim.get("confidence") or session.get("confidence", ""))
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
    acknowledges_insufficiency = bool(session.get("needs_more_evidence")) or (
        root_confidence in {"unknown", "speculative"}
    )
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
    causal_loss, unsupported_claims = _causal_loss(
        assessment,
        session.get("evidence", []),
        exact_root_cause_known=case.exact_root_cause_known,
    )
    supported = (
        bool(session.get("evidence")) and not forbidden_hits and not unsupported_claims
    )
    role_coverage = _causal_role_coverage(case, assessment)
    activity = session.get("tool_activity")
    if not isinstance(activity, dict):
        activity = analyze_tool_activity(session)
    targeted_read_performed = any(
        row.get("tool_name") in TARGETED_READ_TOOLS
        for row in activity.get("rows", [])
        if isinstance(row, dict)
    )
    tool_protocol_compliant = session.get("error_type") != "model_tool_protocol"
    secret_free = redact_data(session) == session
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
        "causal_loss": causal_loss,
        "unsupported_causal_claims": unsupported_claims,
        "causal_role_coverage": role_coverage,
        "acknowledges_insufficient_evidence": acknowledges_insufficiency,
        "tool_protocol_compliant": tool_protocol_compliant,
        "targeted_read_performed": targeted_read_performed,
        "verdict_source": str(session.get("verdict_source", "unknown")),
        "secret_free": secret_free,
        "passed": (
            accuracy
            and supported
            and len(required_hits) == len(case.required_evidence)
            and not forbidden_hits
            and causal_loss == 0
            and all(role_coverage.values())
            and tool_protocol_compliant
            and secret_free
        ),
    }


def _causal_loss(
    assessment: Any,
    evidence: Any,
    *,
    exact_root_cause_known: bool,
) -> tuple[int, list[str]]:
    if not isinstance(assessment, dict):
        return 0, []
    evidence_ids = {
        str(item.get("id", ""))
        for item in evidence
        if isinstance(item, dict) and item.get("id")
    }
    claims: list[tuple[str, dict[str, Any]]] = []
    for role in ("failure_trigger", "root_cause", "post_failure_outcome"):
        value = assessment.get(role)
        if isinstance(value, dict):
            claims.append((role, value))
    for role in ("symptoms", "contributing_factors"):
        claims.extend(
            (role, value)
            for value in assessment.get(role, []) or []
            if isinstance(value, dict)
        )

    weights = {"confirmed": 4, "supported": 2, "speculative": 1}
    loss = 0
    unsupported: list[str] = []
    for role, claim in claims:
        confidence = str(claim.get("confidence", "unknown"))
        if confidence == "unknown":
            continue
        refs = [str(item) for item in claim.get("evidence_ids", [])]
        unresolved = [item for item in refs if item not in evidence_ids]
        if refs and not unresolved:
            continue
        label = f"{role}: {claim.get('claim', '')}".strip()
        unsupported.append(label)
        loss += weights.get(confidence, 1)
    root = assessment.get("root_cause")
    if (
        exact_root_cause_known
        and isinstance(root, dict)
        and root.get("confidence") == "unknown"
    ):
        loss += 1
    return loss, unsupported


def _causal_role_coverage(
    case: EvaluationCase,
    assessment: Any,
) -> dict[str, bool]:
    if not isinstance(assessment, dict):
        return {
            "failure_trigger": not case.expected_failure_triggers,
            "contributing_factors": not case.expected_contributing_factors,
            "post_failure_outcome": not case.expected_post_failure_outcomes,
        }
    trigger = str((assessment.get("failure_trigger") or {}).get("claim", ""))
    outcome = str((assessment.get("post_failure_outcome") or {}).get("claim", ""))
    factors = "\n".join(
        str(item.get("claim", ""))
        for item in assessment.get("contributing_factors", [])
        if isinstance(item, dict)
    )
    return {
        "failure_trigger": all(
            _matches(trigger, pattern) for pattern in case.expected_failure_triggers
        ),
        "contributing_factors": all(
            _matches(factors, pattern) for pattern in case.expected_contributing_factors
        ),
        "post_failure_outcome": all(
            _matches(outcome, pattern)
            for pattern in case.expected_post_failure_outcomes
        ),
    }


def manifest_sha256(artifact_root: Path) -> str:
    manifest = Path(artifact_root) / ".sunbeam-triage-manifest.json"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(_matches(text, pattern) for pattern in patterns)


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
