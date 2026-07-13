import json

import pytest

from sunbeam_triage.core.evaluation import (
    EvaluationCase,
    load_evaluation_cases,
    manifest_sha256,
    score_session,
)


def test_score_session_requires_accepted_cause_and_cited_evidence():
    case = EvaluationCase(
        uuid="uuid",
        phase="maas",
        manifest_sha256="hash",
        accepted_root_causes=("ifaddresses.*bad call flags",),
        required_evidence=("maas-region-api.*postinst",),
        forbidden_claims=("network outage",),
    )
    session = {
        "root_cause": "ifaddresses() method: bad call flags",
        "evidence": [
            {
                "path": "maas-region-api.log",
                "line": 42,
                "excerpt": "postinst failed",
            }
        ],
    }

    score = score_session(case, session)

    assert score["root_cause_accurate"] is True
    assert score["required_evidence_coverage"] == 1.0
    assert score["tool_protocol_compliant"] is True
    assert score["targeted_read_performed"] is False
    assert score["secret_free"] is True
    assert score["passed"] is True


def test_score_session_rejects_model_tool_protocol_errors():
    case = EvaluationCase(
        uuid="uuid",
        phase="deploy",
        manifest_sha256="hash",
        accepted_root_causes=("apt-get update",),
    )

    score = score_session(
        case,
        {
            "root_cause": "apt-get update failed",
            "evidence": [{"path": "unit.log", "line": 1, "excerpt": "failed"}],
            "error_type": "model_tool_protocol",
        },
    )

    assert score["tool_protocol_compliant"] is False
    assert score["passed"] is False


def test_unknown_case_rewards_explicit_insufficient_evidence():
    case = EvaluationCase(
        uuid="uuid",
        phase="deploy",
        manifest_sha256="hash",
        accepted_root_causes=(),
        exact_root_cause_known=False,
        evidence_sufficient=False,
    )

    score = score_session(
        case,
        {
            "root_cause": "",
            "confidence": "unknown",
            "needs_more_evidence": True,
            "evidence": [{"path": "output.log", "line": 2, "excerpt": "timeout"}],
        },
    )

    assert score["root_cause_accurate"] is True


def test_confident_unsupported_cause_costs_more_than_unknown():
    case = EvaluationCase(
        uuid="uuid",
        phase="deploy",
        manifest_sha256="hash",
        accepted_root_causes=("CNI configuration",),
        exact_root_cause_known=True,
    )
    unsupported = score_session(
        case,
        {
            "root_cause": "CNI configuration was missing",
            "confidence": "confirmed",
            "evidence": [],
            "causal_assessment": {
                "failure_trigger": {
                    "claim": "readiness timed out",
                    "confidence": "unknown",
                    "evidence_ids": [],
                },
                "symptoms": [],
                "contributing_factors": [],
                "root_cause": {
                    "claim": "CNI configuration was missing",
                    "confidence": "confirmed",
                    "evidence_ids": ["ev-missing"],
                },
                "post_failure_outcome": {
                    "claim": "",
                    "confidence": "unknown",
                    "evidence_ids": [],
                },
            },
        },
    )
    unresolved = score_session(
        case,
        {
            "root_cause": "The underlying cause is not established.",
            "confidence": "unknown",
            "needs_more_evidence": True,
            "evidence": [
                {"id": "ev-1", "path": "output.log", "line": 2, "excerpt": "timeout"}
            ],
            "causal_assessment": {
                "failure_trigger": {
                    "claim": "readiness timed out",
                    "confidence": "unknown",
                    "evidence_ids": [],
                },
                "symptoms": [],
                "contributing_factors": [],
                "root_cause": {
                    "claim": "The underlying cause is not established.",
                    "confidence": "unknown",
                    "evidence_ids": [],
                },
                "post_failure_outcome": {
                    "claim": "",
                    "confidence": "unknown",
                    "evidence_ids": [],
                },
            },
        },
    )

    assert unsupported["causal_loss"] == 4
    assert unresolved["causal_loss"] == 1


def test_load_evaluation_cases_rejects_duplicate_uuids(tmp_path):
    path = tmp_path / "cases.json"
    case = {
        "uuid": "same",
        "phase": "deploy",
        "manifest_sha256": "hash",
        "accepted_root_causes": [],
    }
    path.write_text(json.dumps([case, case]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        load_evaluation_cases(path)


def test_manifest_sha256_detects_artifact_corpus_drift(tmp_path):
    (tmp_path / ".sunbeam-triage-manifest.json").write_text("[]", encoding="utf-8")

    assert manifest_sha256(tmp_path) == (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )
