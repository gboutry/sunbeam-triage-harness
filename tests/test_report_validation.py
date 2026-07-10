import pytest

from sunbeam_triage.core.report_validation import validate_diagnosis_report
from sunbeam_triage.core.triage_state import observe_tool_result


def _base_report(**overrides):
    report = {
        "summary": "The join step failed while snorlax was not ready.",
        "failure_surface": "sunbeam cluster join failed waiting for k8s readiness.",
        "confidence": "confirmed",
        "root_cause": "snorlax never registered as a Kubernetes node.",
        "needs_more_evidence": False,
        "evidence": [
            {
                "path": "generated/sunbeam/output.log",
                "line": 895,
                "excerpt": "Node snorlax is not ready",
            }
        ],
        "candidate_mechanisms": [
            {
                "name": "snorlax node registration failure",
                "status": "confirmed",
                "rationale": "snorlax never appeared in node readiness output.",
            }
        ],
        "recommendations": [],
        "unknowns": [],
        "triage_confidence": "high",
        "failure_timeline": [],
        "cascading_errors": [],
        "alternatives_considered": [],
        "missing_evidence": [],
        "stop_reason": "",
    }
    report.update(overrides)
    return report


def test_report_validation_rejects_placeholder_diagnosis_fields():
    with pytest.raises(ValueError, match="placeholder text for root_cause"):
        validate_diagnosis_report(_base_report(root_cause="..."), [])


def test_confirmed_report_without_targeted_read_is_downgraded():
    observations = [
        observe_tool_result(
            "search_artifacts",
            {"pattern": "snorlax"},
            '{"ok": true, "matches": ['
            '{"path": "generated/sunbeam/output.log", "line": 895, '
            '"excerpt": "Node snorlax is not ready"}]}',
        )
    ]

    validated = validate_diagnosis_report(_base_report(), observations)

    assert validated["confidence"] == "supported"
    assert validated["triage_confidence"] == "medium"
    assert validated["needs_more_evidence"] is True
    assert validated["candidate_mechanisms"][0]["status"] == "supported"
    assert any("targeted read" in item for item in validated["missing_evidence"])


def test_failed_targeted_read_blocks_entity_claim():
    observations = [
        observe_tool_result(
            "get_artifact_file",
            {"path": "generated/sunbeam/output.log", "line_start": 880},
            '{"ok": true, "path": "generated/sunbeam/output.log", '
            '"line_start": 880, "content": "Node sarabhai is ready\\n"}',
        ),
        observe_tool_result(
            "get_sosreport_file",
            {
                "archive_path": "generated/sunbeam/sosreport-snaroli.tar.xz",
                "member_path": "var/log/syslog",
            },
            '{"ok": false, "error": "Archive does not exist: '
            'generated/sunbeam/sosreport-snaroli.tar.xz"}',
        ),
    ]

    validated = validate_diagnosis_report(_base_report(), observations)

    assert validated["confidence"] == "supported"
    assert validated["triage_confidence"] == "medium"
    assert validated["needs_more_evidence"] is True
    assert any("snorlax" in item for item in validated["missing_evidence"])
    assert any("failed targeted read" in item for item in validated["unknowns"])


def test_needs_more_evidence_prevents_confirmed_confidence():
    validated = validate_diagnosis_report(
        _base_report(needs_more_evidence=True),
        [
            observe_tool_result(
                "get_artifact_file",
                {
                    "path": "generated/sunbeam/output.log",
                    "line_start": 890,
                },
                '{"ok": true, "path": "generated/sunbeam/output.log", '
                '"line_start": 890, "content": "Node snorlax is not ready\\n"}',
            )
        ],
    )

    assert validated["confidence"] == "supported"
    assert validated["candidate_mechanisms"][0]["status"] == "supported"


def test_supported_report_with_targeted_failure_surface_evidence_survives():
    report = _base_report(
        confidence="supported",
        triage_confidence="medium",
        root_cause="k8s readiness did not converge before the join timeout.",
        candidate_mechanisms=[
            {
                "name": "k8s readiness timeout",
                "status": "supported",
                "rationale": "The join output shows the readiness gate timing out.",
            }
        ],
    )
    observations = [
        observe_tool_result(
            "get_artifact_file",
            {"path": "generated/sunbeam/output.log", "line_start": 890},
            '{"ok": true, "path": "generated/sunbeam/output.log", '
            '"line_start": 890, "content": "sunbeam cluster join failed\\n"}',
        )
    ]

    validated = validate_diagnosis_report(report, observations)

    assert validated["confidence"] == "supported"
    assert validated["triage_confidence"] == "medium"
    assert validated["needs_more_evidence"] is False


def test_supported_report_without_targeted_read_becomes_speculative():
    report = _base_report(confidence="supported", triage_confidence="medium")
    observations = [
        observe_tool_result(
            "search_artifacts",
            {"pattern": "snorlax"},
            '{"ok": true, "matches": [{'
            '"path": "generated/sunbeam/output.log", "line": 895, '
            '"excerpt": "Node snorlax is not ready"}]}',
        )
    ]

    validated = validate_diagnosis_report(report, observations)

    assert validated["confidence"] == "speculative"
    assert validated["needs_more_evidence"] is True


def test_supported_report_with_unresolved_citation_becomes_speculative():
    report = _base_report(
        confidence="supported",
        triage_confidence="medium",
        evidence=[{"path": "invented.log", "line": 99, "excerpt": "invented"}],
    )
    observations = [
        observe_tool_result(
            "get_artifact_file",
            {"path": "generated/sunbeam/output.log", "line_start": 890},
            '{"ok": true, "path": "generated/sunbeam/output.log", '
            '"line_start": 890, "content": "sunbeam cluster join failed\\n"}',
        )
    ]

    validated = validate_diagnosis_report(report, observations)

    assert validated["confidence"] == "speculative"
    assert any("did not resolve" in item for item in validated["missing_evidence"])
