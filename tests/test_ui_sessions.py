from pathlib import Path

from sunbeam_triage.evidence import EvidenceCollector
from sunbeam_triage.llm import DiagnosisReport, ReportEvidence
from sunbeam_triage.ui_helpers import (
    build_followup_context,
    list_saved_sessions,
    load_ui_session,
    save_ui_session,
    session_store_root,
)


def test_session_store_round_trips_and_lists_recent_first(tmp_path):
    artifact_root = tmp_path / "artifacts"
    older = {
        "uuid": "older-uuid",
        "model": "model/a",
        "summary": "Older summary",
        "confidence": "supported",
        "updated_at": "2026-06-24T10:00:00Z",
    }
    newer = {
        "uuid": "newer-uuid",
        "model": "model/b",
        "summary": "Newer summary",
        "confidence": "confirmed",
        "updated_at": "2026-06-24T11:00:00Z",
        "chat": [{"role": "user", "content": "What next?"}],
    }

    save_ui_session(artifact_root, older)
    save_ui_session(artifact_root, newer)

    assert load_ui_session(artifact_root, "newer-uuid") == newer
    assert list_saved_sessions(artifact_root) == [
        {
            "uuid": "newer-uuid",
            "model": "model/b",
            "summary": "Newer summary",
            "confidence": "confirmed",
            "updated_at": "2026-06-24T11:00:00Z",
            "chat_count": 1,
        },
        {
            "uuid": "older-uuid",
            "model": "model/a",
            "summary": "Older summary",
            "confidence": "supported",
            "updated_at": "2026-06-24T10:00:00Z",
            "chat_count": 0,
        },
    ]
    assert session_store_root(artifact_root) == artifact_root / ".sunbeam-triage-ui"


def test_load_ui_session_returns_none_for_missing_uuid(tmp_path):
    assert load_ui_session(tmp_path / "artifacts", "missing") is None


def test_build_followup_context_includes_diagnosis_evidence_and_attachments():
    pack = EvidenceCollector(Path("tests/fixtures/sample_uuid"), "sample-uuid").collect()
    report = DiagnosisReport(
        summary="Timed out",
        failure_surface="Deploy timeout",
        confidence="supported",
        root_cause="Readiness did not converge",
        evidence=[
            ReportEvidence(
                path="generated/sunbeam/output.log",
                line=2,
                excerpt="wait timed out",
            )
        ],
        recommendations=["Inspect readiness"],
        unknowns=["No sosreport"],
    )

    context = build_followup_context(
        pack,
        report,
        attachments=[
            {
                "path": "generated/sunbeam/output.log",
                "line": 2,
                "text": "wait timed out",
            }
        ],
    )

    assert "Solutions Run UUID: sample-uuid" in context
    assert "Diagnosis Summary: Timed out" in context
    assert "Root Cause: Readiness did not converge" in context
    assert "Model Evidence:" in context
    assert "generated/sunbeam/output.log:2: wait timed out" in context
    assert "Attached Context:" in context
