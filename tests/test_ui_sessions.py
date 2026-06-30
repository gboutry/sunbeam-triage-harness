import importlib.util
from pathlib import Path

from sunbeam_triage.evidence import EvidenceCollector
from sunbeam_triage.llm import DiagnosisReport, ReportEvidence
from sunbeam_triage.progress import ProgressEvent
from sunbeam_triage.ui_helpers import (
    build_followup_context,
    list_saved_sessions,
    load_ui_session,
    save_ui_session,
    session_store_root,
)
from sunbeam_triage.sessions import load_session_record, save_session_snapshot


def _streamlit_app():
    spec = importlib.util.spec_from_file_location(
        "streamlit_app_for_tests",
        Path("streamlit_app.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_diagnosis_session_persists_needs_more_evidence(tmp_path):
    app = _streamlit_app()
    report = DiagnosisReport(
        summary="Incomplete",
        failure_surface="Wrapper failure",
        confidence="unknown",
        root_cause="",
        needs_more_evidence=True,
    )

    session = app._session_from_diagnosis(
        uuid="uuid",
        model="model/a",
        artifact_root=tmp_path / "uuid",
        output=tmp_path / "diagnostics.html",
        failed_step="sunbeam_test",
        report=report,
        exchanges=[],
        download_failures=[],
    )

    assert session["needs_more_evidence"] is True
    assert app._report_from_session(session).needs_more_evidence is True
    assert app._report_from_session({"summary": "old"}).needs_more_evidence is False


def test_diagnosis_session_persists_progress_events(tmp_path):
    app = _streamlit_app()
    report = DiagnosisReport(
        summary="Incomplete",
        failure_surface="Wrapper failure",
        confidence="unknown",
        root_cause="",
    )
    event = ProgressEvent(
        run_id="uuid",
        run_type="diagnosis",
        phase="model_request",
        status="running",
        message="Model request sent",
    )

    session = app._session_from_diagnosis(
        uuid="uuid",
        model="model/a",
        artifact_root=tmp_path / "uuid",
        output=tmp_path / "diagnostics.html",
        failed_step="sunbeam_test",
        report=report,
        exchanges=[],
        download_failures=[],
        progress_events=[event.to_trace()],
    )

    assert session["progress_events"] == [event.to_trace()]


def test_append_progress_event_records_concise_trace():
    app = _streamlit_app()
    events = []
    event = ProgressEvent(
        run_id="uuid",
        run_type="diagnosis",
        phase="tool_call",
        status="running",
        message="Model requested get_artifact_file",
        tool_name="get_artifact_file",
        target="generated/sunbeam/output.log",
        raw={"not": "stored"},
    )

    app._append_progress_event(events, event)

    assert events == [
        {
            "run_id": "uuid",
            "run_type": "diagnosis",
            "phase": "tool_call",
            "status": "running",
            "message": "Model requested get_artifact_file",
            "tool_name": "get_artifact_file",
            "target": "generated/sunbeam/output.log",
            "created_at": event.created_at,
        }
    ]


def test_diagnosis_session_round_trips_triage_v2_fields(tmp_path):
    app = _streamlit_app()
    report = DiagnosisReport.from_dict(
        {
            "summary": "Timed out",
            "failure_surface": "Deploy timeout",
            "confidence": "supported",
            "root_cause": "RabbitMQ closed first.",
            "triage_confidence": "medium",
            "failure_timeline": [
                {
                    "timestamp": "10:42:29",
                    "source": "rabbitmq.log",
                    "location": "line 120",
                    "event": "RabbitMQ closed AMQP connection.",
                }
            ],
            "cascading_errors": [
                {
                    "path": "nova-api.log",
                    "line": 1242,
                    "excerpt": "oslo.messaging timeout",
                }
            ],
            "alternatives_considered": [
                {
                    "hypothesis": "Database outage",
                    "status": "less_likely",
                    "reason": "No DB errors near first failure timestamp.",
                }
            ],
            "missing_evidence": ["Need neutron-server timing."],
            "stop_reason": "sufficient_evidence",
        }
    )

    session = app._session_from_diagnosis(
        uuid="uuid",
        model="model/a",
        artifact_root=tmp_path / "uuid",
        output=tmp_path / "diagnostics.html",
        failed_step="sunbeam_test",
        report=report,
        exchanges=[],
        download_failures=[],
    )
    loaded = app._report_from_session(session)

    assert session["triage_confidence"] == "medium"
    assert loaded.failure_timeline[0].source == "rabbitmq.log"
    assert loaded.cascading_errors[0].path == "nova-api.log"
    assert loaded.alternatives_considered[0].hypothesis == "Database outage"
    assert loaded.missing_evidence == ["Need neutron-server timing."]
    assert loaded.stop_reason == "sufficient_evidence"


def test_arena_contender_label_hides_model_until_verdict():
    app = _streamlit_app()
    contender = {"contender_id": "A", "model": "model/a"}

    assert app._arena_contender_label(contender, reveal_model=False) == "Contender A"
    assert app._arena_contender_label(contender, reveal_model=True) == "Contender A - model/a"


def test_save_arena_verdict_persists_judged_snapshot_without_promoting_winner(tmp_path):
    app = _streamlit_app()
    artifact_root = tmp_path / "artifacts"
    save_session_snapshot(
        artifact_root,
        {
            "schema_version": 2,
            "session_id": "arena-sample",
            "session_type": "arena",
            "uuid": "sample-uuid",
            "updated_at": "2026-06-30T12:00:00Z",
            "summary": "Arena",
            "status": "completed",
            "contenders": [
                {"contender_id": "A", "model": "model/a", "report": {"summary": "A"}},
                {"contender_id": "B", "model": "model/b", "report": {"summary": "B"}},
            ],
        },
    )
    session = load_session_record(artifact_root, "arena-sample")["snapshot"]

    updated = app._save_arena_verdict(
        artifact_root,
        session,
        winner="B",
        notes="B had better evidence.",
        rubric={
            "A": {
                "root_cause": 2,
                "evidence": 2,
                "timeline": 1,
                "uncertainty": 2,
                "next_steps": 2,
            },
            "B": {
                "root_cause": 5,
                "evidence": 5,
                "timeline": 4,
                "uncertainty": 4,
                "next_steps": 5,
            },
        },
    )

    loaded = load_session_record(artifact_root, "arena-sample")
    assert updated["status"] == "judged"
    assert loaded["snapshot"]["verdict"]["winner"] == "B"
    assert [event["event"] for event in loaded["events"]] == ["arena_verdict_saved"]
    assert load_ui_session(artifact_root, "sample-uuid") is None


def test_persist_diagnosis_session_writes_legacy_and_v2_snapshot(tmp_path):
    app = _streamlit_app()
    artifact_root = tmp_path / "artifacts"
    session = {
        "uuid": "sample-uuid",
        "model": "model/a",
        "summary": "Diagnosis summary",
        "confidence": "supported",
        "updated_at": "2026-06-30T12:00:00Z",
        "chat": [],
    }

    app._persist_diagnosis_session(artifact_root, session)

    assert load_ui_session(artifact_root, "sample-uuid") == session
    loaded = load_session_record(artifact_root, "sample-uuid")
    assert loaded["snapshot"]["schema_version"] == 2
    assert loaded["snapshot"]["session_type"] == "diagnosis"
    assert loaded["snapshot"]["summary"] == "Diagnosis summary"
