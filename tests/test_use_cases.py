import shutil
from pathlib import Path

from typing_extensions import override

from sunbeam_triage.core.config import Config
from sunbeam_triage.core.llm import DiagnosisReport
from sunbeam_triage.core.llm_tool_loop import ModelToolProtocolError
from sunbeam_triage.core.sessions import load_session_record, save_session_snapshot
from sunbeam_triage.core.swift import MirrorManifest, SwiftObject
from sunbeam_triage.core.use_cases import (
    ArenaVerdictRequest,
    DiagnosisRunRequest,
    FollowupRequest,
    TriageUseCases,
    report_from_session,
    session_from_diagnosis,
)


class FakeMirror:
    def __init__(self):
        self.calls = []

    def mirror_uuid(self, uuid, **kwargs):
        self.calls.append({"uuid": uuid, **kwargs})
        return MirrorManifest(
            uuid=uuid,
            root=Path("unused"),
            objects=(
                SwiftObject(
                    name=f"{uuid}/generated/sunbeam/output.log", hash=None, bytes=1
                ),
            ),
        )


class FakeClient:
    def __init__(self, report=None, answer="follow-up answer"):
        self.report = report or DiagnosisReport(
            summary="Diagnosis summary",
            failure_surface="Deploy timeout",
            confidence="supported",
            root_cause="Readiness did not converge",
            triage_confidence="medium",
        )
        self.answer = answer
        self.exchanges = [{"request": {"model": "model/a"}, "response": {}}]
        self.diagnose_calls = []
        self.chat_calls = []

    def diagnose(self, evidence_text, **kwargs):
        self.diagnose_calls.append({"evidence_text": evidence_text, **kwargs})
        return self.report

    def chat(self, context, messages, **kwargs):
        self.chat_calls.append({"context": context, "messages": messages, **kwargs})
        return self.answer


class ProtocolFailingClient(FakeClient):
    @override
    def diagnose(self, evidence_text, **kwargs):
        del evidence_text, kwargs
        raise ModelToolProtocolError("model/a", "required", 1)


def _config(tmp_path):
    config = Config.load(None)
    config.paths.artifact_root = tmp_path / "artifacts"
    config.paths.output_pattern = str(tmp_path / "diagnostics-{uuid}.html")
    config.triage.quick_max_rounds = 2
    return config


def _copy_fixture(tmp_path):
    source = Path("tests/fixtures/sample_uuid")
    target = tmp_path / "artifacts" / "sample-uuid"
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target)
    return target


def test_run_diagnosis_persists_ui_and_v2_session_snapshots(tmp_path):
    _copy_fixture(tmp_path)
    config = _config(tmp_path)
    mirror = FakeMirror()
    client = FakeClient()
    progress_events = []
    use_cases = TriageUseCases(
        config,
        mirror_factory=lambda _swift, _artifact_root: mirror,
        client_factory=lambda _llm: client,
    )

    result = use_cases.run_diagnosis(
        DiagnosisRunRequest(
            uuid="sample-uuid",
            model="model/a",
            budget="quick",
            refresh=True,
            max_tool_rounds=1,
        ),
        progress_events=progress_events,
    )

    assert result.session["uuid"] == "sample-uuid"
    assert result.selected_uuid == "sample-uuid"
    assert result.clear_attachments is True
    assert (tmp_path / "diagnostics-sample-uuid.html").exists()
    assert mirror.calls[0]["continue_on_error"] is True
    assert mirror.calls[0]["refresh"] is True
    assert client.diagnose_calls[0]["max_tool_rounds"] == 1
    assert client.diagnose_calls[0]["session_id"] == "sample-uuid"
    assert client.diagnose_calls[0]["artifact_root"] == (
        tmp_path / "artifacts" / "sample-uuid"
    )

    loaded = load_session_record(config.paths.artifact_root, "sample-uuid")
    assert loaded is not None
    assert loaded["snapshot"]["session_type"] == "diagnosis"
    assert loaded["snapshot"]["summary"] == "Diagnosis summary"


def test_run_diagnosis_offline_skips_mirror_and_uses_precomputed_report(tmp_path):
    _copy_fixture(tmp_path)
    config = _config(tmp_path)
    mirror = FakeMirror()
    client = FakeClient()
    report = DiagnosisReport(
        summary="Precomputed diagnosis",
        failure_surface="Deploy timeout",
        confidence="supported",
        root_cause="Readiness did not converge",
    )
    use_cases = TriageUseCases(
        config,
        mirror_factory=lambda _swift, _artifact_root: mirror,
        client_factory=lambda _llm: client,
    )

    result = use_cases.run_diagnosis(
        DiagnosisRunRequest(
            uuid="sample-uuid",
            model="model/a",
            offline=True,
            max_tool_rounds=1,
            precomputed_report=report,
        ),
        progress_events=[],
    )

    assert mirror.calls == []
    assert client.diagnose_calls == []
    assert result.session["summary"] == "Precomputed diagnosis"
    assert result.evidence_item_count > 0


def test_run_diagnosis_preserves_attachments_on_error(tmp_path):
    _copy_fixture(tmp_path)
    config = _config(tmp_path)
    mirror = FakeMirror()
    client = FakeClient()

    def failing_collect(_artifact_root, _uuid):
        raise RuntimeError("evidence failed")

    use_cases = TriageUseCases(
        config,
        mirror_factory=lambda _swift, _artifact_root: mirror,
        client_factory=lambda _llm: client,
        evidence_collector_factory=failing_collect,
    )

    result = use_cases.run_diagnosis(
        DiagnosisRunRequest(uuid="sample-uuid", model="model/a", budget="quick"),
        progress_events=[],
    )

    assert result.error == "evidence failed"
    assert result.selected_uuid == "sample-uuid"
    assert result.clear_attachments is False
    loaded = load_session_record(config.paths.artifact_root, "sample-uuid")
    assert loaded is not None
    assert loaded["snapshot"]["status"] == "error"


def test_run_diagnosis_persists_required_tool_protocol_failure(tmp_path):
    _copy_fixture(tmp_path)
    config = _config(tmp_path)
    client = ProtocolFailingClient()
    client.exchanges = [{"request": {}, "response": {"tool_calls": []}}]
    use_cases = TriageUseCases(
        config,
        mirror_factory=lambda _swift, _artifact_root: FakeMirror(),
        client_factory=lambda _llm: client,
    )

    result = use_cases.run_diagnosis(
        DiagnosisRunRequest(uuid="sample-uuid", model="model/a"),
        progress_events=[],
    )

    assert result.error == (
        "Model model/a returned no tool call while tool_choice=required"
    )
    assert result.session["error_type"] == "model_tool_protocol"
    assert result.session["investigation_status"] == "model_protocol_error"
    assert result.session["verdict_source"] == "none"
    assert "root_cause" not in result.session


def test_send_followup_appends_chat_and_persists_session(tmp_path):
    artifact_root = _copy_fixture(tmp_path)
    config = _config(tmp_path)
    client = FakeClient(answer="inspect kubectl logs")
    report = DiagnosisReport(
        summary="Timed out",
        failure_surface="Deploy timeout",
        confidence="supported",
        root_cause="Readiness did not converge",
    )
    session = session_from_diagnosis(
        uuid="sample-uuid",
        model="model/a",
        artifact_root=artifact_root,
        output=tmp_path / "diagnostics-sample-uuid.html",
        failed_step="sunbeam_deploy",
        report=report,
        exchanges=[],
        download_failures=[],
    )
    use_cases = TriageUseCases(config, client_factory=lambda _llm: client)

    result = use_cases.send_followup(
        FollowupRequest(
            session=session,
            prompt="What should I inspect next?",
            attachments=[
                {
                    "path": "generated/sunbeam/output.log",
                    "line": 2,
                    "text": "wait timed out",
                }
            ],
        ),
        progress_events=[],
    )

    assert result.answer == "inspect kubectl logs"
    assert result.clear_attachments is True
    assert result.session["chat"][-2]["content"] == "What should I inspect next?"
    assert result.session["chat"][-1]["content"] == "inspect kubectl logs"
    assert "Attached Context:" in client.chat_calls[0]["context"]
    loaded = load_session_record(config.paths.artifact_root, "sample-uuid")
    assert loaded is not None
    assert loaded["snapshot"]["chat"][-1]["content"] == "inspect kubectl logs"


def test_session_conversion_round_trips_report_v2_fields(tmp_path):
    report = DiagnosisReport.from_dict({
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
        "missing_evidence": ["Need neutron-server timing."],
        "stop_reason": "sufficient_evidence",
    })

    session = session_from_diagnosis(
        uuid="uuid",
        model="model/a",
        artifact_root=tmp_path / "uuid",
        output=tmp_path / "diagnostics.html",
        failed_step="sunbeam_test",
        report=report,
        exchanges=[],
        download_failures=[],
    )
    loaded = report_from_session(session)

    assert session["triage_confidence"] == "medium"
    assert loaded.failure_timeline[0].source == "rabbitmq.log"
    assert loaded.missing_evidence == ["Need neutron-server timing."]
    assert loaded.stop_reason == "sufficient_evidence"


def test_save_arena_verdict_persists_judged_snapshot_and_event(tmp_path):
    config = _config(tmp_path)
    artifact_root = config.paths.artifact_root
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
    record = load_session_record(artifact_root, "arena-sample")
    assert record is not None

    updated = TriageUseCases(config).save_arena_verdict(
        ArenaVerdictRequest(
            session=record["snapshot"],
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
    )

    loaded = load_session_record(artifact_root, "arena-sample")
    assert loaded is not None
    assert updated["status"] == "judged"
    assert loaded["snapshot"]["verdict"]["winner"] == "B"
    assert loaded["events"][-1]["event"] == "arena_verdict_saved"
