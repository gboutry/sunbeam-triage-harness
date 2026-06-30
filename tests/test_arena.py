import json
from pathlib import Path

from sunbeam_triage.arena import ArenaOptions, ArenaRunner, render_arena_html
from sunbeam_triage.config import Config
from sunbeam_triage.llm import DiagnosisReport
from sunbeam_triage.sessions import load_session_record
from sunbeam_triage.triage_state import TriageLoopOptions


class FakeArenaClient:
    def __init__(self, model, report=None, error=None):
        self.model = model
        self.report = report
        self.error = error
        self.exchanges = [
            {
                "request": {"model": model, "messages": [{"role": "user", "content": ""}]},
                "response": {"usage": {"total_tokens": 12, "cost": 0.001}},
            }
        ]
        self.calls = []

    def diagnose(self, evidence_text, **kwargs):
        self.calls.append({"evidence_text": evidence_text, **kwargs})
        if self.error:
            raise self.error
        return self.report


class FakeClientFactory:
    def __init__(self):
        self.clients = {}

    def __call__(self, llm_config):
        client = self.clients[llm_config.model]
        return client


def _config(tmp_path):
    config = Config.load(None)
    config.paths.artifact_root = tmp_path / "artifacts"
    config.paths.output_pattern = str(tmp_path / "diagnostics-{uuid}.html")
    config.llm.api_key = "token"
    config.triage.max_tool_result_chars = 1000
    return config


def _copy_fixture(tmp_path):
    source = Path("tests/fixtures/sample_uuid")
    target = tmp_path / "artifacts" / "sample-uuid"
    target.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(source, target)
    return target


def _report(summary, root_cause):
    return DiagnosisReport(
        summary=summary,
        failure_surface="Deploy timeout",
        confidence="supported",
        root_cause=root_cause,
        triage_confidence="medium",
    )


def test_arena_runner_executes_models_sequentially_with_isolated_sessions(tmp_path):
    artifact_root = _copy_fixture(tmp_path)
    config = _config(tmp_path)
    factory = FakeClientFactory()
    factory.clients = {
        "model/a": FakeArenaClient("model/a", _report("A summary", "A cause")),
        "model/b": FakeArenaClient("model/b", _report("B summary", "B cause")),
    }
    runner = ArenaRunner(config, client_factory=factory)

    session = runner.run(
        "sample-uuid",
        ArenaOptions(
            models=["model/a", "model/b"],
            budget="quick",
            output=tmp_path / "arena.html",
            triage_options=TriageLoopOptions(
                max_rounds=1,
                hard_max_rounds=20,
                stall_limit=1,
                min_evidence_items=1,
                max_tool_result_chars=1000,
            ),
        ),
    )

    assert session["session_type"] == "arena"
    assert session["status"] == "completed"
    assert session["artifact_root"] == str(artifact_root)
    assert [item["contender_id"] for item in session["contenders"]] == ["A", "B"]
    assert [item["model"] for item in session["contenders"]] == ["model/a", "model/b"]
    assert session["contenders"][0]["report"]["summary"] == "A summary"
    assert session["contenders"][1]["report"]["summary"] == "B summary"
    assert Path(session["output"]).exists()
    assert "Arena: sample-uuid" in Path(session["output"]).read_text(encoding="utf-8")

    first_call = factory.clients["model/a"].calls[0]
    second_call = factory.clients["model/b"].calls[0]
    assert first_call["evidence_text"] == second_call["evidence_text"]
    assert first_call["session_id"].endswith("-A")
    assert second_call["session_id"].endswith("-B")
    assert first_call["artifact_root"] == artifact_root
    assert first_call["triage_options"].max_rounds == 1

    loaded = load_session_record(config.paths.artifact_root, session["session_id"])
    assert loaded["snapshot"]["session_id"] == session["session_id"]
    assert [event["event"] for event in loaded["events"]] == [
        "arena_started",
        "contender_started",
        "contender_completed",
        "contender_started",
        "contender_completed",
        "arena_completed",
    ]


def test_arena_runner_persists_partial_failures(tmp_path):
    _copy_fixture(tmp_path)
    config = _config(tmp_path)
    factory = FakeClientFactory()
    factory.clients = {
        "model/a": FakeArenaClient("model/a", _report("A summary", "A cause")),
        "model/b": FakeArenaClient("model/b", error=RuntimeError("model exploded")),
    }
    runner = ArenaRunner(config, client_factory=factory)

    session = runner.run(
        "sample-uuid",
        ArenaOptions(
            models=["model/a", "model/b"],
            budget="quick",
            output=tmp_path / "arena.html",
        ),
    )

    assert session["status"] == "completed_with_errors"
    assert session["contenders"][0]["status"] == "completed"
    assert session["contenders"][1]["status"] == "failed"
    assert session["contenders"][1]["error"] == "model exploded"
    loaded = load_session_record(config.paths.artifact_root, session["session_id"])
    assert loaded["snapshot"]["status"] == "completed_with_errors"


def test_render_arena_html_includes_blind_labels_and_model_reveal_after_verdict():
    session = {
        "uuid": "sample-uuid",
        "session_id": "arena-sample",
        "status": "judged",
        "contenders": [
            {
                "contender_id": "A",
                "model": "model/a",
                "status": "completed",
                "report": {"summary": "A summary", "root_cause": "A cause"},
            },
            {
                "contender_id": "B",
                "model": "model/b",
                "status": "completed",
                "report": {"summary": "B summary", "root_cause": "B cause"},
            },
        ],
        "verdict": {"winner": "B", "notes": "B wins"},
    }

    html = render_arena_html(session)

    assert "Contender A" in html
    assert "Contender B" in html
    assert "model/a" in html
    assert "model/b" in html
    assert "Winner: B" in html


def test_render_arena_html_hides_model_names_before_verdict():
    session = {
        "uuid": "sample-uuid",
        "session_id": "arena-sample",
        "status": "completed",
        "contenders": [
            {
                "contender_id": "A",
                "model": "model/a",
                "status": "completed",
                "report": {"summary": "A summary", "root_cause": "A cause"},
            }
        ],
    }

    html = render_arena_html(session)

    assert "Contender A" in html
    assert "model/a" not in html
