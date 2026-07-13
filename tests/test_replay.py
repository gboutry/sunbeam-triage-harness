import hashlib
import shutil

from sunbeam_triage.core.config import Config
from sunbeam_triage.core.evaluation import EvaluationCase
from sunbeam_triage.core.llm import (
    CausalAssessment,
    CausalClaim,
    DiagnosisReport,
    ReportEvidence,
)
from sunbeam_triage.core.replay import ReplayAttemptError, replay_case


class FakeReplayClient:
    def __init__(self, _config):
        self.exchanges = []

    def diagnose(self, _evidence, **_kwargs):
        trigger = ReportEvidence(
            path="generated/sunbeam/output.log",
            line=2,
            excerpt="wait timed out",
            role="failure_trigger",
        )
        cause = ReportEvidence(
            path="generated/sunbeam/output.log",
            line=1,
            excerpt="CNI configuration missing before readiness check",
            role="root_cause",
        )
        return DiagnosisReport(
            summary="The operation timed out.",
            failure_surface="The deployment step timed out.",
            confidence="supported",
            root_cause="CNI configuration was missing.",
            evidence=[trigger, cause],
            causal_assessment=CausalAssessment(
                failure_trigger=CausalClaim(
                    "The deployment step timed out.",
                    "confirmed",
                    evidence_ids=[trigger.id],
                ),
                root_cause=CausalClaim(
                    "CNI configuration was missing.",
                    "supported",
                    evidence_ids=[cause.id],
                ),
            ),
        )


class FailingReplayClient:
    def __init__(self, _config):
        self.exchanges = [{"request": {"messages": []}, "response": {}}]

    def diagnose(self, _evidence, **_kwargs):
        raise RuntimeError("provider failed")


def test_replay_case_persists_isolated_attempt(tmp_path):
    artifact_root = tmp_path / "artifacts"
    case_root = artifact_root / "sample-uuid"
    shutil.copytree("tests/fixtures/sample_uuid", case_root)
    manifest = case_root / ".sunbeam-triage-manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    config = Config.load(None, cli_artifact_root=artifact_root)
    config.llm.model = "z-ai/glm-5.2:floor"
    case = EvaluationCase(
        uuid="sample-uuid",
        phase="sunbeam_deploy",
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        accepted_root_causes=("CNI configuration.*missing",),
        required_evidence=("output\\.log.*wait timed out",),
    )
    output_dir = tmp_path / "replay"
    output_dir.mkdir()

    session = replay_case(
        config,
        case,
        attempt=2,
        output_dir=output_dir,
        client_factory=FakeReplayClient,
    )

    assert session["session_id"] == "sample-uuid--r2"
    assert session["session_type"] == "replay"
    assert session["score"]["passed"] is True
    assert (output_dir / "sample-uuid--r2.json").exists()
    assert (output_dir / "sample-uuid--r2.html").exists()


def test_replay_case_preserves_exchanges_on_failure(tmp_path):
    artifact_root = tmp_path / "artifacts"
    case_root = artifact_root / "sample-uuid"
    shutil.copytree("tests/fixtures/sample_uuid", case_root)
    manifest = case_root / ".sunbeam-triage-manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    config = Config.load(None, cli_artifact_root=artifact_root)
    case = EvaluationCase(
        uuid="sample-uuid",
        phase="sunbeam_deploy",
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        accepted_root_causes=(),
    )

    try:
        replay_case(
            config,
            case,
            attempt=1,
            output_dir=tmp_path,
            client_factory=FailingReplayClient,
        )
    except ReplayAttemptError as exc:
        assert str(exc) == "provider failed"
        assert len(exc.exchanges) == 1
    else:
        raise AssertionError("expected ReplayAttemptError")
