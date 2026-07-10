import json
from pathlib import Path

from sunbeam_triage.core.evidence import EvidenceCollector


def test_evidence_collector_identifies_first_failed_non_cleanup_step():
    root = Path("tests/fixtures/sample_uuid")

    pack = EvidenceCollector(root, "sample-uuid").collect()

    assert pack.run.run_id == 202
    assert pack.run.branch == "main"
    assert pack.failed_step.name == "sunbeam_deploy"
    assert pack.failed_step.number == 2
    assert pack.failed_step.family == "sunbeam"
    assert any("wait timed out" in item.excerpt for item in pack.evidence)
    assert all("Failed to collect files" not in item.excerpt for item in pack.evidence)
    assert any(item.kind == "juju-status" for item in pack.evidence)
    assert pack.step_selection is not None
    assert pack.step_selection.selected.name == "sunbeam_deploy"
    assert pack.step_selection.confidence == "high"


def test_evidence_pack_prompt_is_bounded_and_contains_file_references():
    root = Path("tests/fixtures/sample_uuid")

    prompt = (
        EvidenceCollector(root, "sample-uuid").collect().to_prompt_text(max_chars=4000)
    )

    assert "Solutions Run UUID: sample-uuid" in prompt
    assert "generated/sunbeam/output.log:" in prompt
    assert "Claim only what the evidence supports" in prompt


def test_evidence_pack_prompt_includes_triggered_probe_findings(tmp_path):
    _write_jobs(tmp_path, failed_step_name="Deploy sunbeam")
    (tmp_path / "generated/sunbeam").mkdir(parents=True)
    (tmp_path / "generated/sunbeam/output.log").write_text(
        "Application 'k8s' is not ready: TimeoutError('wait timed out')\n"
        "message='Unready Pods: kube-system/coredns-a'\n",
        encoding="utf-8",
    )

    prompt = EvidenceCollector(tmp_path, "uuid").collect().to_prompt_text()

    assert "Deterministic Probes:" in prompt
    assert "k8s_not_ready" in prompt
    assert "Unready Pods: kube-system/coredns-a" in prompt


def test_evidence_redacts_obvious_secret_values():
    root = Path("tests/fixtures/sample_uuid")

    pack = EvidenceCollector(root, "sample-uuid").collect()
    prompt = pack.to_prompt_text()

    assert "super-secret-value" not in prompt
    assert "OS_PASSWORD=" not in prompt


def test_evidence_redacts_bearer_tokens(tmp_path):
    _write_jobs(tmp_path, failed_step_name="Deploy sunbeam")
    (tmp_path / "generated/sunbeam").mkdir(parents=True)
    (tmp_path / "generated/sunbeam/output.log").write_text(
        "ERROR Authorization: Bearer sk-or-v1-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        encoding="utf-8",
    )

    pack = EvidenceCollector(tmp_path, "uuid").collect()
    prompt = pack.to_prompt_text()

    assert "sk-or-v1-aaaaaaaa" not in prompt
    assert "Authorization: Bearer" not in prompt


def test_evidence_collector_uses_sunbeam_artifacts_for_human_named_sunbeam_steps(
    tmp_path,
):
    _write_jobs(tmp_path, failed_step_name="Deploy sunbeam")
    (tmp_path / "generated/sunbeam").mkdir(parents=True)
    (tmp_path / "generated/sunbeam/output.log").write_text(
        "wait timed out\n",
        encoding="utf-8",
    )
    (tmp_path / "generated/sunbeam/juju_status_openstack.txt").write_text(
        "Model benign header\nkeystone/0 waiting idle\n",
        encoding="utf-8",
    )
    (tmp_path / "generated/github-runner/run.log").write_text(
        "Process completed with exit code 1\n",
        encoding="utf-8",
    )

    pack = EvidenceCollector(tmp_path, "uuid").collect()

    assert pack.failed_step.family == "sunbeam"
    assert any(item.kind == "sunbeam-output" for item in pack.evidence)
    assert any(item.kind == "juju-status" for item in pack.evidence)


def test_evidence_collector_keeps_generic_steps_generic_without_sunbeam_artifacts(
    tmp_path,
):
    _write_jobs(tmp_path, failed_step_name="Build wheels")
    (tmp_path / "generated/github-runner").mkdir(parents=True, exist_ok=True)
    (tmp_path / "generated/github-runner/run.log").write_text(
        "Process completed with exit code 1\n",
        encoding="utf-8",
    )

    pack = EvidenceCollector(tmp_path, "uuid").collect()

    assert pack.failed_step.family == "generic"
    assert {item.kind for item in pack.evidence} == {"github-runner"}


def test_evidence_prompt_lists_maas_step_profile_primary_artifacts(tmp_path):
    _write_jobs(tmp_path, failed_step_name="maas")
    path = tmp_path / "generated/github-runner/run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Process completed with exit code 1\n", encoding="utf-8")

    prompt = EvidenceCollector(tmp_path, "uuid").collect().to_prompt_text()

    assert "Step Profile: sunbeam_maas_deploy" in prompt
    assert "primary_missing=" in prompt


def test_evidence_collector_records_rejected_cleanup_failures(tmp_path):
    jobs = {
        "jobs": [
            {
                "run_id": 202,
                "workflow_name": "workflow",
                "head_branch": "main",
                "name": "Run the pipeline",
                "steps": [
                    {"name": "sunbeam_deploy", "conclusion": "failure", "number": 1},
                    {
                        "name": "Report the job to weebl",
                        "conclusion": "failure",
                        "number": 2,
                    },
                ],
            }
        ]
    }
    path = tmp_path / "generated/github-runner/jobs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(jobs), encoding="utf-8")
    (tmp_path / "generated/sunbeam").mkdir(parents=True)
    (tmp_path / "generated/sunbeam/output.log").write_text(
        "wait timed out\n",
        encoding="utf-8",
    )

    pack = EvidenceCollector(tmp_path, "uuid").collect()

    assert pack.failed_step.name == "sunbeam_deploy"
    assert pack.step_selection is not None
    assert [step.name for step in pack.step_selection.rejected_cleanup] == [
        "Report the job to weebl"
    ]


def test_evidence_collector_uses_cleanup_only_failure_with_low_confidence(tmp_path):
    jobs = {
        "jobs": [
            {
                "run_id": 202,
                "workflow_name": "workflow",
                "head_branch": "main",
                "name": "Run the pipeline",
                "steps": [
                    {"name": "Collect logs", "conclusion": "failure", "number": 1},
                ],
            }
        ]
    }
    path = tmp_path / "generated/github-runner/jobs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(jobs), encoding="utf-8")
    (tmp_path / "generated/github-runner/run.log").write_text(
        "Process completed with exit code 1\n",
        encoding="utf-8",
    )

    pack = EvidenceCollector(tmp_path, "uuid").collect()

    assert pack.failed_step.name == "Collect logs"
    assert pack.step_selection is not None
    assert pack.step_selection.confidence == "low"


def test_status_summary_omits_benign_headers_but_keeps_status_signal(tmp_path):
    status = tmp_path / "status.txt"
    status.write_text(
        "\n".join([
            "Model Controller Cloud",
            "openstack controller localhost",
            "App Version Status",
            "keystone active",
            "nova active",
            "glance active",
            "cinder active",
            "ovn active",
            "rabbitmq active",
            "mysql active",
            "placement active",
            "horizon active",
            "placement-mysql-router waiting idle",
        ]),
        encoding="utf-8",
    )

    items = EvidenceCollector(tmp_path, "uuid")._summarize_status(
        "status.txt",
        "juju-status",
    )

    assert [item.excerpt for item in items] == ["placement-mysql-router waiting idle"]


def _write_jobs(root: Path, *, failed_step_name: str) -> None:
    jobs = {
        "jobs": [
            {
                "run_id": 202,
                "workflow_name": "workflow",
                "head_branch": "main",
                "html_url": "https://example.invalid/job",
                "started_at": "2026-06-01T10:00:00Z",
                "completed_at": "2026-06-01T10:10:00Z",
                "name": "Run the pipeline",
                "steps": [
                    {
                        "name": failed_step_name,
                        "conclusion": "failure",
                        "number": 1,
                    }
                ],
            }
        ]
    }
    path = root / "generated/github-runner/jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs), encoding="utf-8")
