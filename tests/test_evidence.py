from pathlib import Path

from sunbeam_triage.evidence import EvidenceCollector


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


def test_evidence_pack_prompt_is_bounded_and_contains_file_references():
    root = Path("tests/fixtures/sample_uuid")

    prompt = EvidenceCollector(root, "sample-uuid").collect().to_prompt_text(max_chars=4000)

    assert "Solutions Run UUID: sample-uuid" in prompt
    assert "generated/sunbeam/output.log:" in prompt
    assert "Claim only what the evidence supports" in prompt


def test_evidence_redacts_obvious_secret_values():
    root = Path("tests/fixtures/sample_uuid")

    pack = EvidenceCollector(root, "sample-uuid").collect()
    prompt = pack.to_prompt_text()

    assert "super-secret-value" not in prompt
    assert "OS_PASSWORD=<redacted>" in prompt
