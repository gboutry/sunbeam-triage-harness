import json
import subprocess
import sys
from pathlib import Path


def test_cli_offline_writes_diagnostics_html(tmp_path):
    source = Path("tests/fixtures/sample_uuid")
    artifact_root = tmp_path / "artifacts"
    target = artifact_root / "sample-uuid"
    target.parent.mkdir(parents=True)
    subprocess.run(["cp", "-R", str(source), str(target)], check=True)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[paths]
artifact_root = "{artifact_root}"
output_pattern = "{tmp_path}/diagnostics-{{uuid}}.html"
""".strip(),
        encoding="utf-8",
    )

    llm_json = json.dumps({
        "summary": "Offline summary",
        "failure_surface": "sunbeam_deploy failed",
        "confidence": "supported",
        "root_cause": "Timeout",
        "evidence": [],
        "candidate_mechanisms": [],
        "recommendations": [],
        "unknowns": [],
    })
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sunbeam_triage.cli",
            "sample-uuid",
            "--offline",
            "--config",
            str(config_path),
            "--llm-json",
            llm_json,
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    output = tmp_path / "diagnostics-sample-uuid.html"
    assert output.exists()
    assert "Offline summary" in output.read_text(encoding="utf-8")
    assert str(output) in result.stdout


def test_cli_logs_stage_model_and_result(tmp_path):
    source = Path("tests/fixtures/sample_uuid")
    artifact_root = tmp_path / "artifacts"
    target = artifact_root / "sample-uuid"
    target.parent.mkdir(parents=True)
    subprocess.run(["cp", "-R", str(source), str(target)], check=True)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[llm]
model = "configured/test-model"

[paths]
artifact_root = "{artifact_root}"
output_pattern = "{tmp_path}/diagnostics-{{uuid}}.html"
""".strip(),
        encoding="utf-8",
    )
    llm_json = json.dumps({
        "summary": "Offline summary",
        "failure_surface": "sunbeam_deploy failed",
        "confidence": "supported",
        "root_cause": "Timeout",
        "evidence": [],
        "candidate_mechanisms": [],
        "recommendations": [],
        "unknowns": [],
    })

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sunbeam_triage.cli",
            "sample-uuid",
            "--offline",
            "--config",
            str(config_path),
            "--llm-json",
            llm_json,
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "[stage] config" in result.stderr
    assert "[model] configured/test-model" in result.stderr
    assert "[stage] mirror skipped (offline)" in result.stderr
    assert "[stage] evidence" in result.stderr
    assert (
        "[result] failed_step=sunbeam_deploy family=sunbeam evidence_items="
        in result.stderr
    )
    assert "[stage] diagnosis using supplied JSON" in result.stderr
    assert "[result] confidence=supported summary=Offline summary" in result.stderr
    assert "[stage] render" in result.stderr
    assert str(tmp_path / "diagnostics-sample-uuid.html") in result.stdout


def test_cli_accepts_budget_profile_argument():
    from sunbeam_triage.cli import build_parser

    args = build_parser().parse_args(["sample-uuid", "--budget", "quick"])

    assert args.budget == "quick"


def test_cli_accepts_max_tool_rounds_argument():
    from sunbeam_triage.cli import build_parser

    args = build_parser().parse_args(["sample-uuid", "--max-tool-rounds", "9"])

    assert args.max_tool_rounds == 9
