import json
import shutil
import subprocess
import sys
from pathlib import Path


def _report(summary):
    return json.dumps({
        "summary": summary,
        "failure_surface": "Deploy timeout",
        "confidence": "supported",
        "root_cause": "Timeout",
        "evidence": [],
        "candidate_mechanisms": [],
        "recommendations": [],
        "unknowns": [],
    })


def test_arena_cli_run_uses_config_models_and_writes_report(tmp_path):
    source = Path("tests/fixtures/sample_uuid")
    artifact_root = tmp_path / "artifacts"
    shutil.copytree(source, artifact_root / "sample-uuid")
    output = tmp_path / "arena.html"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[arena]
models = ["model/a", "model/b"]

[paths]
artifact_root = "{artifact_root}"
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sunbeam_triage.cli.arena",
            "run",
            "sample-uuid",
            "--offline",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--llm-json",
            _report("A summary"),
            "--llm-json",
            _report("B summary"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert str(output) in result.stdout
    assert "[model] model/a" in result.stderr
    assert "[model] model/b" in result.stderr
    html = output.read_text(encoding="utf-8")
    assert "A summary" in html
    assert "B summary" in html


def test_arena_cli_model_override_replaces_config_roster(tmp_path):
    source = Path("tests/fixtures/sample_uuid")
    artifact_root = tmp_path / "artifacts"
    shutil.copytree(source, artifact_root / "sample-uuid")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[arena]
models = ["configured/a", "configured/b"]

[paths]
artifact_root = "{artifact_root}"
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sunbeam_triage.cli.arena",
            "run",
            "sample-uuid",
            "--offline",
            "--config",
            str(config_path),
            "--models",
            "override/a,override/b",
            "--output",
            str(tmp_path / "arena-override.html"),
            "--llm-json",
            _report("A summary"),
            "--llm-json",
            _report("B summary"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "[model] override/a" in result.stderr
    assert "[model] override/b" in result.stderr
    assert "configured/a" not in result.stderr


def test_arena_cli_export_writes_judged_jsonl(tmp_path):
    artifact_root = tmp_path / "artifacts"
    session_dir = artifact_root / ".sunbeam-triage" / "sessions"
    session_dir.mkdir(parents=True)
    (session_dir / "arena.json").write_text(
        json.dumps({
            "schema_version": 2,
            "session_id": "arena",
            "session_type": "arena",
            "uuid": "sample-uuid",
            "updated_at": "2026-06-30T12:00:00Z",
            "status": "judged",
            "contenders": [
                {
                    "contender_id": "A",
                    "model": "model/a",
                    "report": {"summary": "A summary"},
                },
                {
                    "contender_id": "B",
                    "model": "model/b",
                    "report": {"summary": "B summary"},
                },
            ],
            "verdict": {
                "winner": "A",
                "notes": "A wins",
                "rubric": {"A": {"root_cause": 5}},
            },
        }),
        encoding="utf-8",
    )
    output = tmp_path / "export.jsonl"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[paths]
artifact_root = "{artifact_root}"
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sunbeam_triage.cli.arena",
            "export",
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "[result] exported=1" in result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["winner"] == "A"
