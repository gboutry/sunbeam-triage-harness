from pathlib import Path

from sunbeam_triage.core.step_profile_extractor import extract_profiles
from sunbeam_triage.core.step_profiles import STEP_PROFILES, profile_for_step


def test_committed_sunbeam_step_profiles_cover_requested_steps():
    for name in (
        "sunbeam_deploy",
        "sunbeam_prepare_env",
        "sunbeam_maas_deploy",
        "sunbeam_enable_plugins_all",
        "sunbeam_launch_vm",
        "sunbeam_test_plugins",
        "sunbeam_test_with_validation_plugin",
        "sunbeam_test_with_validation_plugin_no_features",
    ):
        profile = profile_for_step(name)
        assert profile is not None
        assert profile.primary_artifacts
        assert profile.source_path == f"{name}.md"

    assert (
        "generated/sunbeam/output.log"
        in STEP_PROFILES["sunbeam_deploy"].primary_artifacts
    )
    assert any(
        probe.pattern == "FAILED|Ran:|Failed:"
        for probe in STEP_PROFILES[
            "sunbeam_test_with_validation_plugin_no_features"
        ].probes
    )
    assert any(
        probe.pattern == "migration failed"
        for probe in STEP_PROFILES["sunbeam_test_plugins"].probes
    )
    assert sum(len(profile.probes) for profile in STEP_PROFILES.values()) >= 100


def test_extract_profiles_from_markdown_tables_and_grep_blocks(tmp_path):
    steps = tmp_path / "steps"
    steps.mkdir()
    (steps / "sunbeam_custom.md").write_text(
        "\n".join([
            "# Step Knowledge: sunbeam_custom",
            "",
            "## Swift Artifacts",
            "| Path | Description | When to check |",
            "|---|---|---|",
            "| `generated/sunbeam/output.log` | primary | Always |",
            "",
            "## Grep Patterns",
            "```bash",
            "grep -E \"ERROR|FAILED\" generated/sunbeam/output.log",
            "```",
            "",
            "## Known Failure Patterns",
            "### Pattern 1: sample failure",
            "**Evidence to look for:**",
            "- `generated/sunbeam/output.log`: `wait timed out`",
        ]),
        encoding="utf-8",
    )

    profiles = extract_profiles(Path(steps))

    profile = profiles["sunbeam_custom"]
    assert profile.primary_artifacts == ("generated/sunbeam/output.log",)
    assert profile.probes[0].pattern == "ERROR|FAILED"
    assert profile.known_patterns == ("sample failure",)
    assert profile.source_path == "sunbeam_custom.md"


def test_extract_profiles_handles_grep_options_with_arguments(tmp_path):
    steps = tmp_path / "steps"
    steps.mkdir()
    (steps / "sunbeam_custom.md").write_text(
        "\n".join([
            "| Path | Description | When to check |",
            "|---|---|---|",
            "| `generated/github-runner/run.log` | primary | Always |",
            "",
            "grep -n -C 5 \"AssertionError\" generated/github-runner/run.log",
            "grep -A 40 \"<TestClassName>.*FAILED\" validation.log",
        ]),
        encoding="utf-8",
    )

    profile = extract_profiles(Path(steps))["sunbeam_custom"]

    assert [probe.pattern for probe in profile.probes] == [
        "AssertionError",
        "<TestClassName>.*FAILED",
    ]
