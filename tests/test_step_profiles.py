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
    assert sum(len(profile.probes) for profile in STEP_PROFILES.values()) >= 80


def test_committed_sunbeam_step_profiles_drop_high_risk_static_probes():
    risky_patterns = {
        "15:24:5",
        "08:52:22\\|08:52:56",
        "<configure-start-minute>",
        "<TestClassName>.*FAILED",
        "for",
    }
    case_hosts = ("chespin", "behaim", "anonster", "crustle", "ledian.maas")
    patterns = {
        probe.pattern
        for profile in STEP_PROFILES.values()
        for probe in profile.probes
    }

    assert patterns.isdisjoint(risky_patterns)
    assert not any(
        host in pattern for host in case_hosts for pattern in patterns
    )


def test_committed_sunbeam_step_profiles_classify_navigation_and_targeted_reads():
    vault_probes = [
        probe
        for probe in STEP_PROFILES["sunbeam_enable_plugins_all"].probes
        if probe.pattern == "vault"
    ]
    assert vault_probes
    assert all(probe.category == "navigation" for probe in vault_probes)
    assert all(probe.read_class == "broad_search" for probe in vault_probes)

    deploy_profile = STEP_PROFILES["sunbeam_deploy"]
    assert any(
        probe.pattern == "cluster:3: Reconcile event=<RelationJoinedEvent"
        for probe in deploy_profile.probes
    )
    assert any(
        read.selector == "var/log/syslog$"
        for read in deploy_profile.targeted_reads
    )


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
            'grep -E "ERROR|FAILED" generated/sunbeam/output.log',
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
    assert profile.probes[0].source_line == 10
    assert profile.probes[0].source_text == (
        'grep -E "ERROR|FAILED" generated/sunbeam/output.log'
    )
    assert profile.known_patterns == ("sample failure",)
    assert profile.source_path == "sunbeam_custom.md"


def test_extract_profiles_handles_grep_options_with_arguments_and_prose(tmp_path):
    steps = tmp_path / "steps"
    steps.mkdir()
    (steps / "sunbeam_custom.md").write_text(
        "\n".join([
            "| Path | Description | When to check |",
            "|---|---|---|",
            "| `generated/github-runner/run.log` | primary | Always |",
            "",
            "STDOUT is `b''` (grep for `ubuntu@` never matched)",
            "```bash",
            'grep -n -C 5 "AssertionError" generated/github-runner/run.log',
            'grep -A 40 "real.*FAILED" validation.log',
            "```",
        ]),
        encoding="utf-8",
    )

    profile = extract_profiles(Path(steps))["sunbeam_custom"]

    assert [probe.pattern for probe in profile.probes] == [
        "AssertionError",
        "real.*FAILED",
    ]


def test_extract_profiles_sanitizes_risky_probes_and_guidance(tmp_path):
    steps = tmp_path / "steps"
    steps.mkdir()
    (steps / "sunbeam_custom.md").write_text(
        "\n".join([
            "| Path | Description | When to check |",
            "|---|---|---|",
            "| `generated/sunbeam/output.log` | primary | Always |",
            "```bash",
            'grep "15:24:5" generated/sunbeam/output.log',
            'grep "<TestClassName>.*FAILED" generated/sunbeam/output.log',
            'grep "ledian.maas\\|wait timed out after 1799\\|openstack-hypervisor/3\\|cinder-volume/3\\|microceph/3" generated/sunbeam/output.log',
            'grep "cluster:3: Reconcile event=<RelationJoinedEvent" generated/sunbeam/output.log',
            'grep "vault" generated/sunbeam/output.log',
            'tar -xJOf sosreport.tar.xz "*/var/log/syslog" | grep -n "var/log/syslog$"',
            "```",
        ]),
        encoding="utf-8",
    )

    profile = extract_profiles(Path(steps))["sunbeam_custom"]

    assert [probe.pattern for probe in profile.probes] == [
        "wait timed out after 1799\\|openstack-hypervisor/\\d+\\|cinder-volume/\\d+\\|microceph/\\d+",
        "cluster:3: Reconcile event=<RelationJoinedEvent",
        "vault",
    ]
    assert profile.probes[-1].category == "navigation"
    assert profile.probes[-1].read_class == "broad_search"
    assert profile.targeted_reads[0].selector == "var/log/syslog$"
