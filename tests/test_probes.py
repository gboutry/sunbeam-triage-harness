import io
import tarfile
from pathlib import Path

from sunbeam_triage.core.probes import run_preflight_probes


def test_k8s_not_ready_probe_is_not_applicable_without_k8s_timeout(tmp_path):
    _write_output(tmp_path, "Process completed with exit code 1\n")

    results = run_preflight_probes(tmp_path, "uuid")

    assert [result.name for result in results] == [
        "timeout_outcome",
        "machine_add_timeout",
        "k8s_not_ready",
        "juju_lost_unit",
        "juju_migration",
        "certificate_csr_churn",
        "package_install_failure",
        "relation_blockers",
        "workload_crash_recovery",
        "juju_error_units",
    ]
    assert results[0].status == "not_applicable"
    assert results[0].findings == []


def test_machine_add_probe_links_timeout_to_slow_apt_and_successful_control(
    tmp_path,
):
    _write_output(
        tmp_path,
        "\n".join([
            (
                "12:00:00 Calling ssh ubuntu@host03.maas sudo sunbeam cluster join "
                "--token super-secret-token"
            ),
            "12:05:06 Timeout adding machine 10.0.0.3 to Juju",
            "12:10:00 TIMED OUT to add machine 10.0.0.3",
            "12:12:00 Node joined cluster: host05",
        ]),
    )
    _write_sosreport(
        tmp_path / "generated/sunbeam/sosreport-host03-2026-07-22-a.tar.xz",
        {
            "sosreport-host03-2026-07-22-a/var/log/cloud-init-output.log": (
                "Running apt-get update\nFetched 80.1 MB in 11min 1s (121 kB/s)\n"
            ),
            "sosreport-host03-2026-07-22-a/sos_commands/process/ps_auxfwww": (
                "root 1 0 /bin/bash /var/lib/juju/nonce.sh\n"
                "root 2 1 apt-get -q update\n"
            ),
            "sosreport-host03-2026-07-22-a/etc/apt/apt.conf.d/90curtin-aptproxy": (
                'Acquire::http::Proxy "http://10.239.8.11:8000";\n'
            ),
            "sosreport-host03-2026-07-22-a/home/ubuntu/snap/openstack/common/"
            "logs/join.log": (
                "Starting step 'Add machine'\n"
                "Finished running step 'Add machine' ResultType.FAILED\n"
            ),
        },
    )
    _write_sosreport(
        tmp_path / "generated/sunbeam/sosreport-host05-2026-07-22-b.tar.xz",
        {
            "sosreport-host05-2026-07-22-b/var/log/cloud-init-output.log": (
                "Running apt-get update\n"
                "Fetched 80.1 MB in 1min 56s (690 kB/s)\n"
                "Starting Juju machine agent\n"
            ),
            "sosreport-host05-2026-07-22-b/etc/apt/apt.conf.d/90curtin-aptproxy": (
                'Acquire::http::Proxy "http://10.239.8.11:8000";\n'
            ),
            "sosreport-host05-2026-07-22-b/home/ubuntu/snap/openstack/common/"
            "logs/join.log": (
                "Starting step 'Add machine'\n"
                "Finished running step 'Add machine' ResultType.COMPLETED\n"
            ),
        },
    )

    results = run_preflight_probes(tmp_path, "uuid")
    result = _probe_by_name(results, "machine_add_timeout")

    assert result.status == "triggered"
    assert "APT still running or exceeding the deadline" in result.summary
    categories = {finding.category for finding in result.findings}
    assert {
        "machine_add_timeout",
        "remote_add_machine_failed",
        "apt_process_running",
        "apt_deadline_exceeded",
        "apt_proxy_path",
        "successful_join_control",
    } <= categories
    assert "super-secret-token" not in " ".join(
        finding.excerpt for finding in result.findings
    )
    timeout = _probe_by_name(results, "timeout_outcome")
    assert "per node" in timeout.summary


def test_machine_add_probe_does_not_call_proxy_configuration_the_cause(tmp_path):
    _write_output(
        tmp_path,
        "12:00:00 ssh ubuntu@host03.maas sudo sunbeam cluster join\n"
        "12:05:06 Timeout adding machine 10.0.0.3 to Juju\n",
    )
    _write_sosreport(
        tmp_path / "generated/sunbeam/sosreport-host03-2026-07-22-a.tar.xz",
        {
            "sosreport-host03-2026-07-22-a/etc/apt/apt.conf.d/90curtin-aptproxy": (
                'Acquire::http::Proxy "http://10.239.8.11:8000";\n'
            ),
            "sosreport-host03-2026-07-22-a/home/ubuntu/snap/openstack/common/"
            "logs/join.log": (
                "Starting step 'Add machine'\n"
                "Finished running step 'Add machine' ResultType.FAILED\n"
            ),
        },
    )

    result = _probe_by_name(
        run_preflight_probes(tmp_path, "uuid"), "machine_add_timeout"
    )

    assert "APT itself crossed" in result.missing_evidence[-1]
    assert not any(
        finding.category in {"apt_process_running", "apt_deadline_exceeded"}
        for finding in result.findings
    )


def test_timeout_outcome_probe_flags_post_timeout_completion(tmp_path):
    _write_output(
        tmp_path,
        "wait timed out after 1199s\nNode joined cluster: node-2\n",
    )

    result = _probe_by_name(
        run_preflight_probes(tmp_path, "uuid"),
        "timeout_outcome",
    )

    assert "possible false negative" in result.summary
    assert any(
        finding.category == "post_timeout_completion" for finding in result.findings
    )


def test_k8s_not_ready_probe_extracts_output_status_and_later_convergence(tmp_path):
    _write_output(
        tmp_path,
        "\n".join([
            "Jun 29 17:00:16 snorlax stderr:",
            "Application 'k8s' is not ready: TimeoutError('wait timed out')",
            "message='Unready Pods: kube-system/coredns-a, kube-system/coredns-b'",
        ]),
    )
    _write(
        tmp_path,
        "generated/sunbeam/juju_status_openstack-machines.txt",
        "Model Controller Cloud\nk8s 1.32 active Ready\nk8s/0 active idle Ready\n",
    )
    _write(
        tmp_path,
        "generated/sunbeam/kubectl_get_node.txt",
        "NAME STATUS ROLES AGE VERSION\nsnorlax Ready control-plane 38m v1.32\n",
    )
    _write(
        tmp_path,
        "generated/sunbeam/kubectl_get_pod.txt",
        "NAMESPACE NAME READY STATUS RESTARTS AGE\n"
        "kube-system coredns-abc 1/1 Running 0 20m\n",
    )

    result = _probe_by_name(run_preflight_probes(tmp_path, "uuid"), "k8s_not_ready")

    assert result.status == "triggered"
    assert (
        result.summary
        == "K8s readiness timeout probe collected deterministic evidence."
    )
    assert any(finding.category == "failure_surface" for finding in result.findings)
    assert any("Unready Pods" in finding.excerpt for finding in result.findings)
    assert any(finding.category == "later_convergence" for finding in result.findings)


def test_k8s_not_ready_probe_extracts_embedded_status_from_long_failure_line(tmp_path):
    _write_output(
        tmp_path,
        "Application 'k8s' is not ready: TimeoutError(\"wait timed out\\n"
        "app_status=StatusInfo(current='waiting', "
        "message='Waiting for Cluster token', since='29 Jun 2026 16:56:49Z')\")\n",
    )

    result = _probe_by_name(run_preflight_probes(tmp_path, "uuid"), "k8s_not_ready")

    assert any(
        finding.category == "embedded_status"
        and finding.excerpt == "Waiting for Cluster token"
        for finding in result.findings
    )


def test_k8s_not_ready_probe_reads_bounded_matching_sosreport_journals(tmp_path):
    _write_output(
        tmp_path,
        "Jun 29 17:00:16 snorlax Application 'k8s' is not ready: "
        "TimeoutError('wait timed out')\n",
    )
    archive = tmp_path / "generated/sunbeam/sosreport-snorlax-2026-06-29-abcd.tar.xz"
    _write_sosreport(
        archive,
        {
            "sosreport-snorlax-2026-06-29-abcd/"
            "sos_commands/kubernetes/journalctl_--no-pager_--unit_snap.k8s": (
                "Jun 29 17:09:58 snorlax k8s.containerd[1]: "
                "failed to load cni during init: no network config found\n"
                "Jun 29 17:10:24 snorlax k8s.kubelet[2]: "
                "network is not ready: cni plugin not initialized\n"
            )
        },
    )

    result = _probe_by_name(run_preflight_probes(tmp_path, "uuid"), "k8s_not_ready")

    journal_findings = [
        finding
        for finding in result.findings
        if finding.category == "sosreport_journal"
    ]
    assert len(journal_findings) == 2
    assert all("sosreport-snorlax" in finding.path for finding in journal_findings)
    assert any(
        "cni plugin not initialized" in finding.excerpt for finding in journal_findings
    )


def test_k8s_not_ready_probe_infers_host_from_nearby_failed_join_command(tmp_path):
    _write_output(
        tmp_path,
        "Command failed: ssh chespin.maas -- sunbeam cluster join token\n"
        "Application 'k8s' is not ready: TimeoutError('wait timed out')\n",
    )
    _write_sosreport(
        tmp_path / "generated/sunbeam/sosreport-bibarel-2026-06-29-abcd.tar.xz",
        {
            "sosreport-bibarel-2026-06-29-abcd/"
            "sos_commands/kubernetes/journalctl_--no-pager_--unit_snap.k8s": (
                "Jun 29 17:02:53 bibarel k8s.k8sd[1]: Waiting for node to be ready\n"
            )
        },
    )
    _write_sosreport(
        tmp_path / "generated/sunbeam/sosreport-chespin-2026-06-29-abcd.tar.xz",
        {
            "sosreport-chespin-2026-06-29-abcd/"
            "sos_commands/kubernetes/journalctl_--no-pager_--unit_snap.k8s": (
                "Jun 29 17:04:17 chespin k8s.k8sd[1]: Failed to watch configmap\n"
            )
        },
    )

    result = _probe_by_name(run_preflight_probes(tmp_path, "uuid"), "k8s_not_ready")

    journal_paths = [
        finding.path
        for finding in result.findings
        if finding.category == "sosreport_journal"
    ]
    assert journal_paths
    assert all("sosreport-chespin" in path for path in journal_paths)


def test_k8s_not_ready_probe_deduplicates_repeated_journal_messages(tmp_path):
    _write_output(
        tmp_path,
        "Jun 29 17:01:22 chespin Application 'k8s' is not ready\n",
    )
    archive = tmp_path / "generated/sunbeam/sosreport-chespin-2026-06-29-abcd.tar.xz"
    repeated = "\n".join(
        f"Jun 29 17:0{minute}:00 chespin k8s.k8sd[1]: "
        'Failed to watch configmap err="watch error event: too old resource version"'
        for minute in range(3)
    )
    _write_sosreport(
        archive,
        {
            "sosreport-chespin-2026-06-29-abcd/"
            "sos_commands/kubernetes/journalctl_--no-pager_--unit_snap.k8s": repeated
        },
    )

    result = _probe_by_name(run_preflight_probes(tmp_path, "uuid"), "k8s_not_ready")

    journal_findings = [
        finding
        for finding in result.findings
        if finding.category == "sosreport_journal"
    ]
    assert len(journal_findings) == 1
    assert "Failed to watch configmap" in journal_findings[0].excerpt


def test_juju_lost_unit_probe_extracts_lost_unit_and_missing_leader(tmp_path):
    _write_output(
        tmp_path,
        "ERROR leader for application-k8s not found\n",
    )
    _write(
        tmp_path,
        "generated/sunbeam/juju_status_openstack-machines.txt",
        "k8s/0 unknown lost 0 10.195.18.1 6443/tcp agent lost, "
        "see 'juju show-status-log k8s/0'\n",
    )
    _write(
        tmp_path,
        "generated/sunbeam/sunbeam_cluster_list.txt",
        "│ node1 │ running │ active  │ unknown │         │ active  │\n",
    )

    result = _probe_by_name(run_preflight_probes(tmp_path, "uuid"), "juju_lost_unit")

    assert result.status == "triggered"
    assert "Juju unit-agent/leader loss" in result.summary
    assert any(finding.category == "final_status" for finding in result.findings)
    assert any(finding.category == "missing_leader" for finding in result.findings)
    assert any(
        finding.category == "control_plane_unknown" for finding in result.findings
    )
    assert any("juju show-status-log k8s/0" in item for item in result.missing_evidence)


def test_juju_migration_probe_separates_observed_from_failed_migration(tmp_path):
    _write(
        tmp_path,
        "generated/sunbeam/juju_debug_log_openstack-machines.txt",
        "\n".join([
            "15:06:43 INFO migration phase is now QUIESCE",
            "15:07:07 INFO migration phase is now SUCCESS",
            '15:07:07 INFO juju.worker.deployer stopped "k8s/0", err: <nil>',
            '15:07:07 INFO juju.worker.deployer start "unit-k8s-0"',
        ]),
    )

    result = _probe_by_name(run_preflight_probes(tmp_path, "uuid"), "juju_migration")

    assert result.status == "triggered"
    assert "observed; failure is unconfirmed" in result.summary
    assert any(finding.category == "migration_event" for finding in result.findings)
    assert any(finding.category == "unit_lifecycle" for finding in result.findings)
    assert any(
        "No direct failed-migration evidence" in item
        for item in result.missing_evidence
    )


def test_juju_migration_probe_reads_archived_unit_agent_log(tmp_path):
    archive = tmp_path / "generated/sunbeam/sosreport-node1-2026-06-29-abcd.tar.xz"
    _write_sosreport(
        archive,
        {
            "sosreport-node1/var/log/juju/unit-k8s-0.log": (
                "15:07:00 migration phase is now REAP\n"
                "15:07:01 migration failed; agent.conf left unchanged\n"
            )
        },
    )

    result = _probe_by_name(run_preflight_probes(tmp_path, "uuid"), "juju_migration")

    assert "direct failure evidence" in result.summary
    assert any(
        finding.category == "failed_migration_signal"
        and "unit-k8s-0.log" in finding.path
        for finding in result.findings
    )


def test_workload_crash_recovery_probe_records_crash_and_recovery_counterevidence(
    tmp_path,
):
    archive = tmp_path / "generated/sunbeam/sosreport-node1-2026-06-29-abcd.tar.xz"
    _write_sosreport(
        archive,
        {
            "sosreport-node1-2026-06-29-abcd/var/log/apport.log": (
                "INFO: apport 2026-06-29 17:37:44,365: called for global pid "
                "22844, signal 11\n"
                "INFO: apport 2026-06-29 17:37:44,367: executable: "
                '/snap/k8s/5279/bin/k8s (command line "/snap/k8s/5279/bin/k8sd")\n'
            ),
            "sosreport-node1-2026-06-29-abcd/"
            "sos_commands/systemd/systemctl_list-units": (
                "snap.k8s.k8sd.service loaded active running\n"
            ),
            "sosreport-node1-2026-06-29-abcd/"
            "sos_commands/kubernetes/journalctl_--no-pager_--unit_snap.k8s": (
                "Jun 29 15:21:55 node1 k8s.k8sd[22844]: "
                "checking service arguments for drift\n"
                "Jun 29 18:21:55 node1 k8s.k8sd[4114976]: "
                "checking service arguments for drift\n"
            ),
        },
    )

    result = _probe_by_name(
        run_preflight_probes(tmp_path, "uuid"),
        "workload_crash_recovery",
    )

    assert result.status == "triggered"
    assert "counter-evidence was found" in result.summary
    assert any(finding.category == "workload_crash" for finding in result.findings)
    assert any(
        finding.category == "recovery_counterevidence" for finding in result.findings
    )
    assert not any("15:21:55" in finding.excerpt for finding in result.findings)
    assert any(
        "not sufficient evidence for Juju unit lost" in item
        for item in result.missing_evidence
    )


def _probe_by_name(results, name):
    for result in results:
        if result.name == name:
            return result
    raise AssertionError(f"missing probe {name}")


def _write_output(root: Path, text: str) -> None:
    _write(root, "generated/sunbeam/output.log", text)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_sosreport(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:xz") as archive:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
