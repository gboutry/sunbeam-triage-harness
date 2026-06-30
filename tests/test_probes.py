import io
import tarfile
from pathlib import Path

from sunbeam_triage.probes import run_preflight_probes


def test_k8s_not_ready_probe_is_not_applicable_without_k8s_timeout(tmp_path):
    _write_output(tmp_path, "Process completed with exit code 1\n")

    results = run_preflight_probes(tmp_path, "uuid")

    assert [result.name for result in results] == ["k8s_not_ready"]
    assert results[0].status == "not_applicable"
    assert results[0].findings == []


def test_k8s_not_ready_probe_extracts_output_status_and_later_convergence(tmp_path):
    _write_output(
        tmp_path,
        "\n".join(
            [
                "Jun 29 17:00:16 snorlax stderr:",
                "Application 'k8s' is not ready: TimeoutError('wait timed out')",
                "message='Unready Pods: kube-system/coredns-a, kube-system/coredns-b'",
            ]
        ),
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

    result = run_preflight_probes(tmp_path, "uuid")[0]

    assert result.status == "triggered"
    assert result.summary == "K8s readiness timeout probe collected deterministic evidence."
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

    result = run_preflight_probes(tmp_path, "uuid")[0]

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

    result = run_preflight_probes(tmp_path, "uuid")[0]

    journal_findings = [
        finding for finding in result.findings if finding.category == "sosreport_journal"
    ]
    assert len(journal_findings) == 2
    assert all("sosreport-snorlax" in finding.path for finding in journal_findings)
    assert any("cni plugin not initialized" in finding.excerpt for finding in journal_findings)


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
                "Jun 29 17:04:17 chespin k8s.k8sd[1]: "
                "Failed to watch configmap\n"
            )
        },
    )

    result = run_preflight_probes(tmp_path, "uuid")[0]

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

    result = run_preflight_probes(tmp_path, "uuid")[0]

    journal_findings = [
        finding for finding in result.findings if finding.category == "sosreport_journal"
    ]
    assert len(journal_findings) == 1
    assert "Failed to watch configmap" in journal_findings[0].excerpt


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
