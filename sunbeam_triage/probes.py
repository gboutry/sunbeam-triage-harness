from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


K8S_NOT_READY = re.compile(
    r"Application 'k8s' is not ready|k8s.*wait timed out|wait timed out.*k8s",
    re.IGNORECASE,
)
STATUS_SIGNAL = re.compile(
    r"Unready Pods|Waiting for Cluster token|current='waiting'|message='[^']+'",
    re.IGNORECASE,
)
STATUS_MESSAGE = re.compile(r"message='([^']+)'")
K8S_STATUS_MESSAGE = re.compile(r"Unready Pods|Waiting for Cluster token", re.IGNORECASE)
K8S_READY_SIGNAL = re.compile(r"\bk8s(?:/\d+)?\b.*\b(active|Ready)\b", re.IGNORECASE)
NODE_READY_SIGNAL = re.compile(r"\bReady\b", re.IGNORECASE)
POD_READY_SIGNAL = re.compile(r"\bcoredns\b.*\bRunning\b", re.IGNORECASE)
CLUSTER_READY_SIGNAL = re.compile(r"\brunning\b.*\bactive\b", re.IGNORECASE)
JOURNAL_SIGNAL = re.compile(
    r"coredns|dnsrebalancer|cni plugin not initialized|network is not ready|"
    r"Starting etcd learner|Successfully promoted etcd learner|"
    r"Failed to watch configmap|Waiting for Cluster token|Waiting for node to be ready|"
    r"failed to load cni|not ready|readyz|context deadline exceeded|"
    r"failed to compute desired number of replicas",
    re.IGNORECASE,
)
JUJU_LOST_UNIT = re.compile(
    r"\bk8s/\d+\b.*\bunknown\b.*\blost\b|agent lost",
    re.IGNORECASE,
)
K8S_LEADER_MISSING = re.compile(r"leader for application-k8s not found", re.IGNORECASE)
CONTROL_UNKNOWN = re.compile(r"running.*active.*unknown|unknown.*active", re.IGNORECASE)
MIGRATION_SIGNAL = re.compile(
    r"migration phase.*\b(QUIESCE|IMPORT|PROCESSRELATIONS|VALIDATION|SUCCESS|NONE)\b",
    re.IGNORECASE,
)
FAILED_MIGRATION_SIGNAL = re.compile(
    r"invalid entity|invalid password|password invalid|agent\.conf.*unchanged|migration.*failed",
    re.IGNORECASE,
)
K8S_UNIT_LIFECYCLE = re.compile(
    r"stopped \"k8s/0\"|start(?:ing)? .*unit-k8s-0|Starting unit workers for \"k8s/0\"",
    re.IGNORECASE,
)
K8SD_CRASH = re.compile(r"apport.*signal|k8sd|/snap/k8s/.*/bin/k8s", re.IGNORECASE)
K8SD_RECOVERY = re.compile(
    r"snap\.k8s\.k8sd\.service.*active.*running|k8s\.k8sd\[\d+\].*checking service arguments",
    re.IGNORECASE,
)
TIMESTAMP_HOST = re.compile(
    r"^(?P<month>[A-Z][a-z]{2})\s+\d+\s+\d{2}:\d{2}:\d{2}\s+(?P<host>[^\s:]+)"
)
JOIN_COMMAND_HOST = re.compile(
    r"\bssh\b.*?\b(?P<host>[A-Za-z0-9-]+)\.maas\b.*?\bsunbeam\b.*?\bcluster\b.*?\bjoin\b"
)
SOSREPORT_NAME = re.compile(r"^sosreport-(?P<host>.+)-\d{4}-\d{2}-\d{2}-[^.]+\.tar")


@dataclass(frozen=True)
class ProbeFinding:
    category: str
    path: str
    line: int | None
    excerpt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "path": self.path,
            "line": self.line,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class ProbeResult:
    name: str
    status: str
    summary: str
    findings: list[ProbeFinding] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "findings": [finding.to_dict() for finding in self.findings],
            "missing_evidence": list(self.missing_evidence),
        }


def run_preflight_probes(root: Path, uuid: str) -> tuple[ProbeResult, ...]:
    del uuid
    root = Path(root)
    return (
        _run_k8s_not_ready_probe(root),
        _run_juju_lost_unit_probe(root),
        _run_juju_migration_probe(root),
        _run_workload_crash_recovery_probe(root),
    )


def _run_k8s_not_ready_probe(root: Path) -> ProbeResult:
    output = root / "generated/sunbeam/output.log"
    if not output.exists():
        return ProbeResult(
            name="k8s_not_ready",
            status="not_applicable",
            summary="No Sunbeam output log was available.",
        )
    lines = output.read_text(encoding="utf-8", errors="replace").splitlines()
    trigger_lines = [
        number
        for number, line in enumerate(lines, start=1)
        if K8S_NOT_READY.search(line)
    ]
    if not trigger_lines:
        return ProbeResult(
            name="k8s_not_ready",
            status="not_applicable",
            summary="K8s readiness timeout pattern was not found.",
        )

    findings: list[ProbeFinding] = []
    hosts: set[str] = set()
    for line_number in trigger_lines[:4]:
        line = lines[line_number - 1].strip()
        host = _host_from_timestamped_line(line) or _host_from_nearby_join_command(
            lines,
            line_number,
        )
        if host:
            hosts.add(host)
        findings.append(
            ProbeFinding(
                category="failure_surface",
                path="generated/sunbeam/output.log",
                line=line_number,
                excerpt=_excerpt(line),
            )
        )
        findings.extend(_same_line_status_findings(line, line_number))
        findings.extend(_nearby_status_findings(lines, line_number))

    findings.extend(_later_convergence_findings(root))
    findings.extend(_sosreport_journal_findings(root, hosts))

    missing: list[str] = []
    if not any(finding.category == "sosreport_journal" for finding in findings):
        missing.append("No matching k8s sosreport journal snippets were found.")

    return ProbeResult(
        name="k8s_not_ready",
        status="triggered",
        summary="K8s readiness timeout probe collected deterministic evidence.",
        findings=findings[:40],
        missing_evidence=missing,
    )


def _nearby_status_findings(lines: list[str], line_number: int) -> list[ProbeFinding]:
    findings: list[ProbeFinding] = []
    start = line_number
    end = min(len(lines), line_number + 8)
    for number in range(start + 1, end + 1):
        line = lines[number - 1].strip()
        if STATUS_SIGNAL.search(line):
            findings.append(
                ProbeFinding(
                    category="embedded_status",
                    path="generated/sunbeam/output.log",
                    line=number,
                    excerpt=_excerpt(line),
                )
            )
    return findings


def _same_line_status_findings(line: str, line_number: int) -> list[ProbeFinding]:
    findings: list[ProbeFinding] = []
    seen: set[str] = set()
    for match in STATUS_MESSAGE.finditer(line):
        message = _excerpt(match.group(1))
        if not message or message in seen or not K8S_STATUS_MESSAGE.search(message):
            continue
        seen.add(message)
        findings.append(
            ProbeFinding(
                category="embedded_status",
                path="generated/sunbeam/output.log",
                line=line_number,
                excerpt=message,
            )
        )
    return findings


def _later_convergence_findings(root: Path) -> list[ProbeFinding]:
    findings: list[ProbeFinding] = []
    for rel in (
        "generated/sunbeam/juju_status_openstack-machines.txt",
        "generated/sunbeam/sunbeam_cluster_list.txt",
        "generated/sunbeam/kubectl_get_node.txt",
        "generated/sunbeam/kubectl_get_pod.txt",
    ):
        path = root / rel
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if _is_later_convergence_line(rel, stripped):
                findings.append(
                    ProbeFinding(
                        category="later_convergence",
                        path=rel,
                        line=number,
                        excerpt=_excerpt(stripped),
                    )
                )
                break
    return findings


def _is_later_convergence_line(rel: str, line: str) -> bool:
    if not line or line.startswith("NAME ") or line.startswith("NAMESPACE "):
        return False
    if rel.endswith("kubectl_get_pod.txt"):
        return bool(POD_READY_SIGNAL.search(line))
    if rel.endswith("kubectl_get_node.txt"):
        return bool(NODE_READY_SIGNAL.search(line))
    if rel.endswith("sunbeam_cluster_list.txt"):
        return bool(CLUSTER_READY_SIGNAL.search(line))
    return bool(K8S_READY_SIGNAL.search(line))


def _sosreport_journal_findings(root: Path, hosts: set[str]) -> list[ProbeFinding]:
    findings: list[ProbeFinding] = []
    seen: set[str] = set()
    for archive_path in sorted(root.rglob("sosreport-*.tar*")):
        if archive_path.name.endswith(".sha256") or not archive_path.is_file():
            continue
        archive_host = _host_from_archive_name(archive_path.name)
        if hosts and archive_host and archive_host not in hosts:
            continue
        archive_rel = archive_path.relative_to(root).as_posix()
        with tarfile.open(archive_path, "r:*") as archive:
            members = _journal_members(archive)
            archive_count = 0
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read(2_000_000 + 1)
                if b"\x00" in data[: min(len(data), 4096)]:
                    continue
                text = data[:2_000_000].decode("utf-8", errors="replace")
                normalized = _normalized_member_path(member.name) or member.name
                for line_number, line in enumerate(text.splitlines(), start=1):
                    stripped = line.strip()
                    if not JOURNAL_SIGNAL.search(stripped):
                        continue
                    key = _journal_dedupe_key(stripped)
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        ProbeFinding(
                            category="sosreport_journal",
                            path=f"{archive_rel}:{normalized}",
                            line=line_number,
                            excerpt=_excerpt(stripped),
                        )
                    )
                    archive_count += 1
                    if len(findings) >= 24:
                        return findings
                    if archive_count >= 8:
                        break
                if archive_count >= 8:
                    break
    return findings


def _journal_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    fallback: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        if not member.isfile():
            continue
        normalized = _normalized_member_path(member.name)
        if normalized == "sos_commands/kubernetes/journalctl_--no-pager_--unit_snap.k8s":
            members.append(member)
        elif normalized == "sos_commands/logs/journalctl_--no-pager_--boot":
            fallback.append(member)
    return members or fallback


def _run_juju_lost_unit_probe(root: Path) -> ProbeResult:
    findings: list[ProbeFinding] = []
    findings.extend(
        _text_file_findings(
            root,
            "generated/sunbeam/juju_status_openstack-machines.txt",
            JUJU_LOST_UNIT,
            "final_status",
        )
    )
    findings.extend(
        _text_file_findings(
            root,
            "generated/sunbeam/output.log",
            K8S_LEADER_MISSING,
            "missing_leader",
        )
    )
    findings.extend(
        _text_file_findings(
            root,
            "generated/sunbeam/sunbeam_cluster_list.txt",
            CONTROL_UNKNOWN,
            "control_plane_unknown",
        )
    )
    if not findings:
        return ProbeResult(
            name="juju_lost_unit",
            status="not_applicable",
            summary="No Juju k8s unit-agent loss pattern was found.",
        )
    return ProbeResult(
        name="juju_lost_unit",
        status="triggered",
        summary="Juju unit-agent/leader loss is a primary failure-surface candidate.",
        findings=findings[:20],
        missing_evidence=[
            "Need juju show-status-log k8s/0 or unit-k8s-0 agent logs to "
            "confirm why Juju marked the unit lost."
        ],
    )


def _run_juju_migration_probe(root: Path) -> ProbeResult:
    rels = (
        "generated/sunbeam/juju_debug_log_openstack-machines.txt",
        "generated/sunbeam/juju_debug_log_controller.txt",
    )
    findings: list[ProbeFinding] = []
    failed_findings: list[ProbeFinding] = []
    for rel in rels:
        findings.extend(_text_file_findings(root, rel, MIGRATION_SIGNAL, "migration_event"))
        findings.extend(_text_file_findings(root, rel, K8S_UNIT_LIFECYCLE, "unit_lifecycle"))
        failed_findings.extend(
            _text_file_findings(root, rel, FAILED_MIGRATION_SIGNAL, "failed_migration_signal")
        )
    if failed_findings:
        findings.extend(failed_findings)
    if not findings:
        return ProbeResult(
            name="juju_migration",
            status="not_applicable",
            summary="No Juju migration lifecycle evidence was found.",
        )
    if failed_findings:
        summary = "Juju migration lifecycle and direct failure evidence were found."
        missing: list[str] = []
    else:
        summary = "Juju migration lifecycle was observed; failure is unconfirmed."
        missing = [
            "No direct failed-migration evidence found; look for invalid "
            "entity/password or stale agent.conf evidence before confirming "
            "migration failure."
        ]
    return ProbeResult(
        name="juju_migration",
        status="triggered",
        summary=summary,
        findings=findings[:24],
        missing_evidence=missing,
    )


def _run_workload_crash_recovery_probe(root: Path) -> ProbeResult:
    crash_findings: list[ProbeFinding] = []
    crash_times: list[int] = []
    recovery_candidates: list[tuple[ProbeFinding, int | None]] = []
    for archive_path in sorted(root.rglob("sosreport-*.tar*")):
        if archive_path.name.endswith(".sha256") or not archive_path.is_file():
            continue
        archive_rel = archive_path.relative_to(root).as_posix()
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                normalized = _normalized_member_path(member.name)
                if not member.isfile() or not normalized:
                    continue
                if normalized not in {
                    "var/log/apport.log",
                    "sos_commands/systemd/systemctl_list-units",
                    "sos_commands/kubernetes/journalctl_--no-pager_--unit_snap.k8s",
                }:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                text = extracted.read(2_000_000).decode("utf-8", errors="replace")
                for line_number, line in enumerate(text.splitlines(), start=1):
                    stripped = line.strip()
                    path = f"{archive_rel}:{normalized}"
                    if normalized == "var/log/apport.log" and K8SD_CRASH.search(stripped):
                        line_time = _line_time_seconds(stripped)
                        if line_time is not None:
                            crash_times.append(line_time)
                        crash_findings.append(
                            ProbeFinding(
                                category="workload_crash",
                                path=path,
                                line=line_number,
                                excerpt=_excerpt(stripped),
                            )
                        )
                    elif K8SD_RECOVERY.search(stripped):
                        recovery_candidates.append(
                            (
                                ProbeFinding(
                                    category="recovery_counterevidence",
                                    path=path,
                                    line=line_number,
                                    excerpt=_excerpt(stripped),
                                ),
                                _line_time_seconds(stripped),
                            )
                        )
                    if len(crash_findings) >= 4 and len(recovery_candidates) >= 8:
                        break
    if not crash_findings:
        return ProbeResult(
            name="workload_crash_recovery",
            status="not_applicable",
            summary="No k8sd workload crash evidence was found.",
        )
    crash_time = min(crash_times) if crash_times else None
    recovery_findings = [
        finding
        for finding, recovery_time in recovery_candidates
        if recovery_time is None or crash_time is None or recovery_time >= crash_time
    ]
    if recovery_findings:
        summary = "k8sd crash evidence was found, and recovery counter-evidence was found."
    else:
        summary = "k8sd crash evidence was found; recovery counter-evidence was not found."
    return ProbeResult(
        name="workload_crash_recovery",
        status="triggered",
        summary=summary,
        findings=[*crash_findings[:4], *recovery_findings[:8]],
        missing_evidence=[
            "A workload crash is not sufficient evidence for Juju unit lost "
            "without unit-agent or status-log evidence connecting the events."
        ],
    )


def _text_file_findings(
    root: Path,
    rel: str,
    pattern: re.Pattern[str],
    category: str,
    *,
    limit: int = 8,
) -> list[ProbeFinding]:
    path = root / rel
    if not path.exists():
        return []
    findings: list[ProbeFinding] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not pattern.search(stripped):
            continue
        findings.append(
            ProbeFinding(
                category=category,
                path=rel,
                line=number,
                excerpt=_excerpt(stripped),
            )
        )
        if len(findings) >= limit:
            break
    return findings


def _normalized_member_path(name: str) -> str | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    parts = path.parts
    if parts[0].startswith("sosreport-"):
        parts = parts[1:]
    if not parts:
        return None
    return PurePosixPath(*parts).as_posix()


def _host_from_timestamped_line(line: str) -> str:
    match = TIMESTAMP_HOST.match(line)
    return match.group("host") if match else ""


def _host_from_nearby_join_command(lines: list[str], line_number: int) -> str:
    start = max(1, line_number - 12)
    for number in range(line_number - 1, start - 1, -1):
        match = JOIN_COMMAND_HOST.search(lines[number - 1])
        if match:
            return match.group("host")
    end = min(len(lines), line_number + 3)
    for number in range(line_number + 1, end + 1):
        match = JOIN_COMMAND_HOST.search(lines[number - 1])
        if match:
            return match.group("host")
    return ""


def _host_from_archive_name(name: str) -> str:
    match = SOSREPORT_NAME.match(name)
    return match.group("host") if match else ""


def _journal_dedupe_key(line: str) -> str:
    line = re.sub(r"^[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\S+\s+", "", line)
    line = re.sub(r"\[\d+\]", "[]", line)
    line = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", "<ts>", line)
    return line


def _line_time_seconds(line: str) -> int | None:
    match = re.search(r"\b(\d{2}):(\d{2}):(\d{2})\b", line)
    if not match:
        return None
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _excerpt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:500]
