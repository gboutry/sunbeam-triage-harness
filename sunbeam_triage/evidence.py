from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .probes import ProbeResult, run_preflight_probes


CLEANUP_STEP_NAMES = {
    "Collect logs",
    "log_collection",
    "Upload logs to swift",
    "Cancel Testflinger jobs",
    "Clean up public cloud",
    "Clean existing openstack",
    "Report the job to weebl",
    "Release environment lock",
    "Complete job",
}

ERROR_PATTERNS = re.compile(
    r"wait timed out|Traceback|CalledProcessError|Broken pipe|"
    r"Process completed with exit code|Command failed|terraform.*failed",
    re.IGNORECASE,
)

NOISE_PATTERNS = re.compile(
    r"Failed to collect files|validation\*\.log|Juju command \"machines\" not supported",
    re.IGNORECASE,
)

SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:PASSWORD|TOKEN|SECRET|API_KEY|ACCESS_KEY)[A-Z0-9_]*)"
    r"\s*=\s*([^\s'\";]+)"
)


@dataclass(frozen=True)
class RunInfo:
    run_id: int | None
    repository: str
    branch: str
    workflow: str
    html_url: str | None
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class FailedStep:
    name: str
    number: int | None
    conclusion: str | None
    started_at: str | None
    completed_at: str | None
    family: str


@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    path: str
    line: int | None
    excerpt: str


@dataclass(frozen=True)
class EvidencePack:
    uuid: str
    root: Path
    run: RunInfo
    failed_step: FailedStep
    evidence: tuple[EvidenceItem, ...]
    probe_results: tuple[ProbeResult, ...] = ()

    def to_prompt_text(self, max_chars: int = 60000) -> str:
        parts = [
            "You are diagnosing a Sunbeam CI failure.",
            "Claim only what the evidence supports. Separate evidence from inference.",
            "Classify claims as confirmed, supported, or speculative.",
            f"Solutions Run UUID: {self.uuid}",
            f"Run ID: {self.run.run_id}",
            f"Branch: {self.run.branch}",
            f"Workflow: {self.run.workflow}",
            f"Failed Step: {self.failed_step.name}",
            "",
            "Evidence:",
        ]
        for item in self.evidence:
            line = "" if item.line is None else f":{item.line}"
            parts.append(f"- [{item.kind}] {item.path}{line}: {item.excerpt}")
        probe_lines = _probe_prompt_lines(self.probe_results)
        if probe_lines:
            parts.extend(["", "Deterministic Probes:", *probe_lines])
        text = "\n".join(parts)
        if len(text) > max_chars:
            return text[: max_chars - 200] + "\n\n[Evidence truncated by harness]\n"
        return text


class EvidenceCollector:
    def __init__(self, root: Path, uuid: str):
        self.root = Path(root)
        self.uuid = uuid

    def collect(self) -> EvidencePack:
        jobs = self._read_json("generated/github-runner/jobs.json")
        job = self._primary_job(jobs)
        step = self._first_failed_step(job)
        run = RunInfo(
            run_id=job.get("run_id"),
            repository="canonical/sqa-cloud-deployment-pipeline",
            branch=job.get("head_branch", "unknown"),
            workflow=job.get("workflow_name", "unknown"),
            html_url=job.get("html_url"),
            started_at=job.get("started_at"),
            completed_at=job.get("completed_at"),
        )
        failed = FailedStep(
            name=step.get("name", "unknown"),
            number=step.get("number"),
            conclusion=step.get("conclusion"),
            started_at=step.get("started_at"),
            completed_at=step.get("completed_at"),
            family=self._step_family(step.get("name", "")),
        )
        evidence = self._collect_evidence(failed)
        probe_results = run_preflight_probes(self.root, self.uuid)
        return EvidencePack(
            uuid=self.uuid,
            root=self.root,
            run=run,
            failed_step=failed,
            evidence=tuple(evidence),
            probe_results=probe_results,
        )

    def _read_json(self, rel: str) -> dict[str, Any]:
        path = self.root / rel
        if not path.exists():
            raise RuntimeError(f"Missing required artifact: {rel}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _primary_job(jobs: dict[str, Any]) -> dict[str, Any]:
        candidates = [
            job
            for job in jobs.get("jobs", [])
            if job.get("name") == "Run the pipeline"
            or any(step.get("conclusion") == "failure" for step in job.get("steps", []))
        ]
        if not candidates:
            raise RuntimeError("jobs.json does not contain a failed pipeline job")
        return candidates[0]

    @staticmethod
    def _first_failed_step(job: dict[str, Any]) -> dict[str, Any]:
        for step in job.get("steps", []):
            if step.get("conclusion") != "failure":
                continue
            name = step.get("name", "")
            if name in CLEANUP_STEP_NAMES or name.startswith("Post "):
                continue
            return step
        raise RuntimeError("No failed non-cleanup step found in jobs.json")

    def _step_family(self, name: str) -> str:
        if name.startswith("sunbeam_"):
            return "sunbeam"
        if self._has_sunbeam_artifacts():
            return "sunbeam"
        return "generic"

    def _has_sunbeam_artifacts(self) -> bool:
        sunbeam_root = self.root / "generated/sunbeam"
        if not sunbeam_root.is_dir():
            return False
        return any(
            (self.root / rel).exists()
            for rel in (
                "generated/sunbeam/output.log",
                "generated/sunbeam/juju_status_openstack.txt",
                "generated/sunbeam/juju_status_openstack-machines.txt",
                "generated/sunbeam/kubectl_get_pod.txt",
                "generated/sunbeam/sunbeam_cluster_list.txt",
            )
        )

    def _collect_evidence(self, failed: FailedStep) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        if failed.family == "sunbeam":
            evidence.extend(self._scan_log("generated/sunbeam/output.log", "sunbeam-output"))
            evidence.extend(
                self._summarize_status(
                    "generated/sunbeam/juju_status_openstack.txt", "juju-status"
                )
            )
            evidence.extend(
                self._summarize_status(
                    "generated/sunbeam/juju_status_openstack-machines.txt",
                    "juju-status",
                )
            )
            evidence.extend(
                self._summarize_status(
                    "generated/sunbeam/kubectl_get_pod.txt", "kubernetes-status"
                )
            )
            evidence.extend(
                self._scan_log("generated/github-runner/run.log", "github-runner")
            )
            evidence.extend(
                self._summarize_status(
                    "generated/sunbeam/sunbeam_cluster_list.txt", "sunbeam-cluster"
                )
            )
        else:
            evidence.extend(
                self._scan_log("generated/github-runner/run.log", "github-runner")
            )
        return evidence[:80]

    def _scan_log(self, rel: str, kind: str) -> list[EvidenceItem]:
        path = self.root / rel
        if not path.exists():
            return []
        items: list[EvidenceItem] = []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, start=1):
            if (
                not ERROR_PATTERNS.search(line)
                and not SECRET_ASSIGNMENT.search(line)
            ) or NOISE_PATTERNS.search(line):
                continue
            items.append(
                EvidenceItem(
                    kind=kind,
                    path=rel,
                    line=number,
                    excerpt=_redact(line.strip())[:1000],
                )
            )
        return items

    def _summarize_status(self, rel: str, kind: str) -> list[EvidenceItem]:
        path = self.root / rel
        if not path.exists():
            return []
        items: list[EvidenceItem] = []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if re.search(
                r"\b(blocked|error|waiting|maintenance|lost|unknown|executing)\b",
                stripped,
                re.IGNORECASE,
            ):
                items.append(
                    EvidenceItem(
                        kind=kind,
                        path=rel,
                        line=number,
                        excerpt=_redact(stripped)[:1000],
                    )
                )
        return items[:30]


def _redact(text: str) -> str:
    return SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", text)


def _probe_prompt_lines(probe_results: tuple[ProbeResult, ...]) -> list[str]:
    lines: list[str] = []
    for result in probe_results:
        if result.status == "not_applicable":
            continue
        lines.append(f"- [{result.name}] {result.status}: {result.summary}")
        for finding in result.findings[:20]:
            line = "" if finding.line is None else f":{finding.line}"
            lines.append(
                f"  - [{finding.category}] {finding.path}{line}: {finding.excerpt}"
            )
        for missing in result.missing_evidence:
            lines.append(f"  - [missing] {missing}")
    return lines
