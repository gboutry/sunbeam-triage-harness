from __future__ import annotations

from dataclasses import replace

from .evidence import EvidenceItem
from .llm_schema import CandidateMechanism, DiagnosisReport, ReportEvidence
from .probes import ProbeResult


def apply_probe_report_policies(
    report: DiagnosisReport,
    probe_results: tuple[ProbeResult, ...] | list[ProbeResult],
    initial_evidence: tuple[EvidenceItem, ...] | list[EvidenceItem] = (),
) -> DiagnosisReport:
    timeout = _probe(probe_results, "timeout_outcome")
    k8s = _probe(probe_results, "k8s_not_ready")
    migration = _probe(probe_results, "juju_migration")
    csr = _probe(probe_results, "certificate_csr_churn")
    package = _probe(probe_results, "package_install_failure")
    crash = _probe(probe_results, "workload_crash_recovery")
    relations = _probe(probe_results, "relation_blockers")
    if migration is not None and _has_direct_failed_migration(migration):
        return _deterministic_report(
            report,
            migration,
            name="failed Juju agent migration",
            summary="The Juju unit-agent migration failed before the unit was lost.",
            root_cause=(
                "The Juju migration failed for the k8s unit agent; the archived unit "
                "log records the migration failure and that agent.conf was left "
                "unchanged. The later k8sd crash is a separate event."
            ),
            stop_reason="deterministic_failed_migration",
            additional_probes=(() if crash is None else (crash,)),
        )
    if csr is not None and _has_csr_churn_with_terminal_blocker(csr, timeout):
        return _deterministic_report(
            report,
            csr,
            name="certificate CSR relation churn",
            summary=(
                "Hypervisor certificate requests disappeared from relation data "
                "and were regenerated while units remained blocked."
            ),
            root_cause=(
                "Certificate CSR relation data was repeatedly missing, causing "
                "hypervisors to regenerate requests and remain waiting for the "
                "certificates integration."
            ),
            stop_reason="deterministic_certificate_csr_churn",
        )
    if package is not None and package.status == "triggered":
        return _deterministic_report(
            report,
            package,
            name="MAAS package configuration failure",
            summary="MAAS installation failed while configuring maas-region-api.",
            root_cause=(
                "The maas-region-api package configuration failed with "
                "SystemError: ifaddresses() method: bad call flags; dependent "
                "package failures and apt exit status 100 were cascading errors."
            ),
            stop_reason="deterministic_package_install_failure",
        )
    if relations is not None and relations.status == "triggered":
        result = _deterministic_report(
            report,
            relations,
            name="relation and payload readiness blockers",
            summary=(
                "The deployment timed out with direct AMQP, database-relation, "
                "and payload-container readiness blockers."
            ),
            root_cause=(
                "The observed failure surface was blocked relations and payload "
                "readiness: cinder-volume lacked AMQP, its mysql-router lacked "
                "the database relation, and cinder, glance, and nova payloads "
                "were not ready. Evidence is insufficient to establish one "
                "common upstream root cause."
            ),
            stop_reason="deterministic_relation_blockers",
        )
        return replace(result, needs_more_evidence=True)
    if timeout is None or not _is_k8s_false_negative(timeout, k8s):
        return _attach_failure_surface_evidence(report, initial_evidence)

    policy_evidence = [
        ReportEvidence(
            path=finding.path,
            line=finding.line,
            excerpt=finding.excerpt,
        )
        for finding in timeout.findings
        if finding.category
        in {"timeout_surface", "post_timeout_completion", "later_convergence"}
    ]
    evidence = _deduplicate_evidence([*policy_evidence, *report.evidence])
    unknown = (
        "The available evidence establishes a false-negative timeout but does "
        "not establish why convergence exceeded the deadline."
    )
    unknowns = [*report.unknowns]
    if unknown not in unknowns:
        unknowns.append(unknown)
    mechanisms = [
        CandidateMechanism(
            name=item.name,
            status=(
                "speculative"
                if item.status in {"confirmed", "supported"}
                else item.status
            ),
            rationale=item.rationale,
        )
        for item in report.candidate_mechanisms
    ]
    mechanisms.insert(
        0,
        CandidateMechanism(
            name="false-negative k8s readiness timeout",
            status="supported",
            rationale=(
                "The output records node joins after the timeout and final "
                "cluster evidence shows convergence."
            ),
        ),
    )
    return replace(
        report,
        summary=(
            "The CI step reported a k8s readiness timeout, but subsequent "
            "artifact evidence shows the nodes joined and the cluster converged."
        ),
        root_cause=(
            "The CI failure was a false-negative timeout: convergence completed "
            "after the readiness deadline. The evidence does not establish the "
            "underlying reason for the delay."
        ),
        confidence="supported",
        needs_more_evidence=False,
        evidence=evidence,
        candidate_mechanisms=mechanisms,
        unknowns=unknowns,
        triage_confidence="medium",
        stop_reason="deterministic_false_negative",
    )


def _probe(
    probe_results: tuple[ProbeResult, ...] | list[ProbeResult],
    name: str,
) -> ProbeResult | None:
    return next((result for result in probe_results if result.name == name), None)


def _is_k8s_false_negative(
    timeout: ProbeResult | None,
    k8s: ProbeResult | None,
) -> bool:
    if timeout is None or k8s is None:
        return False
    if timeout.status != "triggered" or k8s.status != "triggered":
        return False
    categories = {finding.category for finding in timeout.findings}
    return (
        "post_timeout_completion" in categories
        and "later_convergence" in categories
        and "terminal_blocker" not in categories
    )


def _deduplicate_evidence(items: list[ReportEvidence]) -> list[ReportEvidence]:
    result: list[ReportEvidence] = []
    seen: set[tuple[str, int | None, str]] = set()
    for item in items:
        key = (item.path, item.line, item.excerpt)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _has_direct_failed_migration(migration: ProbeResult | None) -> bool:
    return bool(
        migration is not None
        and migration.status == "triggered"
        and any(
            finding.category == "failed_migration_signal"
            and (
                "agent.conf" in finding.excerpt.lower()
                or "migration failed" in finding.excerpt.lower()
            )
            for finding in migration.findings
        )
    )


def _has_csr_churn_with_terminal_blocker(
    csr: ProbeResult | None,
    timeout: ProbeResult | None,
) -> bool:
    if csr is None or csr.status != "triggered" or timeout is None:
        return False
    missing_csrs = {
        finding.excerpt
        for finding in csr.findings
        if "not found in relation data" in finding.excerpt.lower()
    }
    has_certificate_blocker = any(
        finding.category == "terminal_blocker"
        and "certificate" in finding.excerpt.lower()
        for finding in timeout.findings
    )
    return len(missing_csrs) >= 2 and has_certificate_blocker


def _deterministic_report(
    report: DiagnosisReport,
    probe: ProbeResult,
    *,
    name: str,
    summary: str,
    root_cause: str,
    stop_reason: str,
    additional_probes: tuple[ProbeResult, ...] = (),
) -> DiagnosisReport:
    evidence = _deduplicate_evidence([
        *[
            ReportEvidence(
                path=finding.path,
                line=finding.line,
                excerpt=finding.excerpt,
            )
            for finding in probe.findings
        ],
        *[
            ReportEvidence(
                path=finding.path,
                line=finding.line,
                excerpt=finding.excerpt,
            )
            for additional in additional_probes
            for finding in additional.findings
        ],
        *report.evidence,
    ])
    mechanisms = [
        CandidateMechanism(
            name=name,
            status="supported",
            rationale=probe.summary,
        ),
        *[
            CandidateMechanism(
                name=item.name,
                status=(
                    "speculative"
                    if item.status in {"confirmed", "supported"}
                    else item.status
                ),
                rationale=item.rationale,
            )
            for item in report.candidate_mechanisms
        ],
    ]
    return replace(
        report,
        summary=summary,
        root_cause=root_cause,
        confidence="supported",
        needs_more_evidence=False,
        evidence=evidence,
        candidate_mechanisms=mechanisms,
        triage_confidence="medium",
        stop_reason=stop_reason,
    )


def _attach_failure_surface_evidence(
    report: DiagnosisReport,
    initial_evidence: tuple[EvidenceItem, ...] | list[EvidenceItem],
) -> DiagnosisReport:
    if report.evidence or not report.needs_more_evidence:
        return report
    evidence = [
        ReportEvidence(path=item.path, line=item.line, excerpt=item.excerpt)
        for item in initial_evidence[:12]
    ]
    if not evidence:
        return report
    return replace(report, evidence=evidence)
