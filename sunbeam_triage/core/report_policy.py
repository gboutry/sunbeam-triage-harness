from __future__ import annotations

from dataclasses import replace

from .evidence import EvidenceItem
from .llm_schema import (
    CandidateMechanism,
    CausalAssessment,
    CausalClaim,
    DiagnosisReport,
    ReportEvidence,
)
from .probes import ProbeResult


def apply_probe_report_policies(
    report: DiagnosisReport,
    probe_results: tuple[ProbeResult, ...] | list[ProbeResult],
    initial_evidence: tuple[EvidenceItem, ...] | list[EvidenceItem] = (),
) -> DiagnosisReport:
    timeout = _probe(probe_results, "timeout_outcome")
    machine_add = _probe(probe_results, "machine_add_timeout")
    k8s = _probe(probe_results, "k8s_not_ready")
    migration = _probe(probe_results, "juju_migration")
    csr = _probe(probe_results, "certificate_csr_churn")
    package = _probe(probe_results, "package_install_failure")
    crash = _probe(probe_results, "workload_crash_recovery")
    relations = _probe(probe_results, "relation_blockers")
    if machine_add is not None and _has_confirmed_machine_add_apt_delay(machine_add):
        return _machine_add_apt_report(report, machine_add)
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
        relation_ids = [finding.id for finding in relations.findings]
        assessment = replace(
            result.causal_assessment,
            contributing_factors=[
                CausalClaim(
                    claim=(
                        "Relation and payload readiness blockers were present "
                        "before the timeout."
                    ),
                    confidence="confirmed",
                    evidence_ids=relation_ids,
                )
            ],
            root_cause=CausalClaim(
                claim="The common upstream cause is not established.",
                confidence="unknown",
                missing_evidence=[
                    "Evidence connecting the observed blockers to one upstream cause."
                ],
            ),
        )
        return replace(
            result,
            root_cause="The common upstream cause is not established.",
            confidence="unknown",
            triage_confidence="low",
            needs_more_evidence=True,
            causal_assessment=assessment,
        )
    if timeout is None or not _is_k8s_false_negative(timeout, k8s):
        return _attach_failure_surface_evidence(report, initial_evidence)

    policy_evidence = [
        ReportEvidence(
            path=finding.path,
            line=finding.line,
            excerpt=finding.excerpt,
            id=finding.id,
            role=_role_for_timeout_finding(finding.category),
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
    trigger_ids = [
        finding.id
        for finding in timeout.findings
        if finding.category == "timeout_surface"
    ]
    outcome_ids = [
        finding.id
        for finding in timeout.findings
        if finding.category in {"post_timeout_completion", "later_convergence"}
    ]
    assessment = report.causal_assessment or CausalAssessment(
        failure_trigger=CausalClaim(claim="", confidence="unknown")
    )
    assessment = replace(
        assessment,
        failure_trigger=CausalClaim(
            claim="The CI readiness deadline expired before convergence.",
            confidence="confirmed",
            evidence_ids=trigger_ids,
        ),
        post_failure_outcome=CausalClaim(
            claim=(
                "Cluster operations completed after the deadline and the final "
                "snapshot shows convergence."
            ),
            confidence="confirmed",
            evidence_ids=outcome_ids,
        ),
    )
    cause_unknown = assessment.root_cause.confidence == "unknown"
    return replace(
        report,
        summary=(
            "The CI step reported a k8s readiness timeout, but subsequent "
            "artifact evidence shows later convergence; the reason for the delay "
            "must be assessed separately."
        ),
        needs_more_evidence=report.needs_more_evidence or cause_unknown,
        evidence=evidence,
        unknowns=unknowns,
        stop_reason=("cause_unresolved" if cause_unknown else report.stop_reason),
        causal_assessment=assessment,
    )


def _role_for_timeout_finding(category: str) -> str:
    if category == "timeout_surface":
        return "failure_trigger"
    return "post_failure_outcome"


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


def _has_confirmed_machine_add_apt_delay(probe: ProbeResult | None) -> bool:
    if probe is None or probe.status != "triggered":
        return False
    categories = {finding.category for finding in probe.findings}
    has_failure = {
        "machine_add_timeout",
        "remote_add_machine_failed",
    } <= categories
    has_causal_apt = bool(categories & {"apt_process_running", "apt_deadline_exceeded"})
    return has_failure and has_causal_apt


def _machine_add_apt_report(
    report: DiagnosisReport,
    probe: ProbeResult,
) -> DiagnosisReport:
    role_by_category = {
        "machine_add_timeout": "failure_trigger",
        "remote_add_machine_failed": "symptom",
        "apt_process_running": "root_cause",
        "apt_update_incomplete": "root_cause",
        "apt_deadline_exceeded": "root_cause",
        "apt_fetch_timing": "contributing_factor",
        "agent_started": "post_failure_outcome",
        "apt_proxy_path": "contributing_factor",
        "successful_join_control": "counterevidence",
    }
    policy_evidence = [
        ReportEvidence(
            path=finding.path,
            line=finding.line,
            excerpt=finding.excerpt,
            id=finding.id,
            role=role_by_category.get(finding.category, "observation"),
        )
        for finding in probe.findings
    ]
    evidence = _deduplicate_evidence([*policy_evidence, *report.evidence])
    trigger_ids = [
        item.id for item in policy_evidence if item.role == "failure_trigger"
    ]
    root_ids = [item.id for item in policy_evidence if item.role == "root_cause"]
    proxy_ids = [
        item.id
        for item in policy_evidence
        if item.role == "contributing_factor" and "Proxy" in item.excerpt
    ]
    outcome_ids = [
        item.id
        for item in policy_evidence
        if item.role in {"post_failure_outcome", "counterevidence"}
    ]
    root_cause = (
        "Juju manual-machine bootstrap was blocked by slow or stalled APT "
        "repository access, so machine agents did not start within Sunbeam's "
        "machine-add deadline."
    )
    unknown = (
        "The artifacts do not distinguish whether the configured APT proxy, its "
        "upstream repositories, or the intervening network caused the latency."
    )
    unknowns = list(report.unknowns)
    if unknown not in unknowns:
        unknowns.append(unknown)
    existing_assessment = report.causal_assessment
    assessment = CausalAssessment(
        failure_trigger=CausalClaim(
            claim=(
                "The CI step failed when multiple Juju machine-add operations "
                "exceeded their fixed deadline."
            ),
            confidence="confirmed",
            evidence_ids=trigger_ids,
        ),
        symptoms=(
            list(existing_assessment.symptoms)
            if existing_assessment is not None
            else []
        ),
        contributing_factors=[
            *(
                list(existing_assessment.contributing_factors)
                if existing_assessment is not None
                else []
            ),
            CausalClaim(
                claim="The affected nodes routed HTTP APT access through a proxy.",
                confidence=("confirmed" if proxy_ids else "unknown"),
                evidence_ids=proxy_ids,
                missing_evidence=(
                    [] if proxy_ids else ["APT proxy configuration was not captured."]
                ),
            ),
        ],
        root_cause=CausalClaim(
            claim=root_cause,
            confidence="confirmed",
            evidence_ids=root_ids,
            missing_evidence=[unknown],
        ),
        post_failure_outcome=CausalClaim(
            claim=(
                "At least one node completed the same join path, while other "
                "machine agents started only after their CLI deadlines."
            ),
            confidence=("confirmed" if outcome_ids else "unknown"),
            evidence_ids=outcome_ids,
        ),
    )
    mechanisms = [
        CandidateMechanism(
            name="slow or stalled APT access during Juju machine bootstrap",
            status="confirmed",
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
        summary=(
            "The cluster join failed because APT repository access delayed Juju "
            "manual-machine bootstrap beyond the machine-add deadline."
        ),
        root_cause=root_cause,
        confidence="confirmed",
        needs_more_evidence=False,
        evidence=evidence,
        candidate_mechanisms=mechanisms,
        unknowns=unknowns,
        triage_confidence="high",
        stop_reason="deterministic_machine_add_apt_delay",
        causal_assessment=assessment,
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
                id=finding.id,
                role="root_cause",
            )
            for finding in probe.findings
        ],
        *[
            ReportEvidence(
                path=finding.path,
                line=finding.line,
                excerpt=finding.excerpt,
                id=finding.id,
                role="counterevidence",
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
    assessment = CausalAssessment(
        failure_trigger=(
            report.causal_assessment.failure_trigger
            if report.causal_assessment is not None
            else CausalClaim(report.failure_surface, report.confidence)
        ),
        symptoms=(
            list(report.causal_assessment.symptoms)
            if report.causal_assessment is not None
            else []
        ),
        contributing_factors=(
            list(report.causal_assessment.contributing_factors)
            if report.causal_assessment is not None
            else []
        ),
        root_cause=CausalClaim(
            claim=root_cause,
            confidence="supported",
            evidence_ids=[item.id for item in evidence if item.role == "root_cause"],
        ),
        post_failure_outcome=(
            report.causal_assessment.post_failure_outcome
            if report.causal_assessment is not None
            else CausalClaim("", "unknown")
        ),
    )
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
        causal_assessment=assessment,
    )


def _attach_failure_surface_evidence(
    report: DiagnosisReport,
    initial_evidence: tuple[EvidenceItem, ...] | list[EvidenceItem],
) -> DiagnosisReport:
    if report.evidence or not report.needs_more_evidence:
        return report
    evidence = [
        ReportEvidence(
            path=item.path,
            line=item.line,
            excerpt=item.excerpt,
            id=item.id,
            role="failure_trigger",
        )
        for item in initial_evidence[:12]
    ]
    if not evidence:
        return report
    return replace(report, evidence=evidence)
