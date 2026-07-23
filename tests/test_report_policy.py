from sunbeam_triage.core.evidence import EvidenceItem
from sunbeam_triage.core.llm import CandidateMechanism, DiagnosisReport
from sunbeam_triage.core.probes import ProbeFinding, ProbeResult
from sunbeam_triage.core.report_policy import apply_probe_report_policies


def test_machine_add_policy_confirms_apt_delay_but_not_proxy_root_cause():
    report = DiagnosisReport(
        summary="A Juju unit was lost.",
        failure_surface="cluster create failed",
        confidence="supported",
        root_cause="The orchestrator unit was lost.",
        candidate_mechanisms=[
            CandidateMechanism(
                name="lost orchestrator unit",
                status="supported",
                rationale="The final status contained a lost unit.",
            )
        ],
    )
    probe = ProbeResult(
        name="machine_add_timeout",
        status="triggered",
        summary="APT crossed the machine-add deadline.",
        findings=[
            ProbeFinding(
                "machine_add_timeout",
                "generated/sunbeam/output.log",
                10,
                "host03 timed out after 306s.",
            ),
            ProbeFinding(
                "remote_add_machine_failed",
                "sosreport-host03:join.log",
                20,
                "host03: Add machine failed.",
            ),
            ProbeFinding(
                "apt_process_running",
                "sosreport-host03:ps_auxfwww",
                4,
                "host03: apt-get update was still running.",
            ),
            ProbeFinding(
                "apt_proxy_path",
                "sosreport-host03:90curtin-aptproxy",
                1,
                "host03: Acquire::http::Proxy http://10.239.8.11:8000.",
            ),
            ProbeFinding(
                "successful_join_control",
                "sosreport-host05:join.log",
                20,
                "Successful-node control: host05 completed Add machine.",
            ),
        ],
    )

    result = apply_probe_report_policies(report, (probe,))

    assert result.confidence == "confirmed"
    assert result.stop_reason == "deterministic_machine_add_apt_delay"
    assert "slow or stalled APT" in result.root_cause
    assert result.candidate_mechanisms[0].status == "confirmed"
    assert result.candidate_mechanisms[1].status == "speculative"
    assert result.causal_assessment is not None
    assert result.causal_assessment.root_cause.evidence_ids
    assert "proxy" in result.unknowns[-1]
    assert "upstream repositories" in result.unknowns[-1]


def test_machine_add_policy_requires_apt_to_cross_the_deadline():
    report = DiagnosisReport(
        summary="machine add failed",
        failure_surface="cluster create failed",
        confidence="unknown",
        root_cause="",
    )
    probe = ProbeResult(
        name="machine_add_timeout",
        status="triggered",
        summary="machine add timeout",
        findings=[
            ProbeFinding("machine_add_timeout", "output.log", 10, "timed out"),
            ProbeFinding(
                "remote_add_machine_failed", "join.log", 20, "Add machine failed"
            ),
            ProbeFinding(
                "apt_proxy_path",
                "apt.conf",
                1,
                "Acquire::http::Proxy http://10.239.8.11:8000",
            ),
        ],
    )

    assert apply_probe_report_policies(report, (probe,)) is report


def test_false_negative_policy_separates_outcome_from_delay_mechanism():
    report = DiagnosisReport(
        summary="Migration caused the timeout.",
        failure_surface="k8s readiness timed out.",
        confidence="supported",
        root_cause="Juju migration caused delayed readiness.",
        candidate_mechanisms=[
            CandidateMechanism(
                name="migration",
                status="supported",
                rationale="Migration happened before the timeout.",
            )
        ],
    )
    probes = (
        ProbeResult(
            name="timeout_outcome",
            status="triggered",
            summary="possible false negative",
            findings=[
                ProbeFinding("timeout_surface", "output.log", 10, "k8s timed out"),
                ProbeFinding(
                    "post_timeout_completion",
                    "output.log",
                    20,
                    "Node joined cluster",
                ),
                ProbeFinding(
                    "later_convergence",
                    "juju_status.txt",
                    2,
                    "k8s active Ready",
                ),
            ],
        ),
        ProbeResult(
            name="k8s_not_ready",
            status="triggered",
            summary="k8s timeout",
        ),
    )

    result = apply_probe_report_policies(report, probes)

    assert result.root_cause == "Juju migration caused delayed readiness."
    assert result.causal_assessment is not None
    assert "deadline expired" in result.causal_assessment.failure_trigger.claim
    assert "after the deadline" in result.causal_assessment.post_failure_outcome.claim
    assert result.causal_assessment.root_cause.claim == result.root_cause
    assert result.candidate_mechanisms[0].status == "supported"
    assert any(item.excerpt == "Node joined cluster" for item in result.evidence)


def test_false_negative_policy_requires_k8s_timeout_probe():
    report = DiagnosisReport(
        summary="A generic operation timed out.",
        failure_surface="operation timeout",
        confidence="unknown",
        root_cause="",
    )
    timeout = ProbeResult(
        name="timeout_outcome",
        status="triggered",
        summary="possible false negative",
        findings=[
            ProbeFinding(
                "post_timeout_completion", "output.log", 20, "Node joined cluster"
            )
        ],
    )

    assert apply_probe_report_policies(report, (timeout,)) is report


def test_false_negative_policy_does_not_hide_terminal_blocker():
    report = DiagnosisReport(
        summary="Certificates did not converge.",
        failure_surface="hypervisors remained waiting",
        confidence="supported",
        root_cause="certificate requests were not signed",
    )
    probes = (
        ProbeResult(
            name="timeout_outcome",
            status="triggered",
            summary="mixed outcome",
            findings=[
                ProbeFinding("timeout_surface", "output.log", 10, "k8s timed out"),
                ProbeFinding(
                    "post_timeout_completion",
                    "output.log",
                    20,
                    "Node joined cluster",
                ),
                ProbeFinding("later_convergence", "status.txt", 2, "k8s active Ready"),
                ProbeFinding(
                    "terminal_blocker",
                    "status.txt",
                    5,
                    "openstack-hypervisor/1 waiting certificates incomplete",
                ),
            ],
        ),
        ProbeResult(
            name="k8s_not_ready",
            status="triggered",
            summary="k8s timeout",
        ),
    )

    assert apply_probe_report_policies(report, probes) is report


def test_failed_migration_policy_uses_direct_agent_conf_evidence():
    report = DiagnosisReport(
        summary="k8sd crashed",
        failure_surface="k8s unit lost",
        confidence="unknown",
        root_cause="",
    )
    migration = ProbeResult(
        name="juju_migration",
        status="triggered",
        summary="direct migration failure",
        findings=[
            ProbeFinding(
                "failed_migration_signal",
                "sosreport.tar:var/log/juju/unit-k8s-0.log",
                42,
                "migration REAP failed; agent.conf left unchanged",
            )
        ],
    )

    result = apply_probe_report_policies(report, (migration,))

    assert "Juju migration failed" in result.root_cause
    assert "k8sd crash is a separate event" in result.root_cause
    assert result.stop_reason == "deterministic_failed_migration"


def test_package_failure_policy_identifies_decisive_maas_error():
    report = DiagnosisReport(
        summary="apt failed",
        failure_surface="MAAS install exited 100",
        confidence="unknown",
        root_cause="",
    )
    package = ProbeResult(
        name="package_install_failure",
        status="triggered",
        summary="MAAS package configuration failed in ifaddresses().",
        findings=[
            ProbeFinding(
                "package_install_failure",
                "generated/maas/log.txt",
                1057,
                "SystemError: ifaddresses() method: bad call flags",
            )
        ],
    )

    result = apply_probe_report_policies(report, (package,))

    assert "maas-region-api" in result.root_cause
    assert "cascading errors" in result.root_cause
    assert result.stop_reason == "deterministic_package_install_failure"


def test_csr_churn_policy_requires_multiple_missing_requests_and_blocker():
    report = DiagnosisReport(
        summary="certificate wait",
        failure_surface="hypervisors waiting",
        confidence="unknown",
        root_cause="",
    )
    csr = ProbeResult(
        name="certificate_csr_churn",
        status="triggered",
        summary="CSRs disappeared",
        findings=[
            ProbeFinding(
                "csr_churn",
                "juju-debug.log",
                100,
                "CSR for 'openstack-hypervisor-0' not found in relation data",
            ),
            ProbeFinding(
                "csr_churn",
                "juju-debug.log",
                200,
                "CSR for 'openstack-hypervisor-3' not found in relation data",
            ),
        ],
    )
    timeout = ProbeResult(
        name="timeout_outcome",
        status="triggered",
        summary="timeout",
        findings=[
            ProbeFinding(
                "terminal_blocker",
                "status.txt",
                5,
                "openstack-hypervisor/1 waiting certificates integration incomplete",
            )
        ],
    )

    result = apply_probe_report_policies(report, (csr, timeout))

    assert "CSR relation data was repeatedly missing" in result.root_cause
    assert result.stop_reason == "deterministic_certificate_csr_churn"


def test_insufficient_report_retains_initial_failure_surface_evidence():
    report = DiagnosisReport(
        summary="The cause is unknown.",
        failure_surface="command failed",
        confidence="unknown",
        root_cause="",
        needs_more_evidence=True,
    )
    initial = (
        EvidenceItem(
            kind="runner",
            path="generated/github-runner/run.log",
            line=50,
            excerpt="Process completed with exit code 1",
        ),
    )

    result = apply_probe_report_policies(report, (), initial)

    assert result.confidence == "unknown"
    assert result.needs_more_evidence is True
    assert result.evidence[0].path == "generated/github-runner/run.log"


def test_relation_blocker_policy_reports_surface_without_common_cause():
    report = DiagnosisReport(
        summary="deployment timeout",
        failure_surface="applications waiting",
        confidence="unknown",
        root_cause="",
    )
    relations = ProbeResult(
        name="relation_blockers",
        status="triggered",
        summary="direct blockers",
        findings=[
            ProbeFinding(
                "relation_blocker",
                "machines-status.txt",
                5,
                "cinder-volume blocked (amqp) integration missing",
            ),
            ProbeFinding(
                "relation_blocker",
                "status.txt",
                12,
                "cinder-volume-mysql-router blocked Missing relation: database",
            ),
            ProbeFinding(
                "relation_blocker",
                "status.txt",
                13,
                "glance waiting Payload container not ready",
            ),
        ],
    )

    result = apply_probe_report_policies(report, (relations,))

    assert result.root_cause == "The common upstream cause is not established."
    assert result.causal_assessment.root_cause.confidence == "unknown"
    assert result.causal_assessment.contributing_factors
    assert result.needs_more_evidence is True
    assert result.stop_reason == "deterministic_relation_blockers"
