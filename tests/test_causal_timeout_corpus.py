import json
from pathlib import Path

import pytest

from sunbeam_triage.core.llm import (
    CausalAssessment,
    CausalClaim,
    DiagnosisReport,
    ReportEvidence,
)
from sunbeam_triage.core.probes import ProbeFinding, ProbeResult
from sunbeam_triage.core.report_policy import apply_probe_report_policies
from sunbeam_triage.core.report_validation import validate_causal_report

CASES = json.loads(
    Path("tests/fixtures/causal_timeout_cases.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["uuid"][:8])
def test_timeout_corpus_preserves_unknown_cause_and_specific_contributors(case):
    trigger = ReportEvidence(
        path="generated/sunbeam/output.log",
        line=100,
        excerpt="Application 'k8s' is not ready: wait timed out after 1799s",
        role="failure_trigger",
    )
    signal = ReportEvidence(
        path="generated/sunbeam/juju_debug_log_openstack-machines.txt",
        line=80,
        excerpt=case["contributor"] or "No decisive node-side mechanism was captured",
        role="contributing_factor" if case["contributor"] else "observation",
    )
    factors = (
        [
            CausalClaim(
                claim=case["contributor"],
                confidence="supported",
                evidence_ids=[signal.id],
                missing_evidence=["The dependency link to the timeout is incomplete."],
            )
        ]
        if case["contributor"]
        else []
    )
    report = DiagnosisReport(
        summary="Readiness timed out.",
        failure_surface="The readiness deadline expired.",
        confidence="unknown",
        root_cause="The underlying cause is not established.",
        needs_more_evidence=True,
        evidence=[trigger, signal],
        causal_assessment=CausalAssessment(
            failure_trigger=CausalClaim(
                claim="The readiness deadline expired.",
                confidence="confirmed",
                evidence_ids=[trigger.id],
            ),
            contributing_factors=factors,
            root_cause=CausalClaim(
                claim="The underlying cause is not established.",
                confidence="unknown",
            ),
        ),
    )
    probes = (
        ProbeResult(
            name="timeout_outcome",
            status="triggered",
            summary="late convergence",
            findings=[
                ProbeFinding("timeout_surface", trigger.path, trigger.line, trigger.excerpt),
                ProbeFinding(
                    "post_timeout_completion",
                    "generated/sunbeam/output.log",
                    200,
                    "Node joined cluster after the timeout",
                ),
                ProbeFinding(
                    "later_convergence",
                    "generated/sunbeam/kubectl_get_node.txt",
                    2,
                    "node Ready",
                ),
            ],
        ),
        ProbeResult(name="k8s_not_ready", status="triggered", summary="timeout"),
    )

    result = validate_causal_report(
        apply_probe_report_policies(report, probes), probes
    )

    assert result.causal_assessment is not None
    assert result.causal_assessment.failure_trigger.confidence == "confirmed"
    assert result.causal_assessment.post_failure_outcome.confidence == "confirmed"
    assert result.causal_assessment.root_cause.confidence == "unknown"
    assert result.needs_more_evidence is True
    assert "cause" in result.stop_reason
    assert [item.claim for item in result.causal_assessment.contributing_factors] == (
        [case["contributor"]] if case["contributor"] else []
    )


def test_eventual_convergence_cannot_be_root_cause():
    outcome = ReportEvidence(
        path="generated/sunbeam/kubectl_get_node.txt",
        line=2,
        excerpt="node Ready",
        role="post_failure_outcome",
    )
    report = DiagnosisReport(
        summary="The cluster later converged.",
        failure_surface="Readiness timed out.",
        confidence="supported",
        root_cause="The timeout was a false negative because the cluster converged.",
        evidence=[outcome],
        causal_assessment=CausalAssessment(
            failure_trigger=CausalClaim("Readiness timed out.", "unknown"),
            root_cause=CausalClaim(
                "The timeout was a false negative because the cluster converged.",
                "supported",
                evidence_ids=[outcome.id],
            ),
            post_failure_outcome=CausalClaim(
                "The cluster later converged.",
                "confirmed",
                evidence_ids=[outcome.id],
            ),
        ),
    )

    result = validate_causal_report(report)

    assert result.causal_assessment is not None
    assert result.causal_assessment.root_cause.confidence == "unknown"
    assert result.needs_more_evidence is True
