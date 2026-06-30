from pathlib import Path

from sunbeam_triage.core.evidence import EvidenceCollector
from sunbeam_triage.core.llm import (
    AlternativeConsidered,
    DiagnosisReport,
    ReportEvidence,
    TimelineEvent,
)
from sunbeam_triage.core.render import render_html


def test_render_html_escapes_content_and_includes_sections():
    pack = EvidenceCollector(Path("tests/fixtures/sample_uuid"), "sample-uuid").collect()
    report = DiagnosisReport(
        summary="Timed out <bad>",
        failure_surface="cluster resize timeout",
        confidence="supported",
        root_cause="Readiness did not converge.",
        evidence=[
            ReportEvidence(
                path="generated/sunbeam/output.log",
                line=2,
                excerpt="Error: wait timed out <script>",
            )
        ],
        candidate_mechanisms=[],
        recommendations=["Inspect the readiness gate."],
        unknowns=["No sosreport was inspected."],
    )

    html = render_html(pack, report)

    assert "<h1>Diagnostics: sample-uuid</h1>" in html
    assert "Timed out &lt;bad&gt;" in html
    assert "&lt;script&gt;" in html
    assert "sunbeam_deploy" in html
    assert "Inspect the readiness gate." in html


def test_render_html_includes_triage_v2_sections():
    pack = EvidenceCollector(Path("tests/fixtures/sample_uuid"), "sample-uuid").collect()
    report = DiagnosisReport(
        summary="Timed out",
        failure_surface="Deploy timeout",
        confidence="supported",
        root_cause="RabbitMQ closed the connection first.",
        triage_confidence="medium",
        stop_reason="sufficient_evidence",
        failure_timeline=[
            TimelineEvent(
                timestamp="10:42:29",
                source="rabbitmq.log",
                location="line 120",
                event="RabbitMQ closed AMQP connection.",
            )
        ],
        cascading_errors=[
            ReportEvidence(
                path="nova-api.log",
                line=1242,
                excerpt="oslo.messaging timeout",
            )
        ],
        alternatives_considered=[
            AlternativeConsidered(
                hypothesis="Database outage",
                status="less_likely",
                reason="No DB errors near the first failure timestamp.",
            )
        ],
        missing_evidence=["Need neutron-server timing."],
    )

    html = render_html(pack, report)

    assert "Triage Confidence" in html
    assert "medium" in html
    assert "Failure Timeline" in html
    assert "RabbitMQ closed AMQP connection." in html
    assert "Cascading Errors" in html
    assert "oslo.messaging timeout" in html
    assert "Alternatives Considered" in html
    assert "Database outage" in html
    assert "Missing Evidence" in html
    assert "Need neutron-server timing." in html
