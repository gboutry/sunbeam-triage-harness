from pathlib import Path

from sunbeam_triage.evidence import EvidenceCollector
from sunbeam_triage.llm import DiagnosisReport, ReportEvidence
from sunbeam_triage.render import render_html


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
