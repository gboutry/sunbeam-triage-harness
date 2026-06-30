from __future__ import annotations

from html import escape

from .evidence import EvidencePack
from .llm import DiagnosisReport


def render_html(pack: EvidencePack, report: DiagnosisReport) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>Diagnostics: {escape(pack.uuid)}</title>",
            "<style>",
            CSS,
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Diagnostics: {escape(pack.uuid)}</h1>",
            _summary(pack, report),
            _section("Failure Surface", f"<p>{escape(report.failure_surface)}</p>"),
            _section("Root Cause", f"<p>{escape(report.root_cause)}</p>"),
            _report_evidence(report),
            _failure_timeline(report),
            _cascading_errors(report),
            _candidate_mechanisms(report),
            _alternatives_considered(report),
            _list_section("Recommendations", report.recommendations),
            _list_section("Unknowns", report.unknowns),
            _list_section("Missing Evidence", report.missing_evidence),
            _harness_evidence(pack),
            "</body>",
            "</html>",
        ]
    )


def _summary(pack: EvidencePack, report: DiagnosisReport) -> str:
    return f"""
<section class="summary">
  <h2>Summary</h2>
  <p>{escape(report.summary)}</p>
  <dl>
    <dt>Run ID</dt><dd>{escape(str(pack.run.run_id))}</dd>
    <dt>Branch</dt><dd>{escape(pack.run.branch)}</dd>
    <dt>Workflow</dt><dd>{escape(pack.run.workflow)}</dd>
    <dt>Failed Step</dt><dd>{escape(pack.failed_step.name)}</dd>
    <dt>Confidence</dt><dd>{escape(report.confidence)}</dd>
    <dt>Triage Confidence</dt><dd>{escape(report.triage_confidence)}</dd>
    <dt>Stop Reason</dt><dd>{escape(report.stop_reason or "not reported")}</dd>
  </dl>
</section>
""".strip()


def _section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _report_evidence(report: DiagnosisReport) -> str:
    rows = []
    for item in report.evidence:
        line = "" if item.line is None else f":{item.line}"
        rows.append(
            "<tr>"
            f"<td>{escape(item.path + line)}</td>"
            f"<td><code>{escape(item.excerpt)}</code></td>"
            "</tr>"
        )
    body = "<p>No evidence items returned by model.</p>"
    if rows:
        body = "<table><thead><tr><th>Source</th><th>Excerpt</th></tr></thead><tbody>"
        body += "\n".join(rows)
        body += "</tbody></table>"
    return _section("Evidence Used By Diagnosis", body)


def _candidate_mechanisms(report: DiagnosisReport) -> str:
    if not report.candidate_mechanisms:
        return _section("Candidate Mechanisms", "<p>No candidate mechanisms returned.</p>")
    rows = [
        "<tr>"
        f"<td>{escape(item.name)}</td>"
        f"<td>{escape(item.status)}</td>"
        f"<td>{escape(item.rationale)}</td>"
        "</tr>"
        for item in report.candidate_mechanisms
    ]
    return _section(
        "Candidate Mechanisms",
        "<table><thead><tr><th>Name</th><th>Status</th><th>Rationale</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
    )


def _failure_timeline(report: DiagnosisReport) -> str:
    if not report.failure_timeline:
        return _section("Failure Timeline", "<p>No timeline returned.</p>")
    rows = [
        "<tr>"
        f"<td>{escape(item.timestamp)}</td>"
        f"<td>{escape(item.source)}</td>"
        f"<td>{escape(item.location)}</td>"
        f"<td>{escape(item.event)}</td>"
        "</tr>"
        for item in report.failure_timeline
    ]
    return _section(
        "Failure Timeline",
        "<table><thead><tr><th>Timestamp</th><th>Source</th>"
        "<th>Location</th><th>Event</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
    )


def _cascading_errors(report: DiagnosisReport) -> str:
    if not report.cascading_errors:
        return _section("Cascading Errors", "<p>No cascading errors returned.</p>")
    rows = []
    for item in report.cascading_errors:
        line = "" if item.line is None else f":{item.line}"
        rows.append(
            "<tr>"
            f"<td>{escape(item.path + line)}</td>"
            f"<td><code>{escape(item.excerpt)}</code></td>"
            "</tr>"
        )
    return _section(
        "Cascading Errors",
        "<table><thead><tr><th>Source</th><th>Excerpt</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
    )


def _alternatives_considered(report: DiagnosisReport) -> str:
    if not report.alternatives_considered:
        return _section("Alternatives Considered", "<p>No alternatives returned.</p>")
    rows = [
        "<tr>"
        f"<td>{escape(item.hypothesis)}</td>"
        f"<td>{escape(item.status)}</td>"
        f"<td>{escape(item.reason)}</td>"
        "</tr>"
        for item in report.alternatives_considered
    ]
    return _section(
        "Alternatives Considered",
        "<table><thead><tr><th>Hypothesis</th><th>Status</th><th>Reason</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table>",
    )


def _list_section(title: str, values: list[str]) -> str:
    if not values:
        return _section(title, "<p>None reported.</p>")
    return _section(
        title,
        "<ul>" + "".join(f"<li>{escape(value)}</li>" for value in values) + "</ul>",
    )


def _harness_evidence(pack: EvidencePack) -> str:
    rows = []
    for item in pack.evidence:
        line = "" if item.line is None else f":{item.line}"
        rows.append(
            "<tr>"
            f"<td>{escape(item.kind)}</td>"
            f"<td>{escape(item.path + line)}</td>"
            f"<td><code>{escape(item.excerpt)}</code></td>"
            "</tr>"
        )
    body = "<table><thead><tr><th>Kind</th><th>Source</th><th>Excerpt</th></tr></thead>"
    body += f"<tbody>{''.join(rows)}</tbody></table>"
    return _section("Evidence Pack", body)


CSS = """
:root {
  color-scheme: light;
  font-family: Arial, sans-serif;
  color: #202124;
  background: #f6f7f8;
}
body {
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px 20px 56px;
}
h1 {
  margin: 0 0 24px;
  font-size: 32px;
}
h2 {
  margin: 0 0 12px;
  font-size: 20px;
}
section {
  background: #fff;
  border: 1px solid #d8dee4;
  border-radius: 6px;
  margin: 16px 0;
  padding: 18px;
}
dl {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 8px 18px;
}
dt {
  font-weight: 700;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  border-top: 1px solid #d8dee4;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}
code {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
""".strip()
