from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from .evidence import EvidencePack
from .llm import DiagnosisReport


MANIFEST_NAME = ".sunbeam-triage-manifest.json"
SESSION_DIR_NAME = ".sunbeam-triage-ui"


@dataclass(frozen=True)
class TextPreview:
    text: str
    truncated: bool
    binary: bool


class CapturingHttp:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.exchanges: list[dict[str, Any]] = []

    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        response = self.wrapped.post_json(url, payload, headers)
        self.exchanges.append(
            {
                "url": url,
                "request": {
                    "payload": copy.deepcopy(payload),
                    "headers": _redact_headers(headers),
                },
                "response": copy.deepcopy(response),
            }
        )
        return response


def list_artifact_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    files = [
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    ]
    return sorted(files, key=lambda path: path.as_posix())


def evidence_line_map(report: DiagnosisReport) -> dict[str, set[int]]:
    lines_by_path: dict[str, set[int]] = {}
    for item in report.evidence:
        if item.line is None:
            continue
        lines_by_path.setdefault(item.path, set()).add(item.line)
    return lines_by_path


def read_text_preview(path: Path, *, max_bytes: int = 250_000) -> TextPreview:
    data = Path(path).read_bytes()
    if b"\x00" in data[: min(len(data), max_bytes)]:
        return TextPreview(text="", truncated=False, binary=True)
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    return TextPreview(text=text, truncated=truncated, binary=False)


def render_line_preview(text: str, highlighted_lines: set[int]) -> str:
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        class_attr = ' class="evidence-line"' if number in highlighted_lines else ""
        rows.append(
            f"<tr{class_attr} data-line=\"{number}\">"
            f'<td class="line-number">{number}</td>'
            f'<td class="line-text"><code>{escape(line)}</code></td>'
            "</tr>"
        )
    return '<table class="file-preview"><tbody>' + "\n".join(rows) + "</tbody></table>"


def session_store_root(artifact_root: Path) -> Path:
    return Path(artifact_root) / SESSION_DIR_NAME


def save_ui_session(artifact_root: Path, session: dict[str, Any]) -> None:
    path = _session_path(artifact_root, str(session["uuid"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")


def load_ui_session(artifact_root: Path, uuid: str) -> dict[str, Any] | None:
    path = _session_path(artifact_root, uuid)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_saved_sessions(artifact_root: Path) -> list[dict[str, Any]]:
    sessions_dir = session_store_root(artifact_root) / "sessions"
    if not sessions_dir.exists():
        return []
    summaries = []
    for path in sessions_dir.glob("*.json"):
        session = json.loads(path.read_text(encoding="utf-8"))
        chat = session.get("chat", [])
        summaries.append(
            {
                "uuid": str(session.get("uuid", path.stem)),
                "model": str(session.get("model", "")),
                "summary": str(session.get("summary", "")),
                "confidence": str(session.get("confidence", "")),
                "updated_at": str(session.get("updated_at", "")),
                "chat_count": len(chat) if isinstance(chat, list) else 0,
            }
        )
    return sorted(summaries, key=lambda item: item["updated_at"], reverse=True)


def build_followup_context(
    pack: EvidencePack,
    report: DiagnosisReport,
    *,
    attachments: list[dict[str, Any]] | None = None,
) -> str:
    parts = [
        "You are answering a follow-up question about this active diagnosis.",
        f"Solutions Run UUID: {pack.uuid}",
        f"Run ID: {pack.run.run_id}",
        f"Branch: {pack.run.branch}",
        f"Workflow: {pack.run.workflow}",
        f"Failed Step: {pack.failed_step.name}",
        f"Diagnosis Summary: {report.summary}",
        f"Failure Surface: {report.failure_surface}",
        f"Confidence: {report.confidence}",
        f"Triage Confidence: {report.triage_confidence}",
        f"Stop Reason: {report.stop_reason}",
        f"Root Cause: {report.root_cause}",
        "",
        "Model Evidence:",
    ]
    for item in report.evidence:
        line = "" if item.line is None else f":{item.line}"
        parts.append(f"- {item.path}{line}: {item.excerpt}")
    parts.extend(["", "Harness Evidence:"])
    for item in pack.evidence:
        line = "" if item.line is None else f":{item.line}"
        parts.append(f"- [{item.kind}] {item.path}{line}: {item.excerpt}")
    probe_lines = _probe_context_lines(pack)
    if probe_lines:
        parts.extend(["", "Deterministic Probes:", *probe_lines])
    if report.recommendations:
        parts.extend(["", "Recommendations:"])
        parts.extend(f"- {item}" for item in report.recommendations)
    if report.unknowns:
        parts.extend(["", "Unknowns:"])
        parts.extend(f"- {item}" for item in report.unknowns)
    if report.failure_timeline:
        parts.extend(["", "Failure Timeline:"])
        parts.extend(
            (
                f"- {item.timestamp} {item.source} {item.location}: "
                f"{item.event}"
            )
            for item in report.failure_timeline
        )
    if report.cascading_errors:
        parts.extend(["", "Cascading Errors:"])
        for item in report.cascading_errors:
            line = "" if item.line is None else f":{item.line}"
            parts.append(f"- {item.path}{line}: {item.excerpt}")
    if report.alternatives_considered:
        parts.extend(["", "Alternatives Considered:"])
        parts.extend(
            f"- {item.hypothesis} ({item.status}): {item.reason}"
            for item in report.alternatives_considered
        )
    if report.missing_evidence:
        parts.extend(["", "Missing Evidence:"])
        parts.extend(f"- {item}" for item in report.missing_evidence)
    if attachments:
        parts.extend(["", "Attached Context:"])
        for item in attachments:
            line = "" if item.get("line") is None else f":{item['line']}"
            parts.append(f"- {item.get('path', '')}{line}: {item.get('text', '')}")
    return "\n".join(parts)


def _probe_context_lines(pack: EvidencePack) -> list[str]:
    lines: list[str] = []
    for result in pack.probe_results:
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


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: "<redacted>" if key.lower() == "authorization" else value
        for key, value in headers.items()
    }


def _session_path(artifact_root: Path, uuid: str) -> Path:
    return session_store_root(artifact_root) / "sessions" / f"{uuid}.json"
