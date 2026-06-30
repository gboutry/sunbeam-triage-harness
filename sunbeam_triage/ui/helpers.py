from __future__ import annotations

import copy
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from ..core.llm import DiagnosisReport
from ..core.redaction import redact_data, redact_text

MANIFEST_NAME = ".sunbeam-triage-manifest.json"
STORE_DIR_NAME = ".sunbeam-triage"
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

    def post_json(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> Any:
        response = self.wrapped.post_json(url, payload, headers)
        self.exchanges.append({
            "url": url,
            "request": {
                "payload": redact_data(copy.deepcopy(payload)),
                "headers": _redact_headers(headers),
            },
            "response": redact_data(copy.deepcopy(response)),
        })
        return response


def list_artifact_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    files = [
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and not _is_internal_path(path.relative_to(root))
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
    text = redact_text(data[:max_bytes].decode("utf-8", errors="replace"))
    return TextPreview(text=text, truncated=truncated, binary=False)


def render_line_preview(text: str, highlighted_lines: set[int]) -> str:
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        class_attr = ' class="evidence-line"' if number in highlighted_lines else ""
        rows.append(
            f'<tr{class_attr} data-line="{number}">'
            f'<td class="line-number">{number}</td>'
            f'<td class="line-text"><code>{escape(line)}</code></td>'
            "</tr>"
        )
    return '<table class="file-preview"><tbody>' + "\n".join(rows) + "</tbody></table>"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: "<redacted>" if key.lower() == "authorization" else redact_text(value)
        for key, value in headers.items()
    }


def _is_internal_path(path: Path) -> bool:
    return (
        path.name == MANIFEST_NAME
        or STORE_DIR_NAME in path.parts
        or SESSION_DIR_NAME in path.parts
    )
