from pathlib import Path

from sunbeam_triage.llm import DiagnosisReport, ReportEvidence
from sunbeam_triage.ui_helpers import (
    CapturingHttp,
    evidence_line_map,
    list_artifact_files,
    read_text_preview,
    render_line_preview,
)


def test_list_artifact_files_sorts_files_and_excludes_manifest(tmp_path):
    root = tmp_path / "uuid"
    (root / "generated/sunbeam").mkdir(parents=True)
    (root / "generated/sunbeam/output.log").write_text("log", encoding="utf-8")
    (root / "generated/github-runner").mkdir(parents=True)
    (root / "generated/github-runner/jobs.json").write_text("{}", encoding="utf-8")
    (root / ".sunbeam-triage-manifest.json").write_text("[]", encoding="utf-8")

    files = list_artifact_files(root)

    assert files == [
        Path("generated/github-runner/jobs.json"),
        Path("generated/sunbeam/output.log"),
    ]


def test_evidence_line_map_groups_model_referenced_lines():
    report = DiagnosisReport(
        summary="summary",
        failure_surface="surface",
        confidence="supported",
        root_cause="cause",
        evidence=[
            ReportEvidence("generated/sunbeam/output.log", 2, "wait timed out"),
            ReportEvidence("generated/sunbeam/output.log", 4, "failed"),
            ReportEvidence("generated/github-runner/run.log", None, "exit 1"),
        ],
    )

    assert evidence_line_map(report) == {
        "generated/sunbeam/output.log": {2, 4},
    }


def test_render_line_preview_escapes_and_highlights_referenced_lines():
    html = render_line_preview("safe\n<script>\nlast", {2})

    assert "&lt;script&gt;" in html
    assert 'class="evidence-line"' in html
    assert "data-line=\"2\"" in html
    assert "<script>" not in html


def test_read_text_preview_bounds_large_text_files(tmp_path):
    path = tmp_path / "large.log"
    path.write_text("abcdef", encoding="utf-8")

    preview = read_text_preview(path, max_bytes=3)

    assert preview.text == "abc"
    assert preview.truncated is True
    assert preview.binary is False


def test_read_text_preview_marks_binary_files(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc\x00def")

    preview = read_text_preview(path)

    assert preview.text == ""
    assert preview.truncated is False
    assert preview.binary is True


class FakeHttp:
    def post_json(self, url, payload, headers):
        assert headers["Authorization"] == "Bearer secret-token"
        return {"choices": [{"message": {"content": "{}"}}]}


def test_capturing_http_redacts_authorization_and_records_exchange():
    http = CapturingHttp(FakeHttp())

    response = http.post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": "openrouter/auto"},
        {"Authorization": "Bearer secret-token"},
    )

    assert response == {"choices": [{"message": {"content": "{}"}}]}
    assert http.exchanges == [
        {
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "request": {
                "payload": {"model": "openrouter/auto"},
                "headers": {"Authorization": "<redacted>"},
            },
            "response": {"choices": [{"message": {"content": "{}"}}]},
        }
    ]
