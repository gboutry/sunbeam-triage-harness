import ast
from pathlib import Path


def test_streamlit_app_does_not_use_deprecated_container_width_keyword():
    tree = ast.parse(Path("sunbeam_triage/ui/app.py").read_text(encoding="utf-8"))

    deprecated_keywords = [
        keyword.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "use_container_width"
    ]

    assert deprecated_keywords == []


def test_streamlit_app_does_not_import_backend_orchestration_dependencies():
    tree = ast.parse(Path("sunbeam_triage/ui/app.py").read_text(encoding="utf-8"))
    forbidden = {
        "sunbeam_triage.core.evidence": {"EvidenceCollector"},
        "sunbeam_triage.core.llm": {"OpenRouterClient"},
        "sunbeam_triage.core.render": {"render_html"},
        "sunbeam_triage.core.sessions": {
            "append_session_event",
            "save_session_snapshot",
        },
        "sunbeam_triage.core.swift": {"SwiftMirror"},
    }
    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        imported = forbidden.get(node.module or "")
        if not imported:
            continue
        violations.extend(
            (node.module, alias.name, node.lineno)
            for alias in node.names
            if alias.name in imported
        )

    assert violations == []
