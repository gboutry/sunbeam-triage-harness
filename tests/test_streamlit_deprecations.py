import ast
from pathlib import Path


def test_streamlit_app_does_not_use_deprecated_container_width_keyword():
    tree = ast.parse(Path("streamlit_app.py").read_text(encoding="utf-8"))

    deprecated_keywords = [
        keyword.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "use_container_width"
    ]

    assert deprecated_keywords == []
