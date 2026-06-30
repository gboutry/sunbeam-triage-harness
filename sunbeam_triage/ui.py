from __future__ import annotations

import sys
from pathlib import Path


def streamlit_argv(argv: list[str] | None = None) -> list[str]:
    return ["streamlit", "run", str(_app_path()), *list(argv or [])]


def main(argv: list[str] | None = None) -> int | None:
    from streamlit.web import cli as streamlit_cli

    previous_argv = sys.argv
    sys.argv = streamlit_argv(sys.argv[1:] if argv is None else argv)
    try:
        return streamlit_cli.main()
    finally:
        sys.argv = previous_argv


def _app_path() -> Path:
    import streamlit_app

    return Path(streamlit_app.__file__).resolve()
