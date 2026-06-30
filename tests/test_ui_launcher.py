from pathlib import Path

from sunbeam_triage import ui


def test_streamlit_argv_launches_packaged_cockpit():
    argv = ui.streamlit_argv(["--server.port", "8502"])

    assert argv[:2] == ["streamlit", "run"]
    assert Path(argv[2]).name == "app.py"
    assert argv[3:] == ["--server.port", "8502"]
