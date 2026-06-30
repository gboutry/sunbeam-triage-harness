import tomllib
from pathlib import Path


def test_pyproject_declares_package_script_and_dependencies():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["name"] == "sunbeam-triage"
    assert data["project"]["scripts"]["sunbeam-triage"] == "sunbeam_triage.cli:main"
    assert data["project"]["scripts"]["sunbeam-triage-cli"] == "sunbeam_triage.cli:main"
    assert data["project"]["scripts"]["sunbeam-triage-ui"] == "sunbeam_triage.ui:main"
    assert (
        data["project"]["scripts"]["sunbeam-triage-analyze-rounds"]
        == "sunbeam_triage.cli.analyze_rounds:main"
    )
    assert "streamlit" in data["project"]["dependencies"]
    assert "openrouter>=0.10.0" in data["project"]["dependencies"]
    assert "pytest" in data["dependency-groups"]["dev"]
    assert data["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "sunbeam_triage*"
    ]
    assert "py-modules" not in data["tool"]["setuptools"]
