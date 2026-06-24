from pathlib import Path

from sunbeam_triage.config import Config


def test_config_defaults_and_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[swift]
base_url = "https://swift.example/v1/AUTH/container"

[llm]
model = "configured/model"
api_key_env = "CUSTOM_OPENROUTER_KEY"

[paths]
artifact_root = "cache"
output_pattern = "report-{uuid}.html"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_MODEL", "env/model")
    monkeypatch.setenv("CUSTOM_OPENROUTER_KEY", "secret-token")

    config = Config.load(config_path, cli_model="cli/model")

    assert config.swift.base_url == "https://swift.example/v1/AUTH/container"
    assert config.llm.model == "cli/model"
    assert config.llm.api_key == "secret-token"
    assert config.paths.artifact_root == Path("cache")
    assert config.output_path("abc") == Path("report-abc.html")


def test_config_uses_built_in_openrouter_auto_by_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    config = Config.load(None)

    assert config.llm.model == "openrouter/auto"
    assert config.llm.base_url == "https://openrouter.ai/api/v1"
