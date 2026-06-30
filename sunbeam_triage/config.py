from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SWIFT_BASE_URL = (
    "https://radosgw.ps7.canonical.com/swift/v1/"
    "AUTH_86bac34f174b4ae59994bd51884a9c53/solutions-qa"
)


@dataclass
class SwiftConfig:
    base_url: str = DEFAULT_SWIFT_BASE_URL
    timeout_seconds: int = 60


@dataclass
class LlmConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openrouter/auto"
    api_key_env: str = "OPENROUTER_API_KEY"
    api_key: str | None = None
    timeout_seconds: int = 120


@dataclass
class TriageConfig:
    quick_max_rounds: int = 5
    default_max_rounds: int = 12
    hard_max_rounds: int = 20
    stall_limit: int = 3
    min_evidence_items: int = 2
    max_tool_result_chars: int = 60_000


@dataclass
class ArenaConfig:
    models: list[str] = field(default_factory=list)


@dataclass
class PathConfig:
    artifact_root: Path = Path("artifacts")
    output_pattern: str = "diagnostics-{uuid}.html"


@dataclass
class Config:
    swift: SwiftConfig = field(default_factory=SwiftConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def load(
        cls,
        path: str | Path | None,
        *,
        cli_model: str | None = None,
        cli_output: str | Path | None = None,
        cli_artifact_root: str | Path | None = None,
    ) -> "Config":
        config = cls()
        data: dict[str, Any] = {}
        if path:
            path = Path(path)
            if path.exists():
                data = tomllib.loads(path.read_text(encoding="utf-8"))

        swift = data.get("swift", {})
        config.swift.base_url = str(swift.get("base_url", config.swift.base_url)).rstrip("/")
        config.swift.timeout_seconds = int(
            swift.get("timeout_seconds", config.swift.timeout_seconds)
        )

        llm = data.get("llm", {})
        config.llm.base_url = str(llm.get("base_url", config.llm.base_url)).rstrip("/")
        config.llm.model = str(llm.get("model", config.llm.model))
        config.llm.api_key_env = str(llm.get("api_key_env", config.llm.api_key_env))
        config.llm.timeout_seconds = int(
            llm.get("timeout_seconds", config.llm.timeout_seconds)
        )

        triage = data.get("triage", {})
        config.triage.quick_max_rounds = int(
            triage.get("quick_max_rounds", config.triage.quick_max_rounds)
        )
        config.triage.default_max_rounds = int(
            triage.get("default_max_rounds", config.triage.default_max_rounds)
        )
        config.triage.hard_max_rounds = int(
            triage.get("hard_max_rounds", config.triage.hard_max_rounds)
        )
        config.triage.stall_limit = int(
            triage.get("stall_limit", config.triage.stall_limit)
        )
        config.triage.min_evidence_items = int(
            triage.get("min_evidence_items", config.triage.min_evidence_items)
        )
        config.triage.max_tool_result_chars = int(
            triage.get("max_tool_result_chars", config.triage.max_tool_result_chars)
        )

        arena = data.get("arena", {})
        config.arena.models = [
            str(model) for model in arena.get("models", config.arena.models)
        ]

        paths = data.get("paths", {})
        config.paths.artifact_root = Path(
            paths.get("artifact_root", config.paths.artifact_root)
        )
        config.paths.output_pattern = str(
            paths.get("output_pattern", config.paths.output_pattern)
        )

        env_model = os.environ.get("OPENROUTER_MODEL")
        if env_model:
            config.llm.model = env_model
        if cli_model:
            config.llm.model = cli_model
        if cli_output:
            config.paths.output_pattern = str(cli_output)
        if cli_artifact_root:
            config.paths.artifact_root = Path(cli_artifact_root)

        config.llm.api_key = os.environ.get(config.llm.api_key_env)
        return config

    def output_path(self, uuid: str) -> Path:
        return Path(self.paths.output_pattern.format(uuid=uuid))
