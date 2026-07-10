from __future__ import annotations

from typing import Any

from openrouter import OpenRouter

from .config import LlmConfig


class OpenRouterTransport:
    def __init__(self, config: LlmConfig, sdk_client: Any | None = None):
        self.config = config
        self.sdk_client = sdk_client

    def send(self, request: dict[str, Any]) -> Any:
        try:
            return self.sdk().chat.send(**request)
        except Exception as exc:
            if not _is_transient_provider_error(exc):
                raise
            return self.sdk().chat.send(**request)

    def sdk(self) -> Any:
        if self.sdk_client is not None:
            return self.sdk_client
        if not self.config.api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key. Set {self.config.api_key_env}."
            )
        self.sdk_client = OpenRouter(
            api_key=self.config.api_key,
            server_url=self.config.base_url,
            timeout_ms=self.config.timeout_seconds * 1000,
        )
        return self.sdk_client


def cache_kwargs(model: str) -> dict[str, Any]:
    if model.startswith("anthropic/"):
        return {"cache_control": {"type": "ephemeral"}}
    return {}


def _is_transient_provider_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "rate limit",
            "status 429",
            "status 500",
            "status 502",
            "status 503",
            "status 504",
        )
    )
