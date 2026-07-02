from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
SUNBEAM_ENABLE_PRO_TOKEN_OPTION = re.compile(
    r"(?P<prefix>\bsunbeam\s+enable\s+pro\b(?P<args>[^\n]*?)--token\s+)"
    r"(?P<value>\S+)"
)
SUNBEAM_ENABLE_PRO_POSITIONAL = re.compile(
    r"(?P<prefix>\bsunbeam\s+enable\s+pro\b(?P<args>[^\n]*?))"
    r"(?P<value>\S*(?:token|[A-Za-z0-9]*[a-z][A-Za-z0-9]*[A-Z]|"
    r"[A-Za-z0-9]*[A-Z][A-Za-z0-9]*[a-z])\S{12,})"
    r"(?P<suffix>\s*|$)"
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b"
    r"(?P<key>[A-Z0-9_.-]*(?:PASSWORD|PASSWD|TOKEN|SECRET|API[_-]?KEY|"
    r"ACCESS[_-]?KEY)[A-Z0-9_.-]*)"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<quote>['\"]?)"
    r"(?P<value>[^'\"\s,;}]+)"
    r"(?P=quote)"
)
AUTHORIZATION_VALUE = re.compile(
    r"(?i)\b(?P<prefix>authorization\s*[:=]\s*(?:bearer|basic)\s+)"
    r"(?P<value>[A-Za-z0-9._~+/=-]{8,})"
)
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}\b")
KNOWN_TOKEN_PREFIX = re.compile(
    r"\b(?:sk-or-v1-|sk-[A-Za-z0-9]|ghp_|gho_|github_pat_|xox[baprs]-|"
    r"AKIA|ASIA)[A-Za-z0-9_.:-]{12,}\b"
)
LONG_MIXED_TOKEN = re.compile(r"\b[A-Za-z0-9_+=]{32,}\b")
URL_WITH_CREDENTIALS = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>'\"]+")


def redact_text(text: str) -> str:
    redacted = PRIVATE_KEY_BLOCK.sub("<redacted private key block>", text)
    redacted = SUNBEAM_ENABLE_PRO_TOKEN_OPTION.sub(
        lambda match: f"{match.group('prefix')}<redacted>",
        redacted,
    )
    redacted = SUNBEAM_ENABLE_PRO_POSITIONAL.sub(_redact_sunbeam_enable_pro, redacted)
    redacted = SECRET_ASSIGNMENT.sub(_redact_secret_assignment, redacted)
    redacted = AUTHORIZATION_VALUE.sub(
        lambda match: f"{match.group('prefix')}<redacted>",
        redacted,
    )
    redacted = URL_WITH_CREDENTIALS.sub(_redact_url_credentials, redacted)
    redacted = JWT.sub("<redacted>", redacted)
    redacted = KNOWN_TOKEN_PREFIX.sub("<redacted>", redacted)
    return LONG_MIXED_TOKEN.sub(_redact_long_mixed_token, redacted)


def redact_data(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, int | float | bool):
        return value
    if isinstance(value, dict):
        return {copy.deepcopy(key): redact_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    return value


def _redact_secret_assignment(match: re.Match[str]) -> str:
    return (
        f"{match.group('key')}{match.group('sep')}"
        f"{match.group('quote')}<redacted>{match.group('quote')}"
    )


def _redact_sunbeam_enable_pro(match: re.Match[str]) -> str:
    args = match.group("args")
    if args.rstrip().endswith("--token"):
        return match.group(0)
    return f"{match.group('prefix')}<redacted>{match.group('suffix')}"


def _redact_url_credentials(match: re.Match[str]) -> str:
    value = match.group(0)
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.username and not parsed.password:
        return value
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((
        parsed.scheme,
        f"<redacted>@{host}",
        parsed.path,
        parsed.query,
        parsed.fragment,
    ))


def _redact_long_mixed_token(match: re.Match[str]) -> str:
    value = match.group(0)
    if (
        any(char.islower() for char in value)
        and any(char.isupper() for char in value)
        and any(char.isdigit() for char in value)
    ):
        return "<redacted>"
    return value
