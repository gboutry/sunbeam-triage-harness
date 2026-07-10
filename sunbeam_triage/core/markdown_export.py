from __future__ import annotations

from typing import Any

from .redaction import redact_text


def render_diagnosis_markdown(session: dict[str, Any]) -> str:
    lines = [
        f"# Sunbeam triage report: {_text(session.get('uuid')) or 'unknown'}",
        "",
        "## Metadata",
        *_metadata_lines(session),
        "",
        "## Diagnosis",
        *_diagnosis_lines(session),
        "",
    ]
    lines.extend(_evidence_section(session))
    lines.extend(_list_section("Recommendations", session.get("recommendations")))
    lines.extend(_list_section("Unknowns", session.get("unknowns")))
    lines.extend(_list_section("Missing Evidence", session.get("missing_evidence")))
    lines.extend(_conversation_section(session))
    return redact_text("\n".join(lines).rstrip() + "\n")


def _metadata_lines(session: dict[str, Any]) -> list[str]:
    fields = [
        ("UUID", session.get("uuid")),
        ("Model", session.get("model")),
        ("Failed step", session.get("failed_step")),
        ("Confidence", session.get("confidence")),
        ("Triage confidence", session.get("triage_confidence")),
        ("Stop reason", session.get("stop_reason")),
        ("Updated", session.get("updated_at")),
    ]
    return [f"- {label}: {_text(value)}" for label, value in fields if _text(value)]


def _diagnosis_lines(session: dict[str, Any]) -> list[str]:
    fields = [
        ("Summary", session.get("summary")),
        ("Failure surface", session.get("failure_surface")),
        ("Root cause", session.get("root_cause")),
    ]
    lines = [f"**{label}:** {_text(value)}" for label, value in fields if _text(value)]
    return lines or ["No diagnosis summary recorded."]


def _evidence_section(session: dict[str, Any]) -> list[str]:
    evidence = [
        item for item in _items(session.get("evidence")) if isinstance(item, dict)
    ]
    if not evidence:
        return []
    lines = ["## Evidence"]
    for item in evidence:
        source = _source_ref(item)
        excerpt = _text(item.get("excerpt"))
        if source and excerpt:
            lines.append(f"- `{source}`: {excerpt}")
        elif source:
            lines.append(f"- `{source}`")
        elif excerpt:
            lines.append(f"- {excerpt}")
    lines.append("")
    return lines


def _list_section(title: str, values: Any) -> list[str]:
    items = [_text(value) for value in _items(values) if _text(value)]
    if not items:
        return []
    return [f"## {title}", *(f"- {item}" for item in items), ""]


def _conversation_section(session: dict[str, Any]) -> list[str]:
    lines = ["## Conversation"]
    chat = [item for item in _items(session.get("chat")) if isinstance(item, dict)]
    if not chat:
        return [*lines, "No conversation recorded."]
    for message in chat:
        role = _role_title(message.get("role"))
        created_at = _text(message.get("created_at"))
        title = f"### {role}" if not created_at else f"### {role} - {created_at}"
        content = _text(message.get("content"))
        lines.extend(["", title, "", content or "(empty message)"])
    return lines


def _source_ref(item: dict[str, Any]) -> str:
    path = _text(item.get("path"))
    if not path:
        return ""
    line = item.get("line")
    if line is None or not line:
        return _escape_code_span(path)
    return _escape_code_span(f"{path}:{line}")


def _role_title(value: Any) -> str:
    role = _text(value).strip().lower()
    if role == "assistant":
        return "Assistant"
    if role == "user":
        return "User"
    return role.title() if role else "Message"


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _escape_code_span(value: str) -> str:
    return value.replace("`", "\\`")
