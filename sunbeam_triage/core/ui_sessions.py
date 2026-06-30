from __future__ import annotations

import json
from operator import itemgetter
from pathlib import Path
from typing import Any

from .sessions import SESSION_DIR_NAME


def session_store_root(artifact_root: Path) -> Path:
    return Path(artifact_root) / SESSION_DIR_NAME


def save_ui_session(artifact_root: Path, session: dict[str, Any]) -> None:
    path = _session_path(artifact_root, str(session["uuid"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")


def load_ui_session(artifact_root: Path, uuid: str) -> dict[str, Any] | None:
    path = _session_path(artifact_root, uuid)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_saved_sessions(artifact_root: Path) -> list[dict[str, Any]]:
    sessions_dir = session_store_root(artifact_root) / "sessions"
    if not sessions_dir.exists():
        return []
    summaries = []
    for path in sessions_dir.glob("*.json"):
        session = json.loads(path.read_text(encoding="utf-8"))
        chat = session.get("chat", [])
        summaries.append({
            "uuid": str(session.get("uuid", path.stem)),
            "model": str(session.get("model", "")),
            "summary": str(session.get("summary", "")),
            "confidence": str(session.get("confidence", "")),
            "updated_at": str(session.get("updated_at", "")),
            "chat_count": len(chat) if isinstance(chat, list) else 0,
        })
    return sorted(summaries, key=itemgetter("updated_at"), reverse=True)


def _session_path(artifact_root: Path, uuid: str) -> Path:
    return session_store_root(artifact_root) / "sessions" / f"{uuid}.json"
