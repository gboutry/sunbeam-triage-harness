from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STORE_DIR_NAME = ".sunbeam-triage"
SESSION_DIR_NAME = ".sunbeam-triage-ui"


def save_session_snapshot(artifact_root: Path, snapshot: dict[str, Any]) -> Path:
    session_id = str(snapshot["session_id"])
    path = _snapshot_path(artifact_root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return path


def append_session_event(
    artifact_root: Path,
    session_id: str,
    event: dict[str, Any],
) -> Path:
    path = _events_path(artifact_root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True))
        stream.write("\n")
    return path


def load_session_record(artifact_root: Path, session_id: str) -> dict[str, Any] | None:
    snapshot_path = _snapshot_path(artifact_root, session_id)
    if snapshot_path.exists():
        return {
            "snapshot": json.loads(snapshot_path.read_text(encoding="utf-8")),
            "events": _read_events(_events_path(artifact_root, session_id)),
        }
    legacy_path = _legacy_session_path(artifact_root, session_id)
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        return {
            "snapshot": _legacy_snapshot(session_id, legacy),
            "events": [],
        }
    return None


def list_session_records(artifact_root: Path) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in _sessions_dir(artifact_root).glob("*.json"):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        records[str(snapshot["session_id"])] = _session_summary(snapshot)
    for path in _legacy_sessions_dir(artifact_root).glob("*.json"):
        if path.stem in records:
            continue
        legacy = json.loads(path.read_text(encoding="utf-8"))
        snapshot = _legacy_snapshot(path.stem, legacy)
        records[path.stem] = _session_summary(snapshot)
    return sorted(records.values(), key=lambda item: item["updated_at"], reverse=True)


def export_judged_arenas(artifact_root: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for record in list_session_records(artifact_root):
            if record.get("session_type") != "arena":
                continue
            loaded = load_session_record(artifact_root, str(record["session_id"]))
            if not loaded:
                continue
            snapshot = loaded["snapshot"]
            if snapshot.get("status") != "judged" or "verdict" not in snapshot:
                continue
            stream.write(json.dumps(_arena_export_record(snapshot), sort_keys=True))
            stream.write("\n")
            count += 1
    return count


def _session_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(snapshot.get("schema_version", 1)),
        "session_id": str(snapshot.get("session_id", snapshot.get("uuid", ""))),
        "session_type": str(snapshot.get("session_type", "diagnosis")),
        "uuid": str(snapshot.get("uuid", "")),
        "summary": str(snapshot.get("summary", "")),
        "status": str(snapshot.get("status", "")),
        "updated_at": str(snapshot.get("updated_at", "")),
    }


def _legacy_snapshot(session_id: str, session: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_id": session_id,
        "session_type": "diagnosis",
        "uuid": str(session.get("uuid", session_id)),
        "updated_at": str(session.get("updated_at", "")),
        "summary": str(session.get("summary", "")),
        "status": "legacy",
        "legacy": session,
    }


def _arena_export_record(snapshot: dict[str, Any]) -> dict[str, Any]:
    verdict = snapshot.get("verdict", {})
    return {
        "schema": "sunbeam-triage-arena-eval-v1",
        "session_id": snapshot["session_id"],
        "uuid": snapshot["uuid"],
        "updated_at": snapshot.get("updated_at", ""),
        "evidence": snapshot.get("evidence", []),
        "contenders": [
            {
                "contender_id": contender.get("contender_id", ""),
                "model": contender.get("model", ""),
                "report": contender.get("report", {}),
                "trace_path": contender.get("trace_path", ""),
            }
            for contender in snapshot.get("contenders", [])
        ],
        "winner": verdict.get("winner", ""),
        "notes": verdict.get("notes", ""),
        "rubric": verdict.get("rubric", {}),
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _snapshot_path(artifact_root: Path, session_id: str) -> Path:
    return _sessions_dir(artifact_root) / f"{session_id}.json"


def _events_path(artifact_root: Path, session_id: str) -> Path:
    return _events_dir(artifact_root) / f"{session_id}.jsonl"


def _sessions_dir(artifact_root: Path) -> Path:
    return Path(artifact_root) / STORE_DIR_NAME / "sessions"


def _events_dir(artifact_root: Path) -> Path:
    return Path(artifact_root) / STORE_DIR_NAME / "events"


def _legacy_sessions_dir(artifact_root: Path) -> Path:
    return Path(artifact_root) / SESSION_DIR_NAME / "sessions"


def _legacy_session_path(artifact_root: Path, uuid: str) -> Path:
    return _legacy_sessions_dir(artifact_root) / f"{uuid}.json"
