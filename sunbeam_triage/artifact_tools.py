from __future__ import annotations

from pathlib import Path
from typing import Any


MANIFEST_NAME = ".sunbeam-triage-manifest.json"
SESSION_DIR_NAME = ".sunbeam-triage-ui"
DEFAULT_MAX_BYTES = 120_000
MAX_BYTES_LIMIT = 250_000


def artifact_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_artifact_files",
                "description": (
                    "List artifact files available for the active diagnosis. "
                    "Use this before reading files when the provided evidence "
                    "is insufficient."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_artifact_file",
                "description": (
                    "Read bounded text content from one artifact file. Getting "
                    "a file can be costly and noisy, so this is not first "
                    "intent: use provided evidence and list_artifact_files "
                    "before reading a specific file."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path"],
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Relative path from the active artifact root."
                            ),
                        },
                        "max_bytes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_BYTES_LIMIT,
                            "description": (
                                "Maximum bytes to read. Defaults to 120000."
                            ),
                        },
                    },
                },
            },
        },
    ]


def execute_artifact_tool(
    root: Path,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if name == "list_artifact_files":
        return _list_artifact_files(root)
    if name == "get_artifact_file":
        return _get_artifact_file(root, arguments)
    return {"ok": False, "error": f"Unknown artifact tool: {name}"}


def _list_artifact_files(root: Path) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        return {"ok": True, "files": []}
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_internal_path(relative):
            continue
        files.append({"path": relative.as_posix(), "size_bytes": path.stat().st_size})
    return {"ok": True, "files": sorted(files, key=lambda item: item["path"])}


def _get_artifact_file(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    requested = str(arguments.get("path", ""))
    relative = Path(requested)
    if _is_unsafe_relative_path(relative) or _is_internal_path(relative):
        return {
            "ok": False,
            "error": "Artifact path must be a safe relative path inside the root.",
        }
    path = Path(root) / relative
    if not path.exists() or not path.is_file():
        return {
            "ok": False,
            "path": relative.as_posix(),
            "error": "Artifact file does not exist.",
        }

    max_bytes = _coerce_max_bytes(arguments.get("max_bytes"))
    size = path.stat().st_size
    data = path.read_bytes()
    if b"\x00" in data[: min(len(data), max_bytes)]:
        return {
            "ok": False,
            "path": relative.as_posix(),
            "size_bytes": size,
            "error": "Binary file preview is not available.",
            "binary": True,
        }
    truncated = len(data) > max_bytes
    return {
        "ok": True,
        "path": relative.as_posix(),
        "size_bytes": size,
        "content": data[:max_bytes].decode("utf-8", errors="replace"),
        "truncated": truncated,
        "binary": False,
    }


def _coerce_max_bytes(value: Any) -> int:
    if value is None:
        return DEFAULT_MAX_BYTES
    try:
        max_bytes = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES
    return min(max(max_bytes, 1), MAX_BYTES_LIMIT)


def _is_unsafe_relative_path(path: Path) -> bool:
    return path.is_absolute() or not path.parts or ".." in path.parts


def _is_internal_path(path: Path) -> bool:
    return path.name == MANIFEST_NAME or SESSION_DIR_NAME in path.parts
