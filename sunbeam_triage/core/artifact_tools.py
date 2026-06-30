from __future__ import annotations

import fnmatch
import re
from operator import itemgetter
from pathlib import Path
from typing import Any

from .redaction import redact_text
from .sosreport_tools import (
    get_sosreport_file,
    list_sosreport_files,
    list_sosreports,
    search_sosreport,
)

MANIFEST_NAME = ".sunbeam-triage-manifest.json"
STORE_DIR_NAME = ".sunbeam-triage"
SESSION_DIR_NAME = ".sunbeam-triage-ui"
DEFAULT_MAX_BYTES = 120_000
MAX_BYTES_LIMIT = 250_000
DEFAULT_READ_MAX_BYTES = 40_000
READ_MAX_BYTES_LIMIT = 80_000
DEFAULT_SEARCH_LIMIT = 50
SEARCH_FILE_MAX_BYTES = 2_000_000
SEARCH_EXCERPT_MAX_CHARS = 500


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
                            "maximum": READ_MAX_BYTES_LIMIT,
                            "description": (
                                "Maximum bytes to read. Defaults to 40000."
                            ),
                        },
                        "line_start": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional first 1-based line to return.",
                        },
                        "line_count": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional number of lines to return.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_artifacts",
                "description": (
                    "Search downloaded text artifacts and return compact "
                    "path:line excerpts. Use this before get_artifact_file "
                    "when looking for specific errors or terms."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pattern"],
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Case-insensitive regular expression to search.",
                        },
                        "path_prefix": {
                            "type": "string",
                            "description": "Optional artifact path prefix to search under.",
                        },
                        "path_glob": {
                            "type": "string",
                            "description": "Optional glob matched against relative paths.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "description": "Maximum matching lines to return.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_sosreports",
                "description": (
                    "List sosreport tar archives available for the active diagnosis. "
                    "Use this before searching or reading sosreport contents."
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
                "name": "list_sosreport_files",
                "description": (
                    "List file members inside one sosreport archive. Use a prefix "
                    "to focus on likely useful areas before reading member content."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["archive_path"],
                    "properties": {
                        "archive_path": {
                            "type": "string",
                            "description": "Relative path to a sosreport tar archive.",
                        },
                        "prefix": {
                            "type": "string",
                            "description": (
                                "Optional normalized member prefix, for example "
                                "var/log/ or sos_commands/juju/."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "description": "Maximum members to return.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_sosreport",
                "description": (
                    "Search text members inside one sosreport archive and return "
                    "compact path:line excerpts. Use this before get_sosreport_file "
                    "to preserve context."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["archive_path", "pattern"],
                    "properties": {
                        "archive_path": {
                            "type": "string",
                            "description": "Relative path to a sosreport tar archive.",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Case-insensitive regular expression to search.",
                        },
                        "prefix": {
                            "type": "string",
                            "description": "Optional normalized member prefix to search under.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "description": "Maximum matching lines to return.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_sosreport_file",
                "description": (
                    "Read bounded text from one sosreport member. This can be "
                    "costly and noisy, so use list_sosreports, list_sosreport_files, "
                    "and search_sosreport first, then read a specific promising "
                    "member only."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["archive_path", "member_path"],
                    "properties": {
                        "archive_path": {
                            "type": "string",
                            "description": "Relative path to a sosreport tar archive.",
                        },
                        "member_path": {
                            "type": "string",
                            "description": (
                                "Normalized sosreport member path, without the "
                                "top-level sosreport directory."
                            ),
                        },
                        "line_start": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional first 1-based line to return.",
                        },
                        "line_count": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional number of lines to return.",
                        },
                        "max_bytes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_BYTES_LIMIT,
                            "description": "Maximum member bytes to read.",
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
    if name == "search_artifacts":
        return _search_artifacts(root, arguments)
    if name == "list_sosreports":
        return list_sosreports(root)
    if name == "list_sosreport_files":
        return list_sosreport_files(root, arguments)
    if name == "search_sosreport":
        return search_sosreport(root, arguments)
    if name == "get_sosreport_file":
        return get_sosreport_file(root, arguments)
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
    return {"ok": True, "files": sorted(files, key=itemgetter("path"))}


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

    max_bytes = _coerce_max_bytes(
        arguments.get("max_bytes"),
        default=DEFAULT_READ_MAX_BYTES,
        maximum=READ_MAX_BYTES_LIMIT,
    )
    line_start = _optional_positive_int(arguments.get("line_start"))
    line_count = _optional_positive_int(arguments.get("line_count"))
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
    text = data[:max_bytes].decode("utf-8", errors="replace")
    if line_start is not None:
        lines = text.splitlines()
        count = line_count if line_count is not None else len(lines)
        text = "\n".join(lines[line_start - 1 : line_start - 1 + count])
    result = {
        "ok": True,
        "path": relative.as_posix(),
        "size_bytes": size,
        "content": redact_text(text),
        "truncated": truncated,
        "binary": False,
    }
    if line_start is not None:
        result["line_start"] = line_start
        result["line_count"] = (
            line_count if line_count is not None else len(text.splitlines())
        )
    return result


def _search_artifacts(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_pattern = str(arguments.get("pattern", ""))
    if not raw_pattern:
        return {"ok": False, "error": "Search pattern is required."}
    try:
        pattern = re.compile(raw_pattern, re.IGNORECASE)
    except re.error as exc:
        return {"ok": False, "error": f"Invalid search pattern: {exc}"}

    path_prefix = str(arguments.get("path_prefix", ""))
    path_glob = str(arguments.get("path_glob", ""))
    limit = _coerce_limit(arguments.get("limit"), DEFAULT_SEARCH_LIMIT)
    matches = []
    truncated = False
    root = Path(root)
    for relative in _iter_searchable_files(root):
        rel_posix = relative.as_posix()
        if path_prefix and not rel_posix.startswith(path_prefix):
            continue
        if path_glob and not fnmatch.fnmatch(rel_posix, path_glob):
            continue
        path = root / relative
        if path.stat().st_size > SEARCH_FILE_MAX_BYTES:
            continue
        data = path.read_bytes()
        if b"\x00" in data[: min(len(data), 4096)]:
            continue
        text = data.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                if len(matches) >= limit:
                    truncated = True
                    break
                matches.append({
                    "path": rel_posix,
                    "line": line_number,
                    "excerpt": redact_text(_search_excerpt(line)),
                })
        if truncated:
            break
    return {"ok": True, "matches": matches, "truncated": truncated}


def _iter_searchable_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_internal_path(relative):
            continue
        if _is_archive_path(relative):
            continue
        files.append(relative)
    return sorted(files, key=lambda path: path.as_posix())


def _coerce_max_bytes(
    value: Any,
    *,
    default: int = DEFAULT_MAX_BYTES,
    maximum: int = MAX_BYTES_LIMIT,
) -> int:
    if value is None:
        return default
    try:
        max_bytes = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(max_bytes, 1), maximum)


def _coerce_limit(value: Any, default: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(limit, 500))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 1:
        return None
    return parsed


def _search_excerpt(line: str) -> str:
    excerpt = line.strip()
    if len(excerpt) <= SEARCH_EXCERPT_MAX_CHARS:
        return excerpt
    return excerpt[: SEARCH_EXCERPT_MAX_CHARS - 3] + "..."


def _is_unsafe_relative_path(path: Path) -> bool:
    return path.is_absolute() or not path.parts or ".." in path.parts


def _is_internal_path(path: Path) -> bool:
    return (
        path.name == MANIFEST_NAME
        or STORE_DIR_NAME in path.parts
        or SESSION_DIR_NAME in path.parts
    )


def _is_archive_path(path: Path) -> bool:
    return path.name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz", ".zip"))
