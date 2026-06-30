from __future__ import annotations

import re
import tarfile
from operator import itemgetter
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_LIST_LIMIT = 200
DEFAULT_SEARCH_LIMIT = 50
DEFAULT_MAX_BYTES = 120_000
MAX_BYTES_LIMIT = 250_000
SEARCH_MEMBER_MAX_BYTES = 2_000_000


def list_sosreports(root: Path) -> dict[str, Any]:
    archives = []
    for path in sorted(Path(root).rglob("sosreport-*.tar*")):
        if not path.is_file() or path.name.endswith(".sha256"):
            continue
        relative = path.relative_to(root).as_posix()
        archives.append({
            "path": relative,
            "host": _host_from_archive_name(path.name),
            "size_bytes": path.stat().st_size,
        })
    return {"ok": True, "archives": archives}


def list_sosreport_files(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    archive_ref = _archive_ref(root, arguments)
    if not archive_ref["ok"]:
        return archive_ref

    prefix = str(arguments.get("prefix", ""))
    limit = _coerce_limit(arguments.get("limit"), DEFAULT_LIST_LIMIT)
    files = []
    with tarfile.open(archive_ref["path"], "r:*") as archive:
        for member in archive.getmembers():
            normalized = _normalized_member_path(member.name)
            if normalized is None or not member.isfile():
                continue
            if prefix and not normalized.startswith(prefix):
                continue
            files.append({"path": normalized, "size_bytes": member.size})
    files = sorted(files, key=itemgetter("path"))
    return {
        "ok": True,
        "archive_path": archive_ref["relative"],
        "files": files[:limit],
        "truncated": len(files) > limit,
    }


def search_sosreport(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    archive_ref = _archive_ref(root, arguments)
    if not archive_ref["ok"]:
        return archive_ref

    raw_pattern = str(arguments.get("pattern", ""))
    if not raw_pattern:
        return {"ok": False, "error": "Search pattern is required."}
    try:
        pattern = re.compile(raw_pattern, re.IGNORECASE)
    except re.error as exc:
        return {"ok": False, "error": f"Invalid search pattern: {exc}"}

    prefix = str(arguments.get("prefix", ""))
    limit = _coerce_limit(arguments.get("limit"), DEFAULT_SEARCH_LIMIT)
    matches = []
    truncated = False
    with tarfile.open(archive_ref["path"], "r:*") as archive:
        for member in archive.getmembers():
            normalized = _normalized_member_path(member.name)
            if normalized is None or not member.isfile():
                continue
            if prefix and not normalized.startswith(prefix):
                continue
            if member.size > SEARCH_MEMBER_MAX_BYTES:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read(SEARCH_MEMBER_MAX_BYTES + 1)
            if _is_binary(data):
                continue
            text = data[:SEARCH_MEMBER_MAX_BYTES].decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    if len(matches) >= limit:
                        truncated = True
                        break
                    matches.append({
                        "path": normalized,
                        "line": line_number,
                        "excerpt": line.strip(),
                    })
            if truncated:
                break
    return {
        "ok": True,
        "archive_path": archive_ref["relative"],
        "matches": matches,
        "truncated": truncated,
    }


def get_sosreport_file(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    archive_ref = _archive_ref(root, arguments)
    if not archive_ref["ok"]:
        return archive_ref

    requested = str(arguments.get("member_path", ""))
    requested_path = PurePosixPath(requested)
    if _is_unsafe_member_path(requested_path):
        return {
            "ok": False,
            "error": "Sosreport member path must be a safe relative path.",
        }

    max_bytes = _coerce_max_bytes(arguments.get("max_bytes"))
    line_start = _optional_positive_int(arguments.get("line_start"))
    line_count = _optional_positive_int(arguments.get("line_count"))

    with tarfile.open(archive_ref["path"], "r:*") as archive:
        member = _find_member(archive, requested)
        if member is None:
            return {
                "ok": False,
                "archive_path": archive_ref["relative"],
                "path": requested,
                "error": "Sosreport member does not exist.",
            }
        extracted = archive.extractfile(member)
        if extracted is None:
            return {
                "ok": False,
                "archive_path": archive_ref["relative"],
                "path": requested,
                "error": "Sosreport member is not readable.",
            }
        data = extracted.read(max_bytes + 1)

    if _is_binary(data):
        return {
            "ok": False,
            "archive_path": archive_ref["relative"],
            "path": requested,
            "size_bytes": member.size,
            "error": "Binary sosreport member preview is not available.",
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
        "archive_path": archive_ref["relative"],
        "path": requested,
        "size_bytes": member.size,
        "content": text,
        "truncated": truncated,
        "binary": False,
    }
    if line_start is not None:
        result["line_start"] = line_start
        result["line_count"] = (
            line_count if line_count is not None else len(text.splitlines())
        )
    return result


def _archive_ref(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    requested = str(arguments.get("archive_path", ""))
    relative = Path(requested)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return {
            "ok": False,
            "error": "Sosreport archive path must be a safe relative path.",
        }
    path = Path(root) / relative
    if not path.exists() or not path.is_file():
        return {
            "ok": False,
            "archive_path": relative.as_posix(),
            "error": "Sosreport archive does not exist.",
        }
    if not path.name.startswith("sosreport-") or ".tar" not in path.name:
        return {
            "ok": False,
            "archive_path": relative.as_posix(),
            "error": "Path is not a sosreport tar archive.",
        }
    return {"ok": True, "path": path, "relative": relative.as_posix()}


def _find_member(archive: tarfile.TarFile, requested: str) -> tarfile.TarInfo | None:
    for member in archive.getmembers():
        normalized = _normalized_member_path(member.name)
        if normalized == requested and member.isfile():
            return member
    return None


def _normalized_member_path(name: str) -> str | None:
    path = PurePosixPath(name)
    if _is_unsafe_member_path(path):
        return None
    parts = path.parts
    if not parts:
        return None
    if parts[0].startswith("sosreport-"):
        parts = parts[1:]
    if not parts:
        return None
    return PurePosixPath(*parts).as_posix()


def _is_unsafe_member_path(path: PurePosixPath) -> bool:
    return path.is_absolute() or not path.parts or ".." in path.parts


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[: min(len(data), 4096)]


def _host_from_archive_name(name: str) -> str:
    match = re.match(r"^sosreport-(?P<host>.+)-\d{4}-\d{2}-\d{2}-[^.]+\.tar", name)
    if match:
        return match.group("host")
    return ""


def _coerce_limit(value: Any, default: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(limit, 500))


def _coerce_max_bytes(value: Any) -> int:
    try:
        max_bytes = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES
    return min(max(max_bytes, 1), MAX_BYTES_LIMIT)


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
