import io
import tarfile
from pathlib import Path

from sunbeam_triage.artifact_tools import (
    artifact_tool_definitions,
    execute_artifact_tool,
)


def test_artifact_tool_definitions_warn_that_file_reads_are_costly():
    tools = artifact_tool_definitions()

    assert [tool["function"]["name"] for tool in tools] == [
        "list_artifact_files",
        "get_artifact_file",
        "list_sosreports",
        "list_sosreport_files",
        "search_sosreport",
        "get_sosreport_file",
    ]
    read_description = tools[1]["function"]["description"]
    assert "costly" in read_description
    assert "not first" in read_description
    sos_read_description = tools[-1]["function"]["description"]
    assert "costly" in sos_read_description
    assert "search_sosreport" in sos_read_description


def test_list_artifact_files_returns_sorted_relative_paths_and_sizes(tmp_path):
    root = tmp_path / "uuid"
    (root / "generated/sunbeam").mkdir(parents=True)
    (root / "generated/sunbeam/output.log").write_text("log", encoding="utf-8")
    (root / "generated/github-runner").mkdir(parents=True)
    (root / "generated/github-runner/jobs.json").write_text("{}", encoding="utf-8")
    (root / ".sunbeam-triage-manifest.json").write_text("[]", encoding="utf-8")
    (root / ".sunbeam-triage-ui/sessions").mkdir(parents=True)
    (root / ".sunbeam-triage-ui/sessions/uuid.json").write_text(
        "{}", encoding="utf-8"
    )

    result = execute_artifact_tool(root, "list_artifact_files", {})

    assert result == {
        "ok": True,
        "files": [
            {"path": "generated/github-runner/jobs.json", "size_bytes": 2},
            {"path": "generated/sunbeam/output.log", "size_bytes": 3},
        ],
    }


def test_get_artifact_file_returns_bounded_text(tmp_path):
    root = tmp_path / "uuid"
    path = root / "generated/sunbeam/output.log"
    path.parent.mkdir(parents=True)
    path.write_text("abcdef", encoding="utf-8")

    result = execute_artifact_tool(
        root,
        "get_artifact_file",
        {"path": "generated/sunbeam/output.log", "max_bytes": 3},
    )

    assert result == {
        "ok": True,
        "path": "generated/sunbeam/output.log",
        "size_bytes": 6,
        "content": "abc",
        "truncated": True,
        "binary": False,
    }


def test_get_artifact_file_rejects_paths_outside_artifact_root(tmp_path):
    result = execute_artifact_tool(
        tmp_path / "uuid",
        "get_artifact_file",
        {"path": "../outside.log"},
    )

    assert result["ok"] is False
    assert "relative path" in result["error"]


def test_get_artifact_file_reports_binary_without_content(tmp_path):
    root = tmp_path / "uuid"
    path = root / "blob.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc\x00def")

    result = execute_artifact_tool(root, "get_artifact_file", {"path": "blob.bin"})

    assert result == {
        "ok": False,
        "path": "blob.bin",
        "size_bytes": 7,
        "error": "Binary file preview is not available.",
        "binary": True,
    }


def test_unknown_artifact_tool_returns_error(tmp_path):
    result = execute_artifact_tool(tmp_path / "uuid", "delete_artifact_file", {})

    assert result == {
        "ok": False,
        "error": "Unknown artifact tool: delete_artifact_file",
    }


def test_list_sosreports_returns_archives_with_host_and_member_count(tmp_path):
    root = tmp_path / "uuid"
    archive = root / "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz"
    _write_sosreport(
        archive,
        {
            "sosreport-node-a-2026-06-23-abc/var/log/syslog": "booted\n",
            "sosreport-node-a-2026-06-23-abc/sos_commands/block/lsblk": "sda\n",
        },
    )
    (archive.parent / f"{archive.name}.sha256").write_text("checksum", encoding="utf-8")

    result = execute_artifact_tool(root, "list_sosreports", {})

    assert result == {
        "ok": True,
        "archives": [
            {
                "path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
                "host": "node-a",
                "size_bytes": archive.stat().st_size,
                "member_count": 2,
            }
        ],
    }


def test_list_sosreport_files_lists_normalized_members_with_prefix_and_limit(tmp_path):
    root = tmp_path / "uuid"
    archive = root / "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz"
    _write_sosreport(
        archive,
        {
            "sosreport-node-a-2026-06-23-abc/var/log/syslog": "booted\n",
            "sosreport-node-a-2026-06-23-abc/var/log/kern.log": "kernel\n",
            "sosreport-node-a-2026-06-23-abc/sos_commands/block/lsblk": "sda\n",
        },
    )

    result = execute_artifact_tool(
        root,
        "list_sosreport_files",
        {
            "archive_path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
            "prefix": "var/log/",
            "limit": 1,
        },
    )

    assert result == {
        "ok": True,
        "archive_path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
        "files": [{"path": "var/log/kern.log", "size_bytes": 7}],
        "truncated": True,
    }


def test_search_sosreport_returns_bounded_line_matches(tmp_path):
    root = tmp_path / "uuid"
    archive = root / "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz"
    _write_sosreport(
        archive,
        {
            "sosreport-node-a-2026-06-23-abc/var/log/syslog": (
                "ok\nERROR first failure\nok\nERROR second failure\n"
            ),
            "sosreport-node-a-2026-06-23-abc/var/log/kern.log": "kernel ok\n",
        },
    )

    result = execute_artifact_tool(
        root,
        "search_sosreport",
        {
            "archive_path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
            "pattern": "ERROR",
            "prefix": "var/log/",
            "limit": 1,
        },
    )

    assert result == {
        "ok": True,
        "archive_path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
        "matches": [
            {
                "path": "var/log/syslog",
                "line": 2,
                "excerpt": "ERROR first failure",
            }
        ],
        "truncated": True,
    }


def test_get_sosreport_file_reads_bounded_member_line_window(tmp_path):
    root = tmp_path / "uuid"
    archive = root / "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz"
    _write_sosreport(
        archive,
        {
            "sosreport-node-a-2026-06-23-abc/var/log/syslog": (
                "line 1\nline 2\nline 3\nline 4\n"
            ),
        },
    )

    result = execute_artifact_tool(
        root,
        "get_sosreport_file",
        {
            "archive_path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
            "member_path": "var/log/syslog",
            "line_start": 2,
            "line_count": 2,
        },
    )

    assert result == {
        "ok": True,
        "archive_path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
        "path": "var/log/syslog",
        "size_bytes": 28,
        "content": "line 2\nline 3",
        "truncated": False,
        "binary": False,
        "line_start": 2,
        "line_count": 2,
    }


def test_get_sosreport_file_rejects_unsafe_member_path(tmp_path):
    root = tmp_path / "uuid"
    archive = root / "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz"
    _write_sosreport(
        archive,
        {"sosreport-node-a-2026-06-23-abc/var/log/syslog": "line\n"},
    )

    result = execute_artifact_tool(
        root,
        "get_sosreport_file",
        {
            "archive_path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
            "member_path": "../etc/passwd",
        },
    )

    assert result["ok"] is False
    assert "safe relative" in result["error"]


def test_search_sosreport_skips_binary_members(tmp_path):
    root = tmp_path / "uuid"
    archive = root / "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz"
    _write_sosreport(
        archive,
        {
            "sosreport-node-a-2026-06-23-abc/var/log/syslog": "needle\n",
            "sosreport-node-a-2026-06-23-abc/proc/blob": b"\x00needle",
        },
    )

    result = execute_artifact_tool(
        root,
        "search_sosreport",
        {
            "archive_path": "generated/sunbeam/sosreport-node-a-2026-06-23-abc.tar.xz",
            "pattern": "needle",
            "limit": 10,
        },
    )

    assert result["matches"] == [
        {"path": "var/log/syslog", "line": 1, "excerpt": "needle"}
    ]


def _write_sosreport(path: Path, members: dict[str, str | bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:xz") as archive:
        for name, content in members.items():
            data = content if isinstance(content, bytes) else content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
