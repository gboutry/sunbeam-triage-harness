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
    ]
    read_description = tools[1]["function"]["description"]
    assert "costly" in read_description
    assert "not first" in read_description


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
