import json

from sunbeam_triage.core.sessions import (
    append_session_event,
    export_judged_arenas,
    list_session_records,
    load_session_record,
    save_session_snapshot,
)
from sunbeam_triage.ui.helpers import save_ui_session


def test_v2_session_snapshot_and_events_round_trip(tmp_path):
    artifact_root = tmp_path / "artifacts"
    snapshot = {
        "schema_version": 2,
        "session_id": "arena-sample-uuid-20260630T120000Z",
        "session_type": "arena",
        "uuid": "sample-uuid",
        "updated_at": "2026-06-30T12:00:00Z",
        "summary": "Two models compared",
        "status": "completed",
    }

    path = save_session_snapshot(artifact_root, snapshot)
    append_session_event(
        artifact_root,
        snapshot["session_id"],
        {
            "event": "arena_started",
            "created_at": "2026-06-30T12:00:00Z",
            "uuid": "sample-uuid",
        },
    )
    append_session_event(
        artifact_root,
        snapshot["session_id"],
        {
            "event": "arena_completed",
            "created_at": "2026-06-30T12:01:00Z",
            "contenders": 2,
        },
    )

    loaded = load_session_record(artifact_root, snapshot["session_id"])

    assert path == (
        artifact_root
        / ".sunbeam-triage"
        / "sessions"
        / "arena-sample-uuid-20260630T120000Z.json"
    )
    assert loaded is not None
    assert loaded["snapshot"] == snapshot
    assert [event["event"] for event in loaded["events"]] == [
        "arena_started",
        "arena_completed",
    ]


def test_session_listing_reads_v2_and_legacy_without_rewriting_legacy(tmp_path):
    artifact_root = tmp_path / "artifacts"
    legacy = {
        "uuid": "legacy-uuid",
        "model": "model/a",
        "summary": "Legacy diagnosis",
        "confidence": "supported",
        "updated_at": "2026-06-29T10:00:00Z",
    }
    save_ui_session(artifact_root, legacy)
    legacy_path = artifact_root / ".sunbeam-triage-ui" / "sessions" / "legacy-uuid.json"

    save_session_snapshot(
        artifact_root,
        {
            "schema_version": 2,
            "session_id": "arena-newer",
            "session_type": "arena",
            "uuid": "newer-uuid",
            "updated_at": "2026-06-30T10:00:00Z",
            "summary": "Arena comparison",
            "status": "judged",
        },
    )

    records = list_session_records(artifact_root)
    loaded_legacy = load_session_record(artifact_root, "legacy-uuid")

    assert [record["session_id"] for record in records] == [
        "arena-newer",
        "legacy-uuid",
    ]
    assert records[0]["schema_version"] == 2
    assert records[1]["schema_version"] == 1
    assert loaded_legacy is not None
    assert loaded_legacy["snapshot"]["summary"] == "Legacy diagnosis"
    assert legacy_path.exists()
    assert json.loads(legacy_path.read_text(encoding="utf-8")) == legacy


def test_export_judged_arenas_writes_provider_neutral_jsonl(tmp_path):
    artifact_root = tmp_path / "artifacts"
    save_session_snapshot(
        artifact_root,
        {
            "schema_version": 2,
            "session_id": "arena-judged",
            "session_type": "arena",
            "uuid": "sample-uuid",
            "updated_at": "2026-06-30T12:00:00Z",
            "status": "judged",
            "evidence": [
                {
                    "kind": "sunbeam-output",
                    "path": "generated/sunbeam/output.log",
                    "line": 2,
                    "excerpt": "wait timed out",
                }
            ],
            "contenders": [
                {
                    "contender_id": "A",
                    "model": "model/a",
                    "report": {
                        "summary": "A summary",
                        "root_cause": "A cause",
                        "confidence": "supported",
                    },
                    "trace_path": ".sunbeam-triage/sessions/arena-judged.json",
                },
                {
                    "contender_id": "B",
                    "model": "model/b",
                    "report": {
                        "summary": "B summary",
                        "root_cause": "B cause",
                        "confidence": "supported",
                    },
                    "trace_path": ".sunbeam-triage/sessions/arena-judged.json",
                },
            ],
            "verdict": {
                "winner": "B",
                "notes": "B found the first meaningful error.",
                "rubric": {
                    "A": {
                        "root_cause": 2,
                        "evidence": 2,
                        "timeline": 1,
                        "uncertainty": 2,
                        "next_steps": 2,
                    },
                    "B": {
                        "root_cause": 5,
                        "evidence": 5,
                        "timeline": 4,
                        "uncertainty": 4,
                        "next_steps": 5,
                    },
                },
            },
        },
    )
    save_session_snapshot(
        artifact_root,
        {
            "schema_version": 2,
            "session_id": "arena-unjudged",
            "session_type": "arena",
            "uuid": "other-uuid",
            "updated_at": "2026-06-30T12:05:00Z",
            "status": "completed",
            "contenders": [],
        },
    )
    output = tmp_path / "arena-export.jsonl"

    count = export_judged_arenas(artifact_root, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert count == 1
    assert len(lines) == 1
    exported = json.loads(lines[0])
    assert exported["schema"] == "sunbeam-triage-arena-eval-v1"
    assert exported["uuid"] == "sample-uuid"
    assert exported["winner"] == "B"
    assert exported["rubric"]["B"]["root_cause"] == 5
    assert exported["contenders"][0]["model"] == "model/a"
    assert "exchanges" not in exported["contenders"][0]
    assert exported["evidence"][0]["excerpt"] == "wait timed out"
