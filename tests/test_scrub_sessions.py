import json

from sunbeam_triage.core.sessions import scrub_session_store

TOKEN = "AbCd12EfGh34IjKl56MnOpQr78St"  # noqa: S105


def test_scrub_session_store_redacts_all_stores_and_is_idempotent(tmp_path):
    artifact_root = tmp_path / "artifacts"
    canonical = artifact_root / ".sunbeam-triage" / "sessions" / "one.json"
    events = artifact_root / ".sunbeam-triage" / "events" / "one.jsonl"
    legacy = artifact_root / ".sunbeam-triage-ui" / "sessions" / "old.json"
    for path in (canonical, events, legacy):
        path.parent.mkdir(parents=True, exist_ok=True)
    command = f"sunbeam enable -m manifest.yaml pro {TOKEN}"
    canonical.write_text(json.dumps({"session_id": "one", "summary": command}))
    events.write_text(json.dumps({"event": "exchange", "content": command}) + "\n")
    legacy.write_text(json.dumps({"uuid": "old", "summary": command}))

    dry_run = scrub_session_store(artifact_root, dry_run=True)

    assert dry_run.scanned == 3
    assert dry_run.changed == 3
    assert TOKEN in canonical.read_text()

    applied = scrub_session_store(artifact_root)

    assert applied.changed == 3
    assert all(TOKEN not in path.read_text() for path in (canonical, events, legacy))
    assert scrub_session_store(artifact_root).changed == 0


def test_scrub_session_store_skips_malformed_files_without_partial_rewrite(tmp_path):
    artifact_root = tmp_path / "artifacts"
    malformed = artifact_root / ".sunbeam-triage" / "events" / "bad.jsonl"
    malformed.parent.mkdir(parents=True)
    original = '{"valid": true}\nnot-json\n'
    malformed.write_text(original)

    result = scrub_session_store(artifact_root)

    assert result.scanned == 1
    assert result.changed == 0
    assert result.unchanged == 0
    assert result.malformed == (str(malformed),)
    assert malformed.read_text() == original
