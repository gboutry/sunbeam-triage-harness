import json
from pathlib import Path
from urllib.error import HTTPError

from sunbeam_triage.config import Config
from sunbeam_triage.swift import SwiftMirror


class FakeHttp:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get_text(self, url):
        self.urls.append(url)
        value = self.responses[url]
        if isinstance(value, Exception):
            raise value
        return value

    def download(self, url, path):
        self.urls.append(url)
        value = self.responses[url]
        if isinstance(value, Exception):
            raise value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)


def test_swift_mirror_downloads_all_objects_and_skips_unchanged(tmp_path):
    uuid = "abc-123"
    base = "https://swift.example/v1/AUTH/container"
    listing_url = f"{base}/?prefix={uuid}/&format=json"
    output_url = f"{base}/{uuid}/generated/sunbeam/output.log"
    jobs_url = f"{base}/{uuid}/generated/github-runner/jobs.json"
    listing = [
        {
            "name": f"{uuid}/generated/sunbeam/output.log",
            "hash": "5d41402abc4b2a76b9719d911017c592",
            "bytes": 5,
        },
        {
            "name": f"{uuid}/generated/github-runner/jobs.json",
            "hash": "7d793037a0760186574b0282f2f435e7",
            "bytes": 5,
        },
    ]
    http = FakeHttp(
        {
            listing_url: json.dumps(listing),
            output_url: b"hello",
            jobs_url: b"world",
        }
    )
    config = Config.load(None)
    config.swift.base_url = base
    config.paths.artifact_root = tmp_path / "artifacts"

    mirror = SwiftMirror(config.swift, config.paths.artifact_root, http=http)
    manifest = mirror.mirror_uuid(uuid)
    manifest_again = mirror.mirror_uuid(uuid)

    assert (tmp_path / "artifacts" / uuid / "generated/sunbeam/output.log").read_text() == "hello"
    assert (tmp_path / "artifacts" / uuid / "generated/github-runner/jobs.json").read_text() == "world"
    assert len(manifest.objects) == 2
    assert len(manifest_again.objects) == 2
    assert http.urls.count(output_url) == 1
    assert http.urls.count(jobs_url) == 1


def test_swift_mirror_reports_download_progress(tmp_path):
    uuid = "abc-123"
    base = "https://swift.example/v1/AUTH/container"
    listing_url = f"{base}/?prefix={uuid}/&format=json"
    output_url = f"{base}/{uuid}/generated/sunbeam/output.log"
    listing = [
        {
            "name": f"{uuid}/generated/sunbeam/output.log",
            "hash": "5d41402abc4b2a76b9719d911017c592",
            "bytes": 5,
        },
    ]
    http = FakeHttp({listing_url: json.dumps(listing), output_url: b"hello"})
    config = Config.load(None)
    config.swift.base_url = base
    config.paths.artifact_root = tmp_path / "artifacts"
    events = []

    SwiftMirror(config.swift, config.paths.artifact_root, http=http).mirror_uuid(
        uuid,
        progress=events.append,
    )

    assert events == [
        {
            "index": 1,
            "total": 1,
            "name": f"{uuid}/generated/sunbeam/output.log",
            "path": str(
                tmp_path / "artifacts" / uuid / "generated/sunbeam/output.log"
            ),
            "url": output_url,
            "status": "downloading",
        },
        {
            "index": 1,
            "total": 1,
            "name": f"{uuid}/generated/sunbeam/output.log",
            "path": str(
                tmp_path / "artifacts" / uuid / "generated/sunbeam/output.log"
            ),
            "url": output_url,
            "status": "downloaded",
        },
    ]


def test_swift_mirror_download_error_names_object_url_and_path(tmp_path):
    uuid = "abc-123"
    base = "https://swift.example/v1/AUTH/container"
    listing_url = f"{base}/?prefix={uuid}/&format=json"
    output_url = f"{base}/{uuid}/generated/sunbeam/output.log"
    listing = [
        {
            "name": f"{uuid}/generated/sunbeam/output.log",
            "hash": "5d41402abc4b2a76b9719d911017c592",
            "bytes": 5,
        },
    ]
    http = FakeHttp(
        {
            listing_url: json.dumps(listing),
            output_url: HTTPError(output_url, 404, "Not Found", {}, None),
        }
    )
    config = Config.load(None)
    config.swift.base_url = base
    config.paths.artifact_root = tmp_path / "artifacts"

    try:
        SwiftMirror(config.swift, config.paths.artifact_root, http=http).mirror_uuid(uuid)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert f"Failed to download Swift object {uuid}/generated/sunbeam/output.log" in message
    assert output_url in message
    assert str(tmp_path / "artifacts" / uuid / "generated/sunbeam/output.log") in message
    assert "HTTP Error 404: Not Found" in message


def test_swift_mirror_can_continue_after_download_errors(tmp_path):
    uuid = "abc-123"
    base = "https://swift.example/v1/AUTH/container"
    listing_url = f"{base}/?prefix={uuid}/&format=json"
    missing_url = f"{base}/{uuid}/generated/sunbeam/missing.log"
    jobs_url = f"{base}/{uuid}/generated/github-runner/jobs.json"
    listing = [
        {
            "name": f"{uuid}/generated/sunbeam/missing.log",
            "hash": "5d41402abc4b2a76b9719d911017c592",
            "bytes": 5,
        },
        {
            "name": f"{uuid}/generated/github-runner/jobs.json",
            "hash": "7d793037a0760186574b0282f2f435e7",
            "bytes": 5,
        },
    ]
    http = FakeHttp(
        {
            listing_url: json.dumps(listing),
            missing_url: HTTPError(missing_url, 404, "Not Found", {}, None),
            jobs_url: b"world",
        }
    )
    config = Config.load(None)
    config.swift.base_url = base
    config.paths.artifact_root = tmp_path / "artifacts"
    events = []

    manifest = SwiftMirror(
        config.swift,
        config.paths.artifact_root,
        http=http,
    ).mirror_uuid(uuid, continue_on_error=True, progress=events.append)

    assert (tmp_path / "artifacts" / uuid / "generated/github-runner/jobs.json").read_text() == "world"
    assert [failure.name for failure in manifest.failures] == [
        f"{uuid}/generated/sunbeam/missing.log"
    ]
    assert manifest.failures[0].url == missing_url
    assert manifest.failures[0].path == str(
        tmp_path / "artifacts" / uuid / "generated/sunbeam/missing.log"
    )
    assert "HTTP Error 404: Not Found" in manifest.failures[0].error
    assert events[1]["status"] == "failed"
    assert events[1]["error"] == "HTTP Error 404: Not Found"
    assert jobs_url in http.urls


def test_swift_mirror_raises_clear_error_for_missing_uuid(tmp_path):
    uuid = "missing"
    base = "https://swift.example/v1/AUTH/container"
    listing_url = f"{base}/?prefix={uuid}/&format=json"
    http = FakeHttp(
        {
            listing_url: HTTPError(listing_url, 404, "Not Found", {}, None),
        }
    )

    mirror = SwiftMirror(Config.load(None).swift, tmp_path, http=http)
    mirror.swift_config.base_url = base

    try:
        mirror.mirror_uuid(uuid)
    except RuntimeError as exc:
        assert "No Swift artifacts found" in str(exc) or "404" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
