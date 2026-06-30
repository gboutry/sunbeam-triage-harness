import json
from pathlib import Path
from urllib.error import HTTPError

from sunbeam_triage.core.config import Config
from sunbeam_triage.core.swift import SwiftMirror


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
    index_url = f"{base}/{uuid}/index.html"
    output_url = f"{base}/{uuid}/generated/sunbeam/output.log"
    jobs_url = f"{base}/{uuid}/generated/github-runner/jobs.json"
    index = "\n".join(
        [
            '<a href="generated/sunbeam/output.log">generated/sunbeam/output.log</a><br>',
            '<a href="generated/github-runner/jobs.json">generated/github-runner/jobs.json</a><br>',
        ]
    )
    http = FakeHttp(
        {
            index_url: index,
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
    manifest_json = json.loads(
        (tmp_path / "artifacts" / uuid / ".sunbeam-triage-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest_json == [
        {
            "bytes": -1,
            "hash": None,
            "name": f"{uuid}/generated/sunbeam/output.log",
        },
        {
            "bytes": -1,
            "hash": None,
            "name": f"{uuid}/generated/github-runner/jobs.json",
        },
    ]
    assert http.urls.count(output_url) == 1
    assert http.urls.count(jobs_url) == 1


def test_swift_mirror_reports_download_progress(tmp_path):
    uuid = "abc-123"
    base = "https://swift.example/v1/AUTH/container"
    index_url = f"{base}/{uuid}/index.html"
    output_url = f"{base}/{uuid}/generated/sunbeam/output.log"
    index = '<a href="generated/sunbeam/output.log">generated/sunbeam/output.log</a><br>'
    http = FakeHttp({index_url: index, output_url: b"hello"})
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
    index_url = f"{base}/{uuid}/index.html"
    output_url = f"{base}/{uuid}/generated/sunbeam/output.log"
    index = '<a href="generated/sunbeam/output.log">generated/sunbeam/output.log</a><br>'
    http = FakeHttp(
        {
            index_url: index,
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
    index_url = f"{base}/{uuid}/index.html"
    missing_url = f"{base}/{uuid}/generated/sunbeam/missing.log"
    jobs_url = f"{base}/{uuid}/generated/github-runner/jobs.json"
    index = "\n".join(
        [
            '<a href="generated/sunbeam/missing.log">generated/sunbeam/missing.log</a><br>',
            '<a href="generated/github-runner/jobs.json">generated/github-runner/jobs.json</a><br>',
        ]
    )
    http = FakeHttp(
        {
            index_url: index,
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
    index_url = f"{base}/{uuid}/index.html"
    http = FakeHttp(
        {
            index_url: HTTPError(index_url, 404, "Not Found", {}, None),
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


def test_swift_mirror_uses_index_html_links_and_ignores_unsafe_links(tmp_path):
    uuid = "abc-123"
    base = "https://swift.example/v1/AUTH/container"
    index_url = f"{base}/{uuid}/index.html"
    output_url = f"{base}/{uuid}/generated/sunbeam/output%20one.log"
    index = "\n".join(
        [
            '<a href="generated/sunbeam/output%20one.log">output</a><br>',
            '<a href="index.html">index.html</a><br>',
            '<a href="../other/file.txt">bad</a><br>',
            '<a href="/absolute/file.txt">bad</a><br>',
            '<a href="https://example.com/file.txt">bad</a><br>',
        ]
    )
    http = FakeHttp({index_url: index, output_url: b"hello"})
    config = Config.load(None)
    config.swift.base_url = base
    config.paths.artifact_root = tmp_path / "artifacts"

    manifest = SwiftMirror(config.swift, config.paths.artifact_root, http=http).mirror_uuid(uuid)

    assert [obj.name for obj in manifest.objects] == [
        f"{uuid}/generated/sunbeam/output one.log"
    ]
    assert (tmp_path / "artifacts" / uuid / "generated/sunbeam/output one.log").read_text() == "hello"
    assert f"{base}/?prefix={uuid}/&format=json" not in http.urls
