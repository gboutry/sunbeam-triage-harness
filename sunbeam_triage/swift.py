from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit

from .config import SwiftConfig
from .http import UrlLibHttp


@dataclass(frozen=True)
class SwiftObject:
    name: str
    hash: str | None
    bytes: int


@dataclass(frozen=True)
class SwiftDownloadFailure:
    name: str
    path: str
    url: str
    error: str


@dataclass(frozen=True)
class MirrorManifest:
    uuid: str
    root: Path
    objects: tuple[SwiftObject, ...]
    failures: tuple[SwiftDownloadFailure, ...] = ()


class SwiftMirror:
    def __init__(self, swift_config: SwiftConfig, artifact_root: Path, http=None):
        self.swift_config = swift_config
        self.artifact_root = Path(artifact_root)
        self.http = http or UrlLibHttp(timeout=swift_config.timeout_seconds)

    def mirror_uuid(
        self,
        uuid: str,
        *,
        refresh: bool = False,
        progress: Callable[[dict], None] | None = None,
        continue_on_error: bool = False,
    ) -> MirrorManifest:
        try:
            listing = self._list(uuid)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"No Swift artifacts found for {uuid}: {exc}") from exc
        if not listing:
            raise RuntimeError(f"No Swift artifacts found for {uuid}")

        root = self.artifact_root / uuid
        objects: list[SwiftObject] = []
        for item in listing:
            name = item["name"]
            if not name.startswith(f"{uuid}/"):
                continue
            rel = name[len(uuid) + 1 :]
            if not rel or rel.endswith("/"):
                continue
            obj = SwiftObject(
                name=name,
                hash=item.get("hash"),
                bytes=int(item.get("bytes", 0)),
            )
            objects.append(obj)

        total = len(objects)
        failures: list[SwiftDownloadFailure] = []
        for index, obj in enumerate(objects, start=1):
            name = obj.name
            rel = name[len(uuid) + 1 :]
            path = root / rel
            url = self._object_url(name)
            if not refresh and self._is_unchanged(path, obj):
                _emit_progress(
                    progress,
                    index=index,
                    total=total,
                    name=name,
                    path=path,
                    url=url,
                    status="cached",
                )
                continue
            _emit_progress(
                progress,
                index=index,
                total=total,
                name=name,
                path=path,
                url=url,
                status="downloading",
            )
            try:
                self.http.download(url, path)
            except (HTTPError, URLError, OSError) as exc:
                error = str(exc)
                failure = SwiftDownloadFailure(
                    name=name,
                    path=str(path),
                    url=url,
                    error=error,
                )
                failures.append(failure)
                _emit_progress(
                    progress,
                    index=index,
                    total=total,
                    name=name,
                    path=path,
                    url=url,
                    status="failed",
                    error=error,
                )
                if continue_on_error:
                    continue
                raise RuntimeError(
                    "Failed to download Swift object "
                    f"{name} to {path} from {url}: {exc}"
                ) from exc
            _emit_progress(
                progress,
                index=index,
                total=total,
                name=name,
                path=path,
                url=url,
                status="downloaded",
            )

        manifest_path = root / ".sunbeam-triage-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps([obj.__dict__ for obj in objects], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return MirrorManifest(
            uuid=uuid,
            root=root,
            objects=tuple(objects),
            failures=tuple(failures),
        )

    def _list(self, uuid: str) -> list[dict]:
        url = f"{self.swift_config.base_url}/{quote(uuid, safe='')}/index.html"
        html = self.http.get_text(url)
        return [
            {
                "name": f"{uuid}/{rel}",
                "hash": None,
                "bytes": -1,
            }
            for rel in _parse_index_links(html)
        ]

    def _object_url(self, name: str) -> str:
        return f"{self.swift_config.base_url}/{quote(name, safe='/')}"

    @staticmethod
    def _is_unchanged(path: Path, obj: SwiftObject) -> bool:
        if obj.bytes < 0:
            return path.exists()
        if not path.exists() or path.stat().st_size != obj.bytes:
            return False
        if not obj.hash:
            return True
        digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
        return digest == obj.hash


def _emit_progress(
    progress: Callable[[dict], None] | None,
    *,
    index: int,
    total: int,
    name: str,
    path: Path,
    url: str,
    status: str,
    error: str | None = None,
) -> None:
    if progress is None:
        return
    event = {
        "index": index,
        "total": total,
        "name": name,
        "path": str(path),
        "url": url,
        "status": status,
    }
    if error is not None:
        event["error"] = error
    progress(event)


class _IndexLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def _parse_index_links(html: str) -> list[str]:
    parser = _IndexLinkParser()
    parser.feed(html)
    links: list[str] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or href.startswith("/") or parts.query:
            continue
        rel = unquote(parts.path)
        if not rel or rel == "index.html" or rel.endswith("/") or rel.startswith("../"):
            continue
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            continue
        normalized = path.as_posix()
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links
