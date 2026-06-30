from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any


class UrlLibHttp:
    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def get_text(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "sunbeam-triage/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8")

    def download(self, url: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "sunbeam-triage/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            with path.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)

    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "sunbeam-triage/0.1",
                **headers,
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
