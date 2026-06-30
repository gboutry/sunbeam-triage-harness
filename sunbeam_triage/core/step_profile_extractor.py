from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .step_profiles import ProbeSpec, StepProfile

PATH_CELL = re.compile(r"^\|\s*`([^`]+)`\s*\|")
PATTERN_HEADING = re.compile(r"^### Pattern \d+:\s*(?P<name>.+)", re.MULTILINE)


def extract_profiles(steps_dir: Path) -> dict[str, StepProfile]:
    profiles: dict[str, StepProfile] = {}
    root = Path(steps_dir)
    for path in sorted(root.glob("sunbeam_*.md")):
        name = path.stem
        text = path.read_text(encoding="utf-8")
        artifacts = _artifact_paths(text)
        profiles[name] = StepProfile(
            name=name,
            primary_artifacts=tuple(artifacts),
            probes=tuple(_grep_probes(name, text, artifacts)),
            known_patterns=tuple(
                match.group("name").strip()
                for match in PATTERN_HEADING.finditer(text)
            ),
            source_path=str(path.relative_to(root)),
        )
    return profiles


def profiles_to_dict(profiles: dict[str, StepProfile]) -> dict[str, Any]:
    return {
        "profiles": {
            name: {
                "name": profile.name,
                "source_path": profile.source_path,
                "primary_artifacts": list(profile.primary_artifacts),
                "archive_prefixes": list(profile.archive_prefixes),
                "known_patterns": list(profile.known_patterns),
                "probes": [
                    {
                        "id": probe.id,
                        "artifact": probe.artifact,
                        "pattern": probe.pattern,
                        "category": probe.category,
                        "read_class": probe.read_class,
                    }
                    for probe in profile.probes
                ],
            }
            for name, profile in sorted(profiles.items())
        }
    }


def _artifact_paths(text: str) -> list[str]:
    paths: list[str] = []
    for line in text.splitlines():
        match = PATH_CELL.match(line.strip())
        if not match:
            continue
        path = match.group(1)
        if path not in paths:
            paths.append(path)
    return paths


def _grep_probes(name: str, text: str, artifacts: list[str]) -> list[ProbeSpec]:
    probes: list[ProbeSpec] = []
    fallback = artifacts[0] if artifacts else ""
    for index, line in enumerate(_grep_lines(text), start=1):
        pattern = _grep_pattern(line)
        if not pattern:
            continue
        probes.append(
            ProbeSpec(
                id=f"{name}.grep_{index}",
                artifact=_first_artifact_in_line(line, artifacts) or fallback,
                pattern=pattern,
            )
        )
    return probes


def _grep_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if "grep" in line]


def _grep_pattern(line: str) -> str:
    grep_index = line.find("grep")
    if grep_index < 0:
        return ""
    command = line[grep_index:].strip()
    try:
        words = shlex.split(command)
    except ValueError:
        return ""
    if not words or words[0] != "grep":
        return ""
    index = 1
    while index < len(words):
        word = words[index]
        if word == "--":
            index += 1
            break
        if not word.startswith("-") or word == "-":
            break
        index += 1
        if word in {"-A", "-B", "-C", "-e", "-f", "--after-context", "--before-context", "--context", "--regexp", "--file"}:
            if word in {"-e", "--regexp"}:
                break
            index += 1
    if index >= len(words):
        return ""
    return words[index]


def _first_artifact_in_line(line: str, artifacts: list[str]) -> str:
    for artifact in artifacts:
        if artifact in line:
            return artifact
    return ""
