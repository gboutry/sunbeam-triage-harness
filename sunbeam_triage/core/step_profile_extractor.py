from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .step_profiles import ProbeSpec, StepProfile, TargetedReadSpec

PATH_CELL = re.compile(r"^\|\s*`([^`]+)`\s*\|")
PATTERN_HEADING = re.compile(r"^### Pattern \d+:\s*(?P<name>.+)", re.MULTILINE)
TIME_LITERAL = re.compile(
    r"^(?:\d{1,2}:\d{2}(?::\d{1,2})?)(?:\\\|(?:\d{1,2}:\d{2}(?::\d{1,2})?))*$"
)
PLACEHOLDER_ONLY = re.compile(r"^<[^>]+>(?:\.\*FAILED)?$")
CASE_HOSTS = ("chespin", "behaim", "anonster", "crustle", "ledian.maas")
NAVIGATION_PATTERNS = {
    "vault",
    "FAILED",
    "ubuntu@",
    '"ubuntu@"',
    "maas-api",
    "post-refresh",
    "exit status",
}
ARCHIVE_MEMBER_SELECTORS = {
    "var/log/syslog$",
    "var/log/juju/unit-openstack-hypervisor-3.log$",
}


def extract_profiles(steps_dir: Path) -> dict[str, StepProfile]:
    profiles: dict[str, StepProfile] = {}
    root = Path(steps_dir)
    for path in sorted(root.glob("sunbeam_*.md")):
        name = path.stem
        text = path.read_text(encoding="utf-8")
        artifacts = _artifact_paths(text)
        specs = _grep_specs(name, text, artifacts)
        profiles[name] = StepProfile(
            name=name,
            primary_artifacts=tuple(artifacts),
            probes=tuple(
                probe for probe, _targeted_read in specs if probe is not None
            ),
            targeted_reads=tuple(
                targeted_read
                for _probe, targeted_read in specs
                if targeted_read is not None
            ),
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
                "targeted_reads": [
                    {
                        "id": read.id,
                        "artifact": read.artifact,
                        "selector": read.selector,
                        "reason": read.reason,
                        "source_line": read.source_line,
                        "source_text": read.source_text,
                        "read_class": read.read_class,
                    }
                    for read in profile.targeted_reads
                ],
                "probes": [
                    {
                        "id": probe.id,
                        "artifact": probe.artifact,
                        "pattern": probe.pattern,
                        "category": probe.category,
                        "read_class": probe.read_class,
                        "source_line": probe.source_line,
                        "source_text": probe.source_text,
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


def _grep_specs(
    name: str,
    text: str,
    artifacts: list[str],
) -> list[tuple[ProbeSpec | None, TargetedReadSpec | None]]:
    specs: list[tuple[ProbeSpec | None, TargetedReadSpec | None]] = []
    fallback = artifacts[0] if artifacts else ""
    for index, (line_number, line) in enumerate(_grep_lines(text), start=1):
        pattern = _grep_pattern(line)
        if not pattern:
            continue
        artifact = _first_artifact_in_line(line, artifacts) or fallback
        action = _sanitize_pattern(pattern)
        if action["kind"] == "drop":
            continue
        if action["kind"] == "targeted_read":
            specs.append((
                None,
                TargetedReadSpec(
                    id=f"{name}.targeted_read_{index}",
                    artifact=artifact,
                    selector=action["pattern"],
                    reason=action["reason"],
                    source_line=line_number,
                    source_text=line,
                ),
            ))
            continue
        specs.append((
            ProbeSpec(
                id=f"{name}.grep_{index}",
                artifact=artifact,
                pattern=action["pattern"],
                category=action["category"],
                read_class=action["read_class"],
                source_line=line_number,
                source_text=line,
            ),
            None,
        ))
    return specs


def _grep_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    in_code_block = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block and "grep" in stripped:
            lines.append((line_number, stripped))
    return lines


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


def _sanitize_pattern(pattern: str) -> dict[str, str]:
    if pattern == (
        "ledian.maas\\|wait timed out after 1799\\|openstack-hypervisor/3\\|"
        "cinder-volume/3\\|microceph/3"
    ):
        return {
            "kind": "probe",
            "pattern": (
                "wait timed out after 1799\\|openstack-hypervisor/\\d+\\|"
                "cinder-volume/\\d+\\|microceph/\\d+"
            ),
            "category": "step_profile",
            "read_class": "targeted_search",
        }
    if _drop_pattern(pattern):
        return {"kind": "drop"}
    if pattern in ARCHIVE_MEMBER_SELECTORS:
        return {
            "kind": "targeted_read",
            "pattern": pattern,
            "reason": "archive member selector",
        }
    if pattern in NAVIGATION_PATTERNS:
        return {
            "kind": "probe",
            "pattern": pattern,
            "category": "navigation",
            "read_class": "broad_search",
        }
    return {
        "kind": "probe",
        "pattern": pattern,
        "category": "step_profile",
        "read_class": "targeted_search",
    }


def _drop_pattern(pattern: str) -> bool:
    if pattern == "for":
        return True
    if TIME_LITERAL.fullmatch(pattern):
        return True
    if PLACEHOLDER_ONLY.fullmatch(pattern):
        return True
    if "<failing_node" in pattern or "<configure-start-minute>" in pattern:
        return True
    return any(host in pattern for host in CASE_HOSTS)
