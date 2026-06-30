from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class ProbeSpec:
    id: str
    artifact: str
    pattern: str
    category: str = "step_profile"
    read_class: str = "targeted_search"
    source_line: int | None = None
    source_text: str = ""


@dataclass(frozen=True)
class TargetedReadSpec:
    id: str
    artifact: str
    selector: str
    reason: str
    source_line: int | None = None
    source_text: str = ""
    read_class: str = "targeted_read"


@dataclass(frozen=True)
class StepProfile:
    name: str
    primary_artifacts: tuple[str, ...] = ()
    probes: tuple[ProbeSpec, ...] = ()
    targeted_reads: tuple[TargetedReadSpec, ...] = ()
    known_patterns: tuple[str, ...] = ()
    archive_prefixes: tuple[str, ...] = ()
    source_path: str = ""


def profile_for_step(name: str) -> StepProfile | None:
    return STEP_PROFILES.get(name)


def load_step_profiles() -> dict[str, StepProfile]:
    try:
        text = (
            resources.files("sunbeam_triage.data")
            .joinpath("sunbeam_step_profiles.json")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return {}
    data = json.loads(text)
    return {
        name: _profile_from_dict(profile_data)
        for name, profile_data in sorted(data["profiles"].items())
    }


def _profile_from_dict(data: dict[str, Any]) -> StepProfile:
    return StepProfile(
        name=data["name"],
        primary_artifacts=tuple(data.get("primary_artifacts", ())),
        probes=tuple(_probe_from_dict(probe) for probe in data.get("probes", ())),
        targeted_reads=tuple(
            _targeted_read_from_dict(read)
            for read in data.get("targeted_reads", ())
        ),
        known_patterns=tuple(data.get("known_patterns", ())),
        archive_prefixes=tuple(data.get("archive_prefixes", ())),
        source_path=data.get("source_path", ""),
    )


def _probe_from_dict(data: dict[str, Any]) -> ProbeSpec:
    return ProbeSpec(
        id=data["id"],
        artifact=data["artifact"],
        pattern=data["pattern"],
        category=data.get("category", "step_profile"),
        read_class=data.get("read_class", "targeted_search"),
        source_line=data.get("source_line"),
        source_text=data.get("source_text", ""),
    )


def _targeted_read_from_dict(data: dict[str, Any]) -> TargetedReadSpec:
    return TargetedReadSpec(
        id=data["id"],
        artifact=data["artifact"],
        selector=data["selector"],
        reason=data["reason"],
        source_line=data.get("source_line"),
        source_text=data.get("source_text", ""),
        read_class=data.get("read_class", "targeted_read"),
    )


STEP_PROFILES: dict[str, StepProfile] = load_step_profiles()
