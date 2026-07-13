from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineSignature:
    step: str
    sku: str
    addon: str
    topology: str
    versions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineSignature:
        return cls(
            step=str(data.get("step", "")),
            sku=str(data.get("sku", "")),
            addon=str(data.get("addon", "")),
            topology=str(data.get("topology", "")),
            versions=tuple(sorted(map(str, data.get("versions", ())))),
        )


@dataclass(frozen=True)
class BaselineRun:
    uuid: str
    outcome: str
    signature: BaselineSignature
    signals: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineRun:
        return cls(
            uuid=str(data["uuid"]),
            outcome=str(data.get("outcome", "unknown")),
            signature=BaselineSignature.from_dict(data.get("signature", {})),
            signals=tuple(map(str, data.get("signals", ()))),
        )


@dataclass(frozen=True)
class SignalBaseline:
    signal: str
    successful_run_count: int
    observed_in_successes: int

    @property
    def interpretation(self) -> str:
        if self.observed_in_successes:
            return (
                "counterevidence: the signal also occurs during successful runs "
                "and is not causal by itself"
            )
        return "failure-specific candidate: comparison does not establish causality"


def load_baseline_manifest(path: Path) -> list[BaselineRun]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Baseline manifest must be a JSON array")
    return [BaselineRun.from_dict(item) for item in data]


def match_successful_baselines(
    target: BaselineSignature,
    runs: list[BaselineRun],
) -> list[BaselineRun]:
    return [
        run
        for run in runs
        if run.outcome == "success" and run.signature == target
    ]


def compare_signals(
    failure_signals: list[str] | tuple[str, ...],
    baselines: list[BaselineRun],
) -> list[SignalBaseline]:
    return [
        SignalBaseline(
            signal=signal,
            successful_run_count=len(baselines),
            observed_in_successes=sum(signal in run.signals for run in baselines),
        )
        for signal in failure_signals
    ]
