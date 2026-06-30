from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

BudgetName = Literal["quick", "default", "hard"]


@dataclass(frozen=True)
class BudgetProfile:
    quick_max_rounds: int = 5
    default_max_rounds: int = 12
    hard_max_rounds: int = 20
    stall_limit: int = 3
    min_evidence_items: int = 2
    max_tool_result_chars: int = 60_000


@dataclass(frozen=True)
class TriageLoopOptions:
    max_rounds: int
    hard_max_rounds: int
    stall_limit: int = 3
    min_evidence_items: int = 2
    max_tool_result_chars: int = 60_000


@dataclass(frozen=True)
class ToolObservation:
    tool_name: str
    args_key: str
    result_key: str
    evidence_keys: tuple[str, ...] = ()
    timestamp_count: int = 0
    truncated: bool = False
    error: str = ""
    missing_evidence: tuple[str, ...] = ()


@dataclass
class InvestigationState:
    options: TriageLoopOptions
    phase: str = "artifact_discovery"
    rounds_used: int = 0
    evidence_found: list[dict[str, str]] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    alternatives_considered: list[dict[str, str]] = field(default_factory=list)
    failure_timeline: list[dict[str, str]] = field(default_factory=list)
    current_hypothesis: str = ""
    confidence: str = "unknown"
    next_action: str = ""
    stop_reason: str = ""
    stall_count: int = 0
    _seen_observations: set[str] = field(default_factory=set, init=False, repr=False)
    _seen_evidence: set[str] = field(default_factory=set, init=False, repr=False)

    def record_round(self) -> None:
        self.rounds_used += 1

    def apply_observation(self, observation: ToolObservation) -> None:
        made_progress = False
        observation_key = (
            f"{observation.tool_name}:{observation.args_key}:{observation.result_key}"
        )
        if observation_key not in self._seen_observations:
            self._seen_observations.add(observation_key)

        for evidence_key in observation.evidence_keys:
            if evidence_key in self._seen_evidence:
                continue
            self._seen_evidence.add(evidence_key)
            made_progress = True
            self.evidence_found.append(_evidence_item_from_key(evidence_key))

        for missing in observation.missing_evidence:
            if missing and missing not in self.missing_evidence:
                self.missing_evidence.append(missing)

        if observation.timestamp_count and made_progress:
            self.failure_timeline.append({
                "timestamp": "observed",
                "source": observation.tool_name,
                "location": "",
                "event": "Tool result included timestamped evidence.",
            })

        if made_progress:
            self.stall_count = 0
        else:
            self.stall_count += 1

        if self.evidence_found:
            self.phase = "focused_investigation"
        elif observation.tool_name.startswith("search_"):
            self.phase = "coarse_scan"

    def should_finalize(self) -> bool:
        if self.stop_reason:
            return True
        if self._has_sufficient_evidence():
            self.stop_reason = "sufficient_evidence"
            self.phase = "finalisation"
            return True
        if self.stall_count >= self.options.stall_limit:
            self.stop_reason = "stalled"
            self.phase = "finalisation"
            return True
        if self.rounds_used >= self.options.max_rounds:
            self.stop_reason = "budget_exhausted"
            self.phase = "finalisation"
            return True
        return False

    def to_prompt_summary(self) -> str:
        return json.dumps(
            {
                "phase": self.phase,
                "rounds_used": self.rounds_used,
                "stop_reason": self.stop_reason,
                "evidence_found": self.evidence_found[-8:],
                "failure_timeline": self.failure_timeline[-8:],
                "missing_evidence": self.missing_evidence[-8:],
                "alternatives_considered": self.alternatives_considered[-8:],
                "stall_count": self.stall_count,
                "confidence": self.confidence,
                "next_action": self.next_action,
            },
            sort_keys=True,
        )

    def _has_sufficient_evidence(self) -> bool:
        if len(self.evidence_found) < self.options.min_evidence_items:
            return False
        return bool(self.alternatives_considered or self.missing_evidence)


def resolve_triage_budget(
    profile: BudgetProfile,
    *,
    budget: BudgetName = "default",
    max_tool_rounds: int | None = None,
) -> TriageLoopOptions:
    if budget == "quick":
        rounds = profile.quick_max_rounds
    elif budget == "hard":
        rounds = profile.hard_max_rounds
    else:
        rounds = profile.default_max_rounds
    if max_tool_rounds is not None:
        rounds = max_tool_rounds
    if rounds > profile.hard_max_rounds:
        raise ValueError(
            f"max_tool_rounds={rounds} exceeds hard max {profile.hard_max_rounds}"
        )
    return TriageLoopOptions(
        max_rounds=rounds,
        hard_max_rounds=profile.hard_max_rounds,
        stall_limit=profile.stall_limit,
        min_evidence_items=profile.min_evidence_items,
        max_tool_result_chars=profile.max_tool_result_chars,
    )


def parse_budget_name(value: str) -> BudgetName:
    if value in {"quick", "default", "hard"}:
        return cast("BudgetName", value)
    raise ValueError(f"Unknown triage budget profile: {value}")


def observe_tool_result(
    tool_name: str,
    arguments: dict[str, Any],
    content: str,
) -> ToolObservation:
    result = _loads_object(content)
    evidence_keys = tuple(_evidence_keys(tool_name, arguments, result, content))
    return ToolObservation(
        tool_name=tool_name,
        args_key=_stable_hash(arguments),
        result_key=_stable_hash(result or content),
        evidence_keys=evidence_keys,
        timestamp_count=len(_TIMESTAMP_RE.findall(content)),
        truncated=bool(result.get("tool_result_truncated_by_budget")),
        error=str(result.get("error", "")),
        missing_evidence=tuple(_missing_evidence(tool_name, arguments, result)),
    )


def _evidence_keys(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    content: str,
) -> list[str]:
    keys: list[str] = []
    if _result_is_non_evidence(result):
        return keys
    matches = result.get("matches")
    if isinstance(matches, list):
        for item in matches:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", ""))
            line = item.get("line")
            excerpt = str(item.get("excerpt", ""))
            if path or excerpt:
                keys.append(_pack_evidence(path, line, excerpt))
    if isinstance(result.get("content"), str):
        path = str(
            result.get("path")
            or arguments.get("path")
            or arguments.get("archive_path")
            or tool_name
        )
        line = result.get("line_start") or arguments.get("line_start")
        excerpt = _first_signal_line(str(result.get("content", "")))
        if excerpt:
            keys.append(_pack_evidence(path, line, excerpt))
    if not keys and tool_name in {
        "get_artifact_file",
        "search_artifacts",
        "search_sosreport",
        "get_sosreport_file",
    }:
        path = str(arguments.get("path") or arguments.get("archive_path") or tool_name)
        excerpt = _first_signal_line(content)
        if excerpt:
            keys.append(_pack_evidence(path, arguments.get("line_start"), excerpt))
    return keys


def _missing_evidence(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    if result.get("tool_result_truncated_by_budget"):
        detail = str(result.get("error") or "Tool result was truncated by budget.")
        missing.append(detail)
    if result.get("duplicate_tool_call"):
        missing.append("Duplicate tool call skipped; earlier result should be reused.")
    if result.get("round_tool_limit_reached"):
        missing.append(
            "Tool call skipped because the per-round tool limit was reached."
        )
    if (
        tool_name in {"search_artifacts", "search_sosreport"}
        and result.get("ok") is True
        and result.get("matches") == []
    ):
        pattern = str(arguments.get("pattern", ""))
        target = str(
            arguments.get("path_prefix") or arguments.get("archive_path") or ""
        )
        missing.append(f"No matches found for targeted search {pattern!r} in {target}.")
    return missing


def _result_is_non_evidence(result: dict[str, Any]) -> bool:
    return bool(
        result.get("tool_result_truncated_by_budget")
        or result.get("duplicate_tool_call")
        or result.get("round_tool_limit_reached")
        or result.get("ok") is False
    )


def _pack_evidence(path: str, line: Any, excerpt: str) -> str:
    return json.dumps(
        {
            "source": path,
            "location": "" if line is None else str(line),
            "claim": excerpt,
        },
        sort_keys=True,
    )


def _evidence_item_from_key(key: str) -> dict[str, str]:
    item = _loads_object(key)
    return {
        "source": str(item.get("source", "")),
        "location": str(item.get("location", "")),
        "claim": str(item.get("claim", "")),
    }


def _first_signal_line(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:300]
    return ""


def _loads_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")
