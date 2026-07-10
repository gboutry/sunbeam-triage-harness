from __future__ import annotations

import copy
import json
from typing import Any

from .llm_exchanges import tool_call_name_and_arguments
from .triage_state import ToolObservation, _extract_entities, observe_tool_result


def validate_diagnosis_report(
    data: dict[str, Any],
    observations: list[ToolObservation],
) -> dict[str, Any]:
    validated = copy.deepcopy(data)
    was_confirmed = validated.get("confidence") == "confirmed"
    strong_claim = was_confirmed or any(
        item.get("status") == "confirmed"
        for item in validated.get("candidate_mechanisms", [])
        if isinstance(item, dict)
    )

    if validated.get("needs_more_evidence") is True and was_confirmed:
        _downgrade_confirmed(validated)

    if was_confirmed and not _has_successful_targeted_evidence(observations):
        _downgrade_confirmed(validated)
        _append_unique(
            validated,
            "missing_evidence",
            (
                "Confirmed confidence requires at least one successful targeted "
                "read tied to the failure surface."
            ),
        )

    if (
        not was_confirmed
        and validated.get("confidence") == "supported"
        and not _has_successful_targeted_evidence(observations)
    ):
        _downgrade_supported(validated)
        _append_unique(
            validated,
            "missing_evidence",
            "Supported confidence requires a successful targeted artifact read.",
        )

    if validated.get("confidence") in {"confirmed", "supported"}:
        unresolved = _unresolved_report_evidence(validated, observations)
        if unresolved:
            if validated.get("confidence") == "confirmed":
                _downgrade_confirmed(validated)
            else:
                _downgrade_supported(validated)
            _append_unique(
                validated,
                "missing_evidence",
                "Report citations did not resolve to inspected artifact evidence: "
                + ", ".join(unresolved[:4]),
            )

    if strong_claim:
        _validate_named_entity_coverage(validated, observations)
        _record_failed_targeted_reads(validated, observations)

    return validated


def observations_from_exchanges(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> list[ToolObservation]:
    calls_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    observations: list[ToolObservation] = []
    for exchange in exchanges[start_index:]:
        response = exchange.get("response", {})
        if isinstance(response, dict):
            for tool_call in response.get("tool_calls", []) or []:
                if not isinstance(tool_call, dict):
                    continue
                name, arguments = tool_call_name_and_arguments(tool_call)
                call_id = str(tool_call.get("id", ""))
                calls_by_id[call_id] = (name, arguments)

        request = exchange.get("request", {})
        if not isinstance(request, dict):
            continue
        for message in request.get("messages", []) or []:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            call = calls_by_id.get(str(message.get("tool_call_id", "")))
            if call is None:
                continue
            name, arguments = call
            observations.append(
                observe_tool_result(name, arguments, str(message.get("content", "")))
            )
    return observations


def _has_successful_targeted_evidence(observations: list[ToolObservation]) -> bool:
    return any(
        observation.read_class == "targeted"
        and observation.success
        and observation.evidence_keys
        for observation in observations
    )


def _validate_named_entity_coverage(
    validated: dict[str, Any],
    observations: list[ToolObservation],
) -> None:
    successful_entities = {
        entity
        for observation in observations
        if observation.success and observation.evidence_keys
        for entity in observation.entities
    }
    claimed_entities = _claimed_entities(validated)
    for entity in sorted(claimed_entities - successful_entities):
        _downgrade_confirmed(validated)
        validated["needs_more_evidence"] = True
        _append_unique(
            validated,
            "missing_evidence",
            (
                f"Claim about {entity} is not covered by a successful "
                "artifact read in this investigation."
            ),
        )


def _record_failed_targeted_reads(
    validated: dict[str, Any],
    observations: list[ToolObservation],
) -> None:
    for observation in observations:
        if observation.read_class != "targeted" or observation.success:
            continue
        refs = ", ".join(observation.source_refs) or observation.tool_name
        detail = observation.error or "read failed"
        _append_unique(
            validated,
            "unknowns",
            f"A failed targeted read was not resolved: {refs}: {detail}",
        )


def _claimed_entities(validated: dict[str, Any]) -> set[str]:
    parts = [
        str(validated.get("summary", "")),
        str(validated.get("failure_surface", "")),
        str(validated.get("root_cause", "")),
    ]
    for item in validated.get("evidence", []) or []:
        if not isinstance(item, dict):
            continue
        parts.extend([str(item.get("path", "")), str(item.get("excerpt", ""))])
    for item in validated.get("candidate_mechanisms", []) or []:
        if not isinstance(item, dict):
            continue
        parts.extend([str(item.get("name", "")), str(item.get("rationale", ""))])
    return set(_extract_entities(" ".join(parts)))


def _downgrade_confirmed(data: dict[str, Any]) -> None:
    if data.get("confidence") == "confirmed":
        data["confidence"] = "supported"
    if data.get("triage_confidence") == "high":
        data["triage_confidence"] = "medium"
    for item in data.get("candidate_mechanisms", []) or []:
        if isinstance(item, dict) and item.get("status") == "confirmed":
            item["status"] = "supported"


def _downgrade_supported(data: dict[str, Any]) -> None:
    if data.get("confidence") == "supported":
        data["confidence"] = "speculative"
    if data.get("triage_confidence") in {"high", "medium"}:
        data["triage_confidence"] = "low"
    data["needs_more_evidence"] = True
    for item in data.get("candidate_mechanisms", []) or []:
        if isinstance(item, dict) and item.get("status") == "supported":
            item["status"] = "speculative"


def _unresolved_report_evidence(
    data: dict[str, Any],
    observations: list[ToolObservation],
) -> list[str]:
    inspected: set[tuple[str, str]] = set()
    for observation in observations:
        if not observation.success:
            continue
        for key in observation.evidence_keys:
            try:
                item = json.loads(key)
            except (TypeError, ValueError):
                continue
            inspected.add((str(item.get("source", "")), str(item.get("location", ""))))
    unresolved: list[str] = []
    for item in data.get("evidence", []) or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", ""))
        line = "" if item.get("line") is None else str(item.get("line"))
        if not any(source == path for source, _location in inspected):
            unresolved.append(f"{path}:{line}".rstrip(":"))
    return unresolved


def _append_unique(data: dict[str, Any], field: str, value: str) -> None:
    current = [str(item) for item in data.get(field, []) if item is not None]
    if value not in current:
        current.append(value)
    data[field] = current
