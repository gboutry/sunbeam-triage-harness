from __future__ import annotations

from typing import Any

from .llm_exchanges import tool_call_name_and_arguments

EVIDENCE_PRODUCING_TOOLS = {
    "search_artifacts",
    "get_artifact_file",
    "search_archive",
    "get_archive_file",
    "search_sosreport",
    "get_sosreport_file",
}
TARGETED_READ_TOOLS = {
    "get_artifact_file",
    "get_archive_file",
    "get_sosreport_file",
}
DISCOVERY_TOOLS = {
    "list_artifact_files",
    "list_archive_files",
    "list_sosreports",
    "list_sosreport_files",
}


def exchange_range_has_tool_calls(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    for exchange in exchanges[start_index:]:
        response = exchange.get("response", {})
        if isinstance(response, dict) and response.get("tool_calls"):
            return True
    return False


def exchange_range_has_evidence_tool_calls(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    return bool(
        exchange_range_tool_names(exchanges, start_index) & EVIDENCE_PRODUCING_TOOLS
    )


def exchange_range_has_targeted_read_tool_calls(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    return bool(exchange_range_tool_names(exchanges, start_index) & TARGETED_READ_TOOLS)


def exchange_range_tool_names(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> set[str]:
    names = set()
    for exchange in exchanges[start_index:]:
        response = exchange.get("response", {})
        if not isinstance(response, dict):
            continue
        for tool_call in response.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if isinstance(function, dict):
                names.add(str(function.get("name", "")))
    return names


def exchange_range_has_tool_budget_fallback(
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    for exchange in exchanges[start_index:]:
        request = exchange.get("request", {})
        if not isinstance(request, dict):
            continue
        for message in request.get("messages", []):
            if not isinstance(message, dict):
                continue
            content = str(message.get("content", ""))
            if "artifact tool budget is exhausted" in content:
                return True
    return False


def diagnosis_needs_required_tool_retry(
    data: dict[str, Any],
    exchanges: list[dict[str, Any]],
    start_index: int,
) -> bool:
    if data.get("needs_more_evidence") is True and not exchange_range_has_tool_calls(
        exchanges,
        start_index,
    ):
        return True
    if data.get("confidence") == "confirmed":
        return not exchange_range_has_evidence_tool_calls(exchanges, start_index)
    return diagnosis_confidence_requires_artifact_evidence(
        data
    ) and not exchange_range_has_evidence_tool_calls(exchanges, start_index)


def diagnosis_confidence_requires_artifact_evidence(data: dict[str, Any]) -> bool:
    return data.get("confidence") in {"supported", "confirmed"}


def downgrade_tool_budget_diagnosis(data: dict[str, Any]) -> dict[str, Any]:
    downgraded = dict(data)
    if downgraded.get("confidence") == "confirmed":
        downgraded["confidence"] = "supported"
    downgraded["needs_more_evidence"] = True
    unknowns = [
        str(item) for item in downgraded.get("unknowns", []) if item is not None
    ]
    budget_unknown = (
        "The artifact tool budget was exhausted before targeted artifact reads "
        "could fully validate the mechanism."
    )
    if budget_unknown not in unknowns:
        unknowns.append(budget_unknown)
    downgraded["unknowns"] = unknowns
    mechanisms = []
    for item in downgraded.get("candidate_mechanisms", []):
        if not isinstance(item, dict):
            continue
        mechanism = dict(item)
        if mechanism.get("status") == "confirmed":
            mechanism["status"] = "supported"
        mechanisms.append(mechanism)
    downgraded["candidate_mechanisms"] = mechanisms
    return downgraded


def tool_calls_need_targeted_read_nudge(tool_calls: list[Any]) -> bool:
    names = {tool_call_name_and_arguments(call)[0] for call in tool_calls}
    broad_tools = DISCOVERY_TOOLS | {"search_artifacts"}
    targeted_tools = EVIDENCE_PRODUCING_TOOLS - {"search_artifacts"}
    return bool(names & broad_tools) and not bool(names & targeted_tools)
