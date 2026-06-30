from __future__ import annotations

from typing import Any

from .llm_exchanges import response_content
from .triage_state import InvestigationState


def diagnosis_system_prompt() -> str:
    return (
        "You are a CI failure diagnostician. Use only the provided "
        "evidence. Do not invent files, timestamps, or causes. "
        "Actively inspect the next likely decisive artifact when "
        "the current evidence only identifies a wrapper failure. "
        "Prefer the earliest event that explains the final failure "
        "surface over later cascades or merely crisp symptoms. "
        "A workload process crash is not sufficient evidence for "
        "Juju unit-agent lost unless Juju status history, unit-agent "
        "logs, or controller logs connect those events. Separate "
        "direct evidence from inference, use timestamps to "
        "correlate services, identify cascades when possible, and "
        "check at least one plausible alternative before claiming "
        "high confidence. For each candidate mechanism, include "
        "evidence for it, evidence against it, missing evidence, "
        "and a confirmed/supported/speculative classification. "
        "Use search_artifacts before reading whole files. For "
        "sosreport archives, use search_sosreport before reading "
        "a specific member. Prefer line windows over full file "
        "reads. Read a specific file only when it is likely to "
        "materially improve the diagnosis, because file reads can "
        "be costly. Set needs_more_evidence to true only when a "
        "likely decisive artifact remains uninspected; if tools "
        "are available, use them before finalizing that response."
    )


def chat_system_prompt() -> str:
    return (
        "You are continuing a Sunbeam CI failure diagnosis. "
        "Use the provided diagnosis context and evidence. "
        "Separate evidence from inference. Actively inspect the "
        "next likely decisive artifact when the current context "
        "only identifies a wrapper failure. "
        "Prefer the earliest event that explains the final failure "
        "surface over later cascades or merely crisp symptoms. "
        "A workload process crash is not sufficient evidence for "
        "Juju unit-agent lost unless Juju status history, unit-agent "
        "logs, or controller logs connect those events. Correlate "
        "timestamps across logs, and preserve uncertainty when "
        "evidence is incomplete. "
        "Use search_artifacts "
        "before reading whole files. For sosreport archives, use "
        "search_sosreport before reading a specific member. "
        "Prefer line windows over full file reads. Read a "
        "specific file only when it is likely to materially "
        "improve the answer, because file reads can be costly."
    )


def request_with_evidence_retry(
    request: dict[str, Any],
    response: Any,
    *,
    data: dict[str, Any],
) -> dict[str, Any]:
    if data.get("needs_more_evidence") is True:
        retry_content = (
            "You marked needs_more_evidence=true without using artifact "
            "tools. Do not answer yet. First use the artifact tools to "
            "search for the most likely decisive log or read a narrow "
            "line window, then finalize the diagnosis. Keep tool usage "
            "targeted."
        )
    else:
        retry_content = (
            "You returned a supported or confirmed diagnosis without using "
            "an evidence-producing artifact tool. Do not answer yet. First "
            "use an evidence-producing artifact tool such as search_artifacts, "
            "get_artifact_file, search_sosreport, or get_sosreport_file to "
            "validate the most likely decisive evidence, then finalize the "
            "diagnosis. Keep tool usage targeted."
        )
    return {
        **request,
        "messages": [
            *request["messages"],
            {"role": "assistant", "content": response_content(response)},
            {"role": "user", "content": retry_content},
        ],
    }


def targeted_read_nudge_message() -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "The previous tool round only discovered candidate artifacts or "
            "broad matches. If the evidence is still insufficient, make the "
            "next tool call a targeted read or targeted archive search, such "
            "as get_artifact_file, search_sosreport, or get_sosreport_file. "
            "If no targeted read is justified, keep the final diagnosis "
            "conservative and leave needs_more_evidence=true."
        ),
    }


def investigation_state_message(state: InvestigationState) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Current harness investigation state after the last tool round:\n"
            f"{state.to_prompt_summary()}\n"
            "Use this state to avoid repeating unproductive searches. Continue "
            "only if the next tool call is likely to add new evidence, refine "
            "the failure timeline, or check a plausible alternative."
        ),
    }


def finalization_message(state: InvestigationState) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Finalize the triage now because the harness stopping rule "
            f"triggered: {state.stop_reason}. The artifact tool budget is "
            "exhausted if stop_reason is budget_exhausted. Answer using "
            "the initial evidence, tool results already provided, and this "
            f"investigation state: {state.to_prompt_summary()}. Do not "
            "request more files. Prefer the earliest event that explains "
            "the final failure surface over later cascades or merely crisp "
            "symptoms. A workload process crash is not sufficient evidence "
            "for Juju unit-agent lost unless Juju evidence connects the "
            "events. Separate evidence from inference. Include a partial "
            "but useful conclusion, confidence, missing evidence, "
            "alternatives considered or unchecked due budget, and "
            "recommended next checks. Do not mark a diagnosis confirmed "
            "solely from discovery-only tool results; preserve uncertainty "
            "when targeted files or sosreport members were not read."
        ),
    }
