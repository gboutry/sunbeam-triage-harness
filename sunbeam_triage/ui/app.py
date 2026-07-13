from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st

from sunbeam_triage.core.config import Config
from sunbeam_triage.core.markdown_export import render_diagnosis_markdown
from sunbeam_triage.core.progress import (
    ProgressEvent,
    ProgressSink,
    summarize_progress_events,
)
from sunbeam_triage.core.redaction import redact_data
from sunbeam_triage.core.sessions import (
    list_session_records,
    load_session_record,
)
from sunbeam_triage.core.tool_activity import analyze_tool_activity
from sunbeam_triage.core.use_cases import (
    ArenaRetryRequest,
    ArenaRunRequest,
    ArenaVerdictRequest,
    DiagnosisRunRequest,
    FollowupRequest,
    TriageUseCases,
    report_from_session,
)
from sunbeam_triage.ui.helpers import (
    evidence_line_map,
    list_artifact_files,
    read_text_preview,
    render_line_preview,
)

PREVIEW_CSS = """
<style>
.file-preview {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
}
.file-preview td {
  border-top: 1px solid #e5e7eb;
  padding: 2px 8px;
  vertical-align: top;
}
.file-preview .line-number {
  color: #6b7280;
  text-align: right;
  user-select: none;
  width: 5rem;
}
.file-preview .line-text {
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}
.file-preview .evidence-line {
  background: #fff3bf;
}
</style>
""".strip()


def main() -> None:
    st.set_page_config(page_title="Sunbeam Triage Cockpit", layout="wide")
    config = Config.load("config.toml")
    _initialize_state(config)

    selected_uuid = _sidebar(config)
    session = _load_active_session(config, selected_uuid)
    if st.session_state.get("pending_run"):
        _execute_pending_run(config, st.session_state["pending_run"], session)
        return

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        _render_diagnosis_chat(config, session)
    with right:
        _render_context_panel(config, session)


def _initialize_state(config: Config) -> None:
    st.session_state.setdefault("selected_uuid", "")
    st.session_state.setdefault("selected_arena_id", "")
    st.session_state.setdefault("pending_attachments", [])
    st.session_state.setdefault("pending_run", None)
    st.session_state.setdefault("active_progress_events", [])
    if not st.session_state["selected_uuid"]:
        sessions = _list_diagnosis_sessions(config.paths.artifact_root)
        if sessions:
            st.session_state["selected_uuid"] = sessions[0]["uuid"]


def _sidebar(config: Config) -> str:
    with st.sidebar:
        st.header("Triage Cockpit")
        with st.form("start-diagnosis", clear_on_submit=False):
            uuid = st.text_input("Solutions Run UUID")
            model = st.text_input("Model", value=config.llm.model)
            budget = st.selectbox(
                "Budget",
                ["default", "quick", "hard"],
                index=0,
                help="Tool-round budget profile for the diagnosis loop.",
            )
            submitted = st.form_submit_button("Start diagnosis", type="primary")
        if submitted:
            st.session_state["pending_run"] = {
                "type": "diagnosis",
                "uuid": uuid.strip(),
                "model": model.strip() or config.llm.model,
                "budget": budget,
            }
            st.rerun()

        with st.expander("Arena"):
            with st.form("start-arena", clear_on_submit=False):
                arena_uuid = st.text_input("Arena UUID")
                arena_models = st.text_input(
                    "Contenders",
                    value=", ".join(config.arena.models),
                )
                arena_budget = st.selectbox(
                    "Arena budget",
                    ["default", "quick", "hard"],
                    index=0,
                )
                arena_submitted = st.form_submit_button("Start arena")
            if arena_submitted:
                st.session_state["pending_run"] = {
                    "type": "arena",
                    "uuid": arena_uuid.strip(),
                    "models": [
                        item.strip() for item in arena_models.split(",") if item.strip()
                    ],
                    "budget": arena_budget,
                }
                st.rerun()

        st.divider()
        query = st.text_input("Search history")
        sessions = _list_diagnosis_sessions(config.paths.artifact_root)
        if query:
            sessions = [
                item
                for item in sessions
                if query.lower() in item["uuid"].lower()
                or query.lower() in item["summary"].lower()
            ]

        for item in sessions:
            label = f"{item['uuid']}\n\n{item['confidence']} · {item['model']}"
            if st.button(label, key=f"history-{item['uuid']}", width="stretch"):
                st.session_state["selected_uuid"] = item["uuid"]

    return st.session_state.get("selected_uuid", "")


def _execute_pending_run(
    config: Config,
    pending: dict[str, Any],
    session: dict[str, Any] | None,
) -> None:
    st.title("Sunbeam Triage")
    events: list[dict[str, Any]] = []
    st.session_state["active_progress_events"] = events
    progress_area = st.empty()

    def progress(event: ProgressEvent) -> None:
        _append_progress_event(events, event)
        with progress_area.container():
            _render_progress_console(events, title=_pending_run_title(pending))

    with progress_area.container():
        _render_progress_console(events, title=_pending_run_title(pending))
    try:
        if pending.get("type") == "diagnosis":
            _start_diagnosis(
                config,
                str(pending.get("uuid", "")),
                str(pending.get("model", config.llm.model)),
                str(pending.get("budget", "default")),
                progress=progress,
                progress_events=events,
            )
        elif pending.get("type") == "arena":
            _start_arena(
                config,
                str(pending.get("uuid", "")),
                [str(item) for item in pending.get("models", [])],
                str(pending.get("budget", "default")),
                progress=progress,
                progress_events=events,
            )
        elif pending.get("type") == "arena_retry":
            _retry_failed_arena(
                config,
                str(pending.get("session_id", "")),
                progress=progress,
                progress_events=events,
            )
        elif pending.get("type") == "followup" and session:
            _send_followup(
                config,
                session,
                str(pending.get("prompt", "")),
                progress=progress,
                progress_events=events,
            )
        else:
            st.error("Unknown pending run.")
            return
    finally:
        st.session_state["pending_run"] = None
    st.rerun()


def _pending_run_title(pending: dict[str, Any]) -> str:
    run_type = str(pending.get("type", "run"))
    uuid = str(pending.get("uuid", ""))
    if run_type == "followup":
        return "Follow-up in progress"
    return f"{run_type.title()} in progress: {uuid}"


def _start_arena(
    config: Config,
    uuid: str,
    models: list[str],
    budget: str,
    *,
    progress: ProgressSink | None = None,
    progress_events: list[dict[str, Any]] | None = None,
) -> None:
    if not uuid:
        st.sidebar.error("Enter a Solutions Run UUID.")
        return
    if len(models) < 2:
        st.sidebar.error("Enter at least two contender models.")
        return
    try:
        result = TriageUseCases(config).run_arena(
            ArenaRunRequest(uuid=uuid, models=models, budget=budget),
            progress=progress,
            progress_events=progress_events,
        )
        st.session_state["selected_arena_id"] = result.selected_arena_id
    except Exception as exc:
        _emit_ui_progress(
            progress,
            uuid,
            "arena",
            "failed",
            "failed",
            str(exc),
            warning=str(exc),
        )
        st.error(str(exc))


def _retry_failed_arena(
    config: Config,
    session_id: str,
    *,
    progress: ProgressSink | None = None,
    progress_events: list[dict[str, Any]] | None = None,
) -> None:
    try:
        result = TriageUseCases(config).retry_failed_arena(
            ArenaRetryRequest(session_id=session_id),
            progress=progress,
            progress_events=progress_events,
        )
    except Exception as exc:
        st.error(str(exc))
        return
    st.session_state["selected_arena_id"] = result.selected_arena_id


def _start_diagnosis(
    config: Config,
    uuid: str,
    model: str,
    budget: str,
    *,
    progress: ProgressSink | None = None,
    progress_events: list[dict[str, Any]] | None = None,
) -> None:
    if not uuid:
        st.sidebar.error("Enter a Solutions Run UUID.")
        return

    result = TriageUseCases(config).run_diagnosis(
        DiagnosisRunRequest(uuid=uuid, model=model, budget=budget),
        progress=progress,
        progress_events=progress_events,
    )
    st.session_state["selected_uuid"] = result.selected_uuid
    if result.clear_attachments:
        st.session_state["pending_attachments"] = []
    if result.error:
        st.error(result.error)


def _load_active_session(config: Config, uuid: str) -> dict[str, Any] | None:
    if not uuid:
        return None
    return _load_diagnosis_session(config.paths.artifact_root, uuid)


def _list_diagnosis_sessions(artifact_root: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in list_session_records(artifact_root)
        if record.get("session_type") == "diagnosis"
    ]


def _load_diagnosis_session(
    artifact_root: Path,
    session_id: str,
) -> dict[str, Any] | None:
    loaded = load_session_record(artifact_root, session_id)
    if not loaded:
        return None
    snapshot = loaded["snapshot"]
    if snapshot.get("session_type") != "diagnosis":
        return None
    return snapshot


def _render_diagnosis_chat(config: Config, session: dict[str, Any] | None) -> None:
    st.title("Sunbeam Triage")
    if not session:
        st.info("Start a diagnosis or select a UUID from history.")
        return
    if session.get("error"):
        st.error(session["error"])
        _render_markdown_export_button(session)
        _render_download_failures(session)
        if st.button("Retry diagnosis"):
            st.session_state["pending_run"] = {
                "type": "diagnosis",
                "uuid": session["uuid"],
                "model": session.get("model", config.llm.model),
                "budget": "default",
            }
            st.rerun()
        return

    _render_diagnosis_result(session)

    st.subheader("Conversation")
    for message in session.get("chat", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    attachments = st.session_state.get("pending_attachments", [])
    if attachments:
        labels = [_format_attachment(item) for item in attachments]
        st.caption("Attached to next message: " + "; ".join(labels))
        if st.button("Clear attachments"):
            st.session_state["pending_attachments"] = []
            st.rerun()

    prompt = st.chat_input("Ask about this diagnosis")
    if prompt:
        st.session_state["pending_run"] = {
            "type": "followup",
            "uuid": session["uuid"],
            "prompt": prompt,
        }
        st.rerun()


def _send_followup(
    config: Config,
    session: dict[str, Any],
    prompt: str,
    *,
    progress: ProgressSink | None = None,
    progress_events: list[dict[str, Any]] | None = None,
) -> None:
    result = TriageUseCases(config).send_followup(
        FollowupRequest(
            session=session,
            prompt=prompt,
            attachments=st.session_state.get("pending_attachments", []),
        ),
        progress=progress,
        progress_events=progress_events,
    )
    if result.clear_attachments:
        st.session_state["pending_attachments"] = []


def _render_context_panel(config: Config, session: dict[str, Any] | None) -> None:
    st.subheader("Context")
    if not session:
        tabs = st.tabs(["Arenas"])
        with tabs[0]:
            _render_arena_tab(config)
        return
    tabs = st.tabs([
        "Evidence",
        "Probes",
        "Files",
        "Tool Activity",
        "Progress",
        "API",
        "Arenas",
    ])
    with tabs[0]:
        _render_evidence_tab(session)
    with tabs[1]:
        _render_probe_tab(session)
    with tabs[2]:
        _render_files_tab(config, session)
    with tabs[3]:
        _render_tool_activity_tab(session)
    with tabs[4]:
        _render_saved_progress_tab(session)
    with tabs[5]:
        st.json(session.get("exchanges", []), expanded=False)
    with tabs[6]:
        _render_arena_tab(config)


def _render_diagnosis_result(session: dict[str, Any]) -> None:
    st.caption(f"{session['uuid']} · {session['model']}")
    st.subheader(session.get("failed_step") or "Diagnosis result")
    metadata = _result_metadata(session)
    if metadata:
        st.caption(" · ".join(metadata))
    _render_markdown_export_button(session)
    st.markdown("**Primary finding**")
    st.write(_primary_finding(session))

    timeline_rows = _failure_timeline_rows(session)
    if timeline_rows:
        st.markdown("**Failure timeline**")
        st.dataframe(timeline_rows, width="stretch", hide_index=True)

    cols = st.columns(3)
    cols[0].metric("Confidence", session.get("confidence", ""))
    cols[1].metric("Failed step", session.get("failed_step", ""))
    cols[2].metric("Chat turns", len(session.get("chat", [])) // 2)

    candidate_rows = _candidate_mechanism_rows(session)
    st.markdown("**Candidate mechanisms**")
    if candidate_rows:
        st.dataframe(candidate_rows, width="stretch", hide_index=True)
    else:
        st.caption("No candidate mechanisms recorded.")

    with st.expander("Diagnosis details", expanded=False):
        _render_download_failures(session)
        assessment = session.get("causal_assessment")
        if isinstance(assessment, dict):
            for label, value in _causal_rows(assessment):
                st.markdown(f"**{label}**")
                st.write(value or "None established.")
        else:
            st.markdown("**Failure Surface**")
            st.write(session.get("failure_surface", ""))
            st.markdown("**Root Cause**")
            st.write(session.get("root_cause", ""))


def _primary_finding(session: dict[str, Any]) -> str:
    assessment = session.get("causal_assessment")
    if isinstance(assessment, dict):
        root = assessment.get("root_cause")
        if isinstance(root, dict) and root.get("confidence") != "unknown":
            claim = str(root.get("claim", "")).strip()
            if claim:
                return claim
        trigger = assessment.get("failure_trigger")
        if isinstance(trigger, dict):
            claim = str(trigger.get("claim", "")).strip()
            if claim:
                return claim
    root_cause = str(session.get("root_cause", "")).strip()
    if root_cause:
        return root_cause
    summary = str(session.get("summary", "")).strip()
    return summary or "No diagnosis summary recorded."


def _causal_rows(assessment: dict[str, Any]) -> list[tuple[str, str]]:
    def claim(name: str) -> str:
        value = assessment.get(name)
        return str(value.get("claim", "")) if isinstance(value, dict) else ""

    def claims(name: str) -> str:
        values = assessment.get(name, [])
        if not isinstance(values, list):
            return ""
        return "; ".join(
            str(value.get("claim", ""))
            for value in values
            if isinstance(value, dict) and value.get("claim")
        )

    return [
        ("Failure Trigger", claim("failure_trigger")),
        ("Symptoms", claims("symptoms")),
        ("Contributing Factors", claims("contributing_factors")),
        ("Root Cause", claim("root_cause")),
        ("Post-Failure Outcome", claim("post_failure_outcome")),
    ]


def _render_markdown_export_button(session: dict[str, Any]) -> None:
    st.download_button(
        "Export report + chat",
        data=render_diagnosis_markdown(session),
        file_name=_markdown_export_filename(session),
        mime="text/markdown",
        key=f"export-markdown-{session.get('session_id', session.get('uuid', 'active'))}",
    )


def _markdown_export_filename(session: dict[str, Any]) -> str:
    uuid = str(session.get("uuid", "")).strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", uuid).strip(".-")
    if not safe:
        return "sunbeam-triage-report.md"
    return f"sunbeam-triage-{safe}.md"


def _result_metadata(session: dict[str, Any]) -> list[str]:
    metadata = []
    triage_confidence = str(session.get("triage_confidence", "")).strip()
    stop_reason = str(session.get("stop_reason", "")).strip()
    investigation_status = str(session.get("investigation_status", "")).strip()
    verdict_source = str(session.get("verdict_source", "")).strip()
    if investigation_status:
        metadata.append(f"investigation: {investigation_status}")
    if verdict_source:
        metadata.append(f"verdict source: {verdict_source}")
    if triage_confidence and triage_confidence != "unknown":
        metadata.append(f"triage confidence: {triage_confidence}")
    if stop_reason:
        metadata.append(f"stop reason: {stop_reason}")
    if session.get("needs_more_evidence"):
        metadata.append("needs more evidence")
    return metadata


def _failure_timeline_rows(
    session: dict[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, str]]:
    rows = []
    for item in session.get("failure_timeline", [])[:limit]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        location = str(item.get("location", "")).strip()
        source = f"{source}: {location}" if source and location else source or location
        rows.append({
            "time": str(item.get("timestamp", "")).strip(),
            "source": source,
            "event": str(item.get("event", "")).strip(),
        })
    return rows


def _candidate_mechanism_rows(session: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for item in session.get("candidate_mechanisms", []):
        if not isinstance(item, dict):
            continue
        rows.append({
            "mechanism": str(item.get("name", "")).strip(),
            "status": str(item.get("status", "")).strip(),
            "rationale": str(item.get("rationale", "")).strip(),
        })
    return rows


def _render_arena_tab(config: Config) -> None:
    records = [
        record
        for record in list_session_records(config.paths.artifact_root)
        if record.get("session_type") == "arena"
    ]
    if not records:
        st.info("No arena records found.")
        return
    selected = st.selectbox(
        "Arena",
        records,
        index=_selected_arena_index(records),
        format_func=lambda item: f"{item['uuid']} · {item['status']}",
    )
    st.session_state["selected_arena_id"] = selected["session_id"]
    loaded = load_session_record(config.paths.artifact_root, selected["session_id"])
    if not loaded:
        st.error("Arena record is missing.")
        return
    arena = loaded["snapshot"]
    reveal = "verdict" in arena
    st.caption(f"{arena['uuid']} · {arena['status']}")
    if arena.get("output"):
        st.write(f"Report: `{arena['output']}`")
    if arena.get("progress_events"):
        with st.expander("Progress trace"):
            _render_saved_progress_tab(arena)
    for contender in arena.get("contenders", []):
        label = _arena_contender_label(contender, reveal_model=reveal)
        with st.expander(label, expanded=True):
            provenance = " · ".join(
                value
                for value in (
                    str(contender.get("investigation_status", "")),
                    str(contender.get("verdict_source", "")),
                )
                if value
            )
            if provenance:
                st.caption(provenance)
            if contender.get("status") != "completed":
                st.error(contender.get("error", "Contender failed."))
                continue
            report = contender.get("report", {})
            st.markdown("**Summary**")
            st.write(report.get("summary", ""))
            st.markdown("**Root Cause**")
            st.write(report.get("root_cause", ""))
            st.caption(f"Confidence: {report.get('confidence', '')}")
    if any(
        contender.get("status") == "failed" for contender in arena.get("contenders", [])
    ) and st.button(
        "Retry failed contenders",
        key=f"retry-arena-{arena['session_id']}",
    ):
        st.session_state["pending_run"] = {
            "type": "arena_retry",
            "session_id": arena["session_id"],
        }
        st.rerun()
    _render_arena_verdict_form(config, arena)


def _selected_arena_index(records: list[dict[str, Any]]) -> int:
    selected_id = st.session_state.get("selected_arena_id", "")
    for index, record in enumerate(records):
        if record.get("session_id") == selected_id:
            return index
    return 0


def _render_arena_verdict_form(config: Config, arena: dict[str, Any]) -> None:
    contenders = [
        contender
        for contender in arena.get("contenders", [])
        if contender.get("status") == "completed"
    ]
    if len(contenders) < 2:
        return
    contender_ids = [str(contender["contender_id"]) for contender in contenders]
    with st.form(f"arena-verdict-{arena['session_id']}"):
        winner = st.selectbox("Winner", contender_ids)
        rubric: dict[str, dict[str, int]] = {}
        for contender_id in contender_ids:
            st.markdown(f"**Contender {contender_id}**")
            rubric[contender_id] = {
                "root_cause": st.slider(
                    f"{contender_id} root cause",
                    min_value=1,
                    max_value=5,
                    value=3,
                ),
                "evidence": st.slider(
                    f"{contender_id} evidence",
                    min_value=1,
                    max_value=5,
                    value=3,
                ),
                "timeline": st.slider(
                    f"{contender_id} timeline",
                    min_value=1,
                    max_value=5,
                    value=3,
                ),
                "uncertainty": st.slider(
                    f"{contender_id} uncertainty",
                    min_value=1,
                    max_value=5,
                    value=3,
                ),
                "next_steps": st.slider(
                    f"{contender_id} next steps",
                    min_value=1,
                    max_value=5,
                    value=3,
                ),
            }
        notes = st.text_area("Notes", value=arena.get("verdict", {}).get("notes", ""))
        if st.form_submit_button("Save verdict"):
            _save_arena_verdict(
                config,
                arena,
                winner=winner,
                notes=notes,
                rubric=rubric,
            )
            st.rerun()


def _arena_contender_label(contender: dict[str, Any], *, reveal_model: bool) -> str:
    label = f"Contender {contender.get('contender_id', '')}"
    if reveal_model:
        label += f" - {contender.get('model', '')}"
    return label


def _save_arena_verdict(
    config: Config,
    session: dict[str, Any],
    *,
    winner: str,
    notes: str,
    rubric: dict[str, dict[str, int]],
) -> dict[str, Any]:
    return TriageUseCases(config).save_arena_verdict(
        ArenaVerdictRequest(
            session=session,
            winner=winner,
            notes=notes,
            rubric=rubric,
        )
    )


def _render_download_failures(session: dict[str, Any]) -> None:
    failures = session.get("download_failures", [])
    if not failures:
        return
    st.warning(f"{len(failures)} Swift objects failed to download.")
    st.dataframe(failures, width="stretch", hide_index=True)


def _render_evidence_tab(session: dict[str, Any]) -> None:
    evidence = session.get("evidence", [])
    if not evidence:
        st.info("No model evidence recorded.")
        return
    for index, item in enumerate(evidence):
        label = _format_attachment(item)
        st.code(item.get("excerpt", ""), language=None)
        if st.button(f"Attach {label}", key=f"attach-evidence-{index}"):
            _add_attachment({
                "path": item.get("path", ""),
                "line": item.get("line"),
                "text": item.get("excerpt", ""),
            })
            st.rerun()


def _render_probe_tab(session: dict[str, Any]) -> None:
    probe_results = [
        result
        for result in session.get("probe_results", [])
        if isinstance(result, dict)
    ]
    if not probe_results:
        st.info("No deterministic probe results recorded.")
        return
    rows = []
    for result in probe_results:
        for finding in result.get("findings", []):
            if not isinstance(finding, dict):
                continue
            rows.append({
                "probe": result.get("name", ""),
                "status": result.get("status", ""),
                "category": finding.get("category", ""),
                "path": finding.get("path", ""),
                "line": finding.get("line", ""),
                "excerpt": finding.get("excerpt", ""),
            })
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    for result in probe_results:
        st.markdown(
            f"**{result.get('name', '')}** · {result.get('status', '')}: "
            f"{result.get('summary', '')}"
        )
        for index, finding in enumerate(result.get("findings", [])):
            if not isinstance(finding, dict):
                continue
            label = _format_attachment(finding)
            st.code(finding.get("excerpt", ""), language=None)
            if st.button(
                f"Attach probe {label}",
                key=f"attach-probe-{result.get('name', '')}-{index}",
            ):
                _add_attachment({
                    "path": finding.get("path", ""),
                    "line": finding.get("line"),
                    "text": finding.get("excerpt", ""),
                })
                st.rerun()
        missing = result.get("missing_evidence", [])
        if missing:
            st.caption("Missing evidence: " + "; ".join(str(item) for item in missing))


def _render_tool_activity_tab(session: dict[str, Any]) -> None:
    analysis = analyze_tool_activity(session)
    cols = st.columns(5)
    cols[0].metric("Exchanges", analysis["exchange_count"])
    cols[1].metric("Tool calls", analysis["tool_call_count"])
    cols[2].metric("Tool result chars", analysis["tool_result_chars"])
    cols[3].metric("Session tokens", analysis["total_tokens"])
    cols[4].metric("Session cost", _format_usd(analysis["total_cost"]))
    if analysis["warnings"]:
        st.warning(", ".join(analysis["warnings"]))
    if analysis["repeated_reads"]:
        st.dataframe(analysis["repeated_reads"], width="stretch", hide_index=True)
    if analysis["rows"]:
        st.dataframe(analysis["rows"], width="stretch", hide_index=True)
    else:
        st.info("No tool activity recorded.")


def _render_saved_progress_tab(session: dict[str, Any]) -> None:
    events = [
        event for event in session.get("progress_events", []) if isinstance(event, dict)
    ]
    if not events:
        st.info("No progress trace recorded for this session.")
        return
    _render_progress_console(events, title="Saved progress trace")


def _render_progress_console(events: list[dict[str, Any]], *, title: str) -> None:
    st.subheader(title)
    summary = summarize_progress_events(events)
    cols = st.columns(5)
    cols[0].metric("Events", summary["event_count"])
    cols[1].metric("Tool calls", summary["tool_call_count"])
    cols[2].metric("Tool result chars", summary["tool_result_chars"])
    cols[3].metric("Tokens", summary["total_tokens"])
    cols[4].metric("Cost", _format_usd(summary["total_cost"]))
    if summary["warnings"]:
        st.warning(", ".join(summary["warnings"]))
    if not events:
        st.info("Waiting for the run to start.")
        return

    arena_events = [event for event in events if event.get("run_type") == "arena"]
    contenders = sorted({
        str(event.get("contender_id"))
        for event in arena_events
        if event.get("contender_id")
    })
    if contenders:
        columns = st.columns(len(contenders))
        for column, contender_id in zip(columns, contenders, strict=False):
            contender_events = [
                event
                for event in arena_events
                if event.get("contender_id") == contender_id
            ]
            status = (
                contender_events[-1].get("status", "queued")
                if contender_events
                else "queued"
            )
            phase = contender_events[-1].get("phase", "") if contender_events else ""
            column.metric(f"Contender {contender_id}", status, phase)

    st.dataframe(_progress_event_rows(events), width="stretch", hide_index=True)
    with st.expander("Raw progress trace"):
        st.json(events, expanded=False)


def _progress_event_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "time": event.get("created_at", ""),
            "phase": event.get("phase", ""),
            "status": event.get("status", ""),
            "contender": event.get("contender_id", ""),
            "message": event.get("message", ""),
            "tool": event.get("tool_name", ""),
            "target": event.get("target", ""),
            "chars": event.get("result_chars"),
        }
        for event in events[-25:]
    ]


def _emit_ui_progress(
    progress: ProgressSink | None,
    run_id: str,
    run_type: str,
    phase: str,
    status: str,
    message: str,
    *,
    warning: str | None = None,
) -> None:
    if progress is None:
        return
    progress(
        ProgressEvent(
            run_id=run_id,
            run_type=run_type,
            phase=phase,
            status=status,
            message=message,
            warning=warning,
        )
    )


def _render_files_tab(config: Config, session: dict[str, Any]) -> None:
    root = Path(
        session.get("artifact_root", config.paths.artifact_root / session["uuid"])
    )
    files = list_artifact_files(root)
    if not files:
        st.info(f"No downloaded files found under `{root}`.")
        return
    selected = st.selectbox(
        "File",
        files,
        format_func=lambda path: path.as_posix(),
    )
    path = root / selected
    report = report_from_session(session)
    highlights = evidence_line_map(report).get(selected.as_posix(), set())
    preview = read_text_preview(path)
    st.caption(f"`{path}`")
    if preview.binary:
        st.warning("Binary file preview is not available.")
        return
    if preview.truncated:
        st.warning("Preview truncated for responsiveness.")
    if highlights:
        st.caption(f"Highlighted model-referenced lines: {sorted(highlights)}")
        if st.button("Attach highlighted lines"):
            for line_number in sorted(highlights):
                line_text = _line_at(preview.text, line_number)
                _add_attachment({
                    "path": selected.as_posix(),
                    "line": line_number,
                    "text": line_text,
                })
            st.rerun()
    st.markdown(PREVIEW_CSS, unsafe_allow_html=True)
    st.markdown(render_line_preview(preview.text, highlights), unsafe_allow_html=True)


def _add_attachment(item: dict[str, Any]) -> None:
    item = redact_data(item)
    attachments = st.session_state.setdefault("pending_attachments", [])
    if item not in attachments:
        attachments.append(item)


def _append_progress_event(events: list[dict[str, Any]], event: ProgressEvent) -> None:
    events.append(event.to_trace())


def _format_attachment(item: dict[str, Any]) -> str:
    line = "" if item.get("line") is None else f":{item['line']}"
    return f"{item.get('path', '')}{line}"


def _format_usd(value: float) -> str:
    return f"${value:.6f}"


def _line_at(text: str, number: int) -> str:
    lines = text.splitlines()
    if number < 1 or number > len(lines):
        return ""
    return lines[number - 1].strip()


if __name__ == "__main__":
    main()
