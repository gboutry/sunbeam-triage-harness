from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from sunbeam_triage.arena import ArenaOptions, ArenaRunner
from sunbeam_triage.config import Config
from sunbeam_triage.evidence import EvidenceCollector
from sunbeam_triage.llm import DiagnosisReport, OpenRouterClient
from sunbeam_triage.progress import ProgressEvent, ProgressSink, summarize_progress_events
from sunbeam_triage.render import render_html
from sunbeam_triage.sessions import (
    append_session_event,
    list_session_records,
    load_session_record,
    save_session_snapshot,
)
from sunbeam_triage.swift import SwiftMirror
from sunbeam_triage.tool_activity import analyze_tool_activity
from sunbeam_triage.triage_state import BudgetProfile, resolve_triage_budget
from sunbeam_triage.ui_helpers import (
    build_followup_context,
    evidence_line_map,
    list_artifact_files,
    list_saved_sessions,
    load_ui_session,
    read_text_preview,
    render_line_preview,
    save_ui_session,
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
        sessions = list_saved_sessions(config.paths.artifact_root)
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
                        item.strip()
                        for item in arena_models.split(",")
                        if item.strip()
                    ],
                    "budget": arena_budget,
                }
                st.rerun()

        st.divider()
        query = st.text_input("Search history")
        sessions = list_saved_sessions(config.paths.artifact_root)
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
        _emit_ui_progress(
            progress,
            uuid,
            "arena",
            "download",
            "running",
            "Downloading artifacts",
        )
        SwiftMirror(config.swift, config.paths.artifact_root).mirror_uuid(
            uuid,
            continue_on_error=True,
        )
        _emit_ui_progress(
            progress,
            uuid,
            "arena",
            "arena_running",
            "running",
            f"Running {len(models)} contenders",
        )
        session = ArenaRunner(config).run(
            uuid,
            ArenaOptions(models=models, budget=budget),
            progress=progress,
        )
        session["progress_events"] = list(progress_events or [])
        save_session_snapshot(config.paths.artifact_root, session)
        st.session_state["selected_arena_id"] = session["session_id"]
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
    loaded = load_session_record(config.paths.artifact_root, session_id)
    if not loaded:
        st.error("Arena record is missing.")
        return
    arena = loaded["snapshot"]
    models = [str(item.get("model", "")) for item in arena.get("contenders", [])]
    updated = ArenaRunner(config).retry_failed(
        arena,
        ArenaOptions(models=models, budget=str(arena.get("budget", "default"))),
        progress=progress,
    )
    updated["progress_events"] = [
        *[
            event
            for event in arena.get("progress_events", [])
            if isinstance(event, dict)
        ],
        *list(progress_events or []),
    ]
    save_session_snapshot(config.paths.artifact_root, updated)
    st.session_state["selected_arena_id"] = updated["session_id"]


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

    run_config = Config.load("config.toml", cli_model=model)
    triage_options = resolve_triage_budget(
        BudgetProfile(
            quick_max_rounds=run_config.triage.quick_max_rounds,
            default_max_rounds=run_config.triage.default_max_rounds,
            hard_max_rounds=run_config.triage.hard_max_rounds,
            stall_limit=run_config.triage.stall_limit,
            min_evidence_items=run_config.triage.min_evidence_items,
            max_tool_result_chars=run_config.triage.max_tool_result_chars,
        ),
        budget=budget,
    )
    artifact_root = run_config.paths.artifact_root / uuid
    llm_client = OpenRouterClient(run_config.llm)

    download_failures: list[dict[str, Any]] = []
    try:
        _emit_ui_progress(
            progress,
            uuid,
            "diagnosis",
            "download",
            "running",
            "Downloading artifacts",
        )

        def show_download(event: dict[str, Any]) -> None:
            _emit_ui_progress(
                progress,
                uuid,
                "diagnosis",
                "download",
                "running",
                (
                    f"{event['status']} {event['index']}/{event['total']}: "
                    f"{event['name']}"
                ),
                warning=str(event["error"]) if event.get("error") else None,
            )

        manifest = SwiftMirror(run_config.swift, run_config.paths.artifact_root).mirror_uuid(
            uuid,
            progress=show_download,
            continue_on_error=True,
        )
        _emit_ui_progress(
            progress,
            uuid,
            "diagnosis",
            "download",
            "completed",
            f"Downloaded or reused {len(manifest.objects)} objects",
        )
        if manifest.failures:
            download_failures = [asdict(item) for item in manifest.failures]

        _emit_ui_progress(
            progress,
            uuid,
            "diagnosis",
            "evidence",
            "running",
            "Collecting evidence",
        )
        pack = EvidenceCollector(artifact_root, uuid).collect()

        report = llm_client.diagnose(
            pack.to_prompt_text(),
            session_id=uuid,
            artifact_root=artifact_root,
            max_tool_rounds=triage_options.max_rounds,
            max_tool_result_chars=triage_options.max_tool_result_chars,
            triage_options=triage_options,
            progress=progress,
        )

        output = run_config.output_path(uuid)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_html(pack, report), encoding="utf-8")

        _persist_diagnosis_session(
            run_config.paths.artifact_root,
            _session_from_diagnosis(
                uuid=uuid,
                model=run_config.llm.model,
                artifact_root=artifact_root,
                output=output,
                failed_step=pack.failed_step.name,
                report=report,
                exchanges=llm_client.exchanges,
                download_failures=download_failures,
                progress_events=progress_events,
            ),
        )
        st.session_state["selected_uuid"] = uuid
        st.session_state["pending_attachments"] = []
    except Exception as exc:
        _emit_ui_progress(
            progress,
            uuid,
            "diagnosis",
            "failed",
            "failed",
            str(exc),
            warning=str(exc),
        )
        _persist_diagnosis_session(
            run_config.paths.artifact_root,
            {
                "uuid": uuid,
                "model": run_config.llm.model,
                "summary": str(exc),
                "confidence": "error",
                "updated_at": _now(),
                "artifact_root": str(artifact_root),
                "error": str(exc),
                "chat": [],
                "exchanges": llm_client.exchanges,
                "download_failures": download_failures,
                "progress_events": list(progress_events or []),
            },
        )
        st.session_state["selected_uuid"] = uuid
        st.error(str(exc))


def _load_active_session(config: Config, uuid: str) -> dict[str, Any] | None:
    if not uuid:
        return None
    return load_ui_session(config.paths.artifact_root, uuid)


def _render_diagnosis_chat(config: Config, session: dict[str, Any] | None) -> None:
    st.title("Sunbeam Triage")
    if not session:
        st.info("Start a diagnosis or select a UUID from history.")
        return
    if session.get("error"):
        st.error(session["error"])
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

    st.caption(f"{session['uuid']} · {session['model']}")
    st.subheader(session.get("summary", "Diagnosis"))
    cols = st.columns(3)
    cols[0].metric("Confidence", session.get("confidence", ""))
    cols[1].metric("Failed step", session.get("failed_step", ""))
    cols[2].metric("Chat turns", len(session.get("chat", [])) // 2)

    with st.expander("Diagnosis details", expanded=True):
        _render_download_failures(session)
        st.markdown("**Failure Surface**")
        st.write(session.get("failure_surface", ""))
        st.markdown("**Root Cause**")
        st.write(session.get("root_cause", ""))

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
    pack = EvidenceCollector(Path(session["artifact_root"]), session["uuid"]).collect()
    report = _report_from_session(session)
    context = build_followup_context(
        pack,
        report,
        attachments=st.session_state.get("pending_attachments", []),
    )
    run_config = Config.load("config.toml", cli_model=session["model"])
    llm_client = OpenRouterClient(run_config.llm)
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in session.get("chat", [])
        if item.get("role") in {"user", "assistant"}
    ]
    messages = [*history, {"role": "user", "content": prompt}]
    answer = llm_client.chat(
        context,
        messages,
        session_id=session["uuid"],
        artifact_root=Path(session["artifact_root"]),
        progress=progress,
    )

    session.setdefault("chat", []).extend(
        [
            {"role": "user", "content": prompt, "created_at": _now()},
            {"role": "assistant", "content": answer, "created_at": _now()},
        ]
    )
    session.setdefault("exchanges", []).extend(llm_client.exchanges)
    session.setdefault("progress_events", []).extend(progress_events or [])
    session["updated_at"] = _now()
    _persist_diagnosis_session(config.paths.artifact_root, session)
    st.session_state["pending_attachments"] = []


def _render_context_panel(config: Config, session: dict[str, Any] | None) -> None:
    st.subheader("Context")
    if not session:
        tabs = st.tabs(["Arenas"])
        with tabs[0]:
            _render_arena_tab(config)
        return
    tabs = st.tabs(["Evidence", "Files", "Tool Activity", "Progress", "API", "Arenas"])
    with tabs[0]:
        _render_evidence_tab(session)
    with tabs[1]:
        _render_files_tab(config, session)
    with tabs[2]:
        _render_tool_activity_tab(session)
    with tabs[3]:
        _render_saved_progress_tab(session)
    with tabs[4]:
        st.json(session.get("exchanges", []), expanded=False)
    with tabs[5]:
        _render_arena_tab(config)


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
            if contender.get("status") != "completed":
                st.error(contender.get("error", "Contender failed."))
                continue
            report = contender.get("report", {})
            st.markdown("**Summary**")
            st.write(report.get("summary", ""))
            st.markdown("**Root Cause**")
            st.write(report.get("root_cause", ""))
            st.caption(f"Confidence: {report.get('confidence', '')}")
    if any(contender.get("status") == "failed" for contender in arena.get("contenders", [])):
        if st.button("Retry failed contenders", key=f"retry-arena-{arena['session_id']}"):
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
                config.paths.artifact_root,
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
    artifact_root: Path,
    session: dict[str, Any],
    *,
    winner: str,
    notes: str,
    rubric: dict[str, dict[str, int]],
) -> dict[str, Any]:
    updated = dict(session)
    updated["status"] = "judged"
    updated["updated_at"] = _now()
    updated["verdict"] = {
        "winner": winner,
        "notes": notes,
        "rubric": rubric,
        "created_at": _now(),
    }
    save_session_snapshot(artifact_root, updated)
    append_session_event(
        artifact_root,
        str(updated["session_id"]),
        {
            "event": "arena_verdict_saved",
            "created_at": updated["updated_at"],
            "winner": winner,
        },
    )
    return updated


def _render_download_failures(session: dict[str, Any]) -> None:
    failures = session.get("download_failures", [])
    if not failures:
        return
    st.warning(f"{len(failures)} Swift objects failed to download.")
    st.dataframe(failures, use_container_width=True, hide_index=True)


def _render_evidence_tab(session: dict[str, Any]) -> None:
    evidence = session.get("evidence", [])
    if not evidence:
        st.info("No model evidence recorded.")
        return
    for index, item in enumerate(evidence):
        label = _format_attachment(item)
        st.code(item.get("excerpt", ""), language=None)
        if st.button(f"Attach {label}", key=f"attach-evidence-{index}"):
            _add_attachment(
                {
                    "path": item.get("path", ""),
                    "line": item.get("line"),
                    "text": item.get("excerpt", ""),
                }
            )
            st.rerun()


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
        st.dataframe(analysis["repeated_reads"], use_container_width=True, hide_index=True)
    if analysis["rows"]:
        st.dataframe(analysis["rows"], use_container_width=True, hide_index=True)
    else:
        st.info("No tool activity recorded.")


def _render_saved_progress_tab(session: dict[str, Any]) -> None:
    events = [
        event
        for event in session.get("progress_events", [])
        if isinstance(event, dict)
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
    contenders = sorted(
        {
            str(event.get("contender_id"))
            for event in arena_events
            if event.get("contender_id")
        }
    )
    if contenders:
        columns = st.columns(len(contenders))
        for column, contender_id in zip(columns, contenders, strict=False):
            contender_events = [
                event for event in arena_events if event.get("contender_id") == contender_id
            ]
            status = contender_events[-1].get("status", "queued") if contender_events else "queued"
            phase = contender_events[-1].get("phase", "") if contender_events else ""
            column.metric(f"Contender {contender_id}", status, phase)

    rows = [
        {
            "time": event.get("created_at", ""),
            "phase": event.get("phase", ""),
            "status": event.get("status", ""),
            "contender": event.get("contender_id", ""),
            "message": event.get("message", ""),
            "tool": event.get("tool_name", ""),
            "target": event.get("target", ""),
            "chars": event.get("result_chars", ""),
        }
        for event in events[-25:]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Raw progress trace"):
        st.json(events, expanded=False)


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
    root = Path(session.get("artifact_root", config.paths.artifact_root / session["uuid"]))
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
    report = _report_from_session(session)
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
                _add_attachment(
                    {
                        "path": selected.as_posix(),
                        "line": line_number,
                        "text": line_text,
                    }
                )
            st.rerun()
    st.markdown(PREVIEW_CSS, unsafe_allow_html=True)
    st.markdown(render_line_preview(preview.text, highlights), unsafe_allow_html=True)


def _add_attachment(item: dict[str, Any]) -> None:
    attachments = st.session_state.setdefault("pending_attachments", [])
    if item not in attachments:
        attachments.append(item)


def _session_from_diagnosis(
    *,
    uuid: str,
    model: str,
    artifact_root: Path,
    output: Path,
    failed_step: str,
    report: DiagnosisReport,
    exchanges: list[dict[str, Any]],
    download_failures: list[dict[str, Any]],
    progress_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "model": model,
        "summary": report.summary,
        "failure_surface": report.failure_surface,
        "confidence": report.confidence,
        "root_cause": report.root_cause,
        "needs_more_evidence": report.needs_more_evidence,
        "failed_step": failed_step,
        "updated_at": _now(),
        "artifact_root": str(artifact_root),
        "output": str(output),
        "evidence": [asdict(item) for item in report.evidence],
        "candidate_mechanisms": [asdict(item) for item in report.candidate_mechanisms],
        "recommendations": report.recommendations,
        "unknowns": report.unknowns,
        "triage_confidence": report.triage_confidence,
        "failure_timeline": [asdict(item) for item in report.failure_timeline],
        "cascading_errors": [asdict(item) for item in report.cascading_errors],
        "alternatives_considered": [
            asdict(item) for item in report.alternatives_considered
        ],
        "missing_evidence": report.missing_evidence,
        "stop_reason": report.stop_reason,
        "chat": [],
        "exchanges": exchanges,
        "download_failures": download_failures,
        "progress_events": list(progress_events or []),
    }


def _append_progress_event(events: list[dict[str, Any]], event: ProgressEvent) -> None:
    events.append(event.to_trace())


def _persist_diagnosis_session(
    artifact_root: Path,
    session: dict[str, Any],
) -> None:
    save_ui_session(artifact_root, session)
    snapshot = {
        **session,
        "schema_version": 2,
        "session_id": str(session["uuid"]),
        "session_type": "diagnosis",
        "status": "error" if session.get("error") else "completed",
    }
    save_session_snapshot(artifact_root, snapshot)


def _report_from_session(session: dict[str, Any]) -> DiagnosisReport:
    return DiagnosisReport.from_dict(
        {
            "summary": session.get("summary", ""),
            "failure_surface": session.get("failure_surface", ""),
            "confidence": session.get("confidence", "unknown"),
            "root_cause": session.get("root_cause", ""),
            "needs_more_evidence": session.get("needs_more_evidence", False),
            "evidence": session.get("evidence", []),
            "candidate_mechanisms": session.get("candidate_mechanisms", []),
            "recommendations": session.get("recommendations", []),
            "unknowns": session.get("unknowns", []),
            "triage_confidence": session.get("triage_confidence", "unknown"),
            "failure_timeline": session.get("failure_timeline", []),
            "cascading_errors": session.get("cascading_errors", []),
            "alternatives_considered": session.get("alternatives_considered", []),
            "missing_evidence": session.get("missing_evidence", []),
            "stop_reason": session.get("stop_reason", ""),
        }
    )


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


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
