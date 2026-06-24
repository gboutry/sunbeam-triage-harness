from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from sunbeam_triage.config import Config
from sunbeam_triage.evidence import EvidenceCollector
from sunbeam_triage.llm import DiagnosisReport, OpenRouterClient
from sunbeam_triage.render import render_html
from sunbeam_triage.swift import SwiftMirror
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

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        _render_diagnosis_chat(config, session)
    with right:
        _render_context_panel(config, session)


def _initialize_state(config: Config) -> None:
    st.session_state.setdefault("selected_uuid", "")
    st.session_state.setdefault("pending_attachments", [])
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
            submitted = st.form_submit_button("Start diagnosis", type="primary")
        if submitted:
            _start_diagnosis(config, uuid.strip(), model.strip() or config.llm.model)

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
            if st.button(label, key=f"history-{item['uuid']}", use_container_width=True):
                st.session_state["selected_uuid"] = item["uuid"]

    return st.session_state.get("selected_uuid", "")


def _start_diagnosis(config: Config, uuid: str, model: str) -> None:
    if not uuid:
        st.sidebar.error("Enter a Solutions Run UUID.")
        return

    run_config = Config.load("config.toml", cli_model=model)
    artifact_root = run_config.paths.artifact_root / uuid
    llm_client = OpenRouterClient(run_config.llm)

    with st.status(f"Diagnosing {uuid}", expanded=True) as status:
        download_failures: list[dict[str, Any]] = []
        try:
            st.write("Downloading artifacts")
            progress_area = st.empty()

            def show_download(event: dict[str, Any]) -> None:
                error = ""
                if event.get("error"):
                    error = f"\n\nError: `{event['error']}`"
                progress_area.write(
                    (
                        f"{event['status']} {event['index']}/{event['total']}: "
                        f"`{event['name']}`\n\n"
                        f"Destination: `{event['path']}`\n\n"
                        f"URL: `{event['url']}`"
                        f"{error}"
                    )
                )

            manifest = SwiftMirror(run_config.swift, run_config.paths.artifact_root).mirror_uuid(
                uuid,
                progress=show_download,
                continue_on_error=True,
            )
            st.write(f"Downloaded or reused {len(manifest.objects)} objects.")
            if manifest.failures:
                download_failures = [asdict(item) for item in manifest.failures]
                st.warning(f"{len(manifest.failures)} Swift objects failed to download.")
                st.dataframe(
                    [
                        {
                            "object": failure.name,
                            "path": failure.path,
                            "url": failure.url,
                            "error": failure.error,
                        }
                        for failure in manifest.failures
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            st.write("Collecting evidence")
            pack = EvidenceCollector(artifact_root, uuid).collect()

            st.write(f"Querying {run_config.llm.model}")
            report = llm_client.diagnose(
                pack.to_prompt_text(),
                session_id=uuid,
            )

            output = run_config.output_path(uuid)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_html(pack, report), encoding="utf-8")

            save_ui_session(
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
                ),
            )
            st.session_state["selected_uuid"] = uuid
            st.session_state["pending_attachments"] = []
            status.update(label="Diagnosis complete", state="complete")
        except Exception as exc:
            status.update(label="Diagnosis failed", state="error")
            save_ui_session(
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
        _send_followup(config, session, prompt)
        st.rerun()


def _send_followup(config: Config, session: dict[str, Any], prompt: str) -> None:
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
    answer = llm_client.chat(context, messages, session_id=session["uuid"])

    session.setdefault("chat", []).extend(
        [
            {"role": "user", "content": prompt, "created_at": _now()},
            {"role": "assistant", "content": answer, "created_at": _now()},
        ]
    )
    session.setdefault("exchanges", []).extend(llm_client.exchanges)
    session["updated_at"] = _now()
    save_ui_session(config.paths.artifact_root, session)
    st.session_state["pending_attachments"] = []


def _render_context_panel(config: Config, session: dict[str, Any] | None) -> None:
    st.subheader("Context")
    if not session:
        st.info("Select a diagnosis to inspect context.")
        return
    tabs = st.tabs(["Evidence", "Files", "API"])
    with tabs[0]:
        _render_evidence_tab(session)
    with tabs[1]:
        _render_files_tab(config, session)
    with tabs[2]:
        st.json(session.get("exchanges", []), expanded=False)


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
) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "model": model,
        "summary": report.summary,
        "failure_surface": report.failure_surface,
        "confidence": report.confidence,
        "root_cause": report.root_cause,
        "failed_step": failed_step,
        "updated_at": _now(),
        "artifact_root": str(artifact_root),
        "output": str(output),
        "evidence": [asdict(item) for item in report.evidence],
        "candidate_mechanisms": [asdict(item) for item in report.candidate_mechanisms],
        "recommendations": report.recommendations,
        "unknowns": report.unknowns,
        "chat": [],
        "exchanges": exchanges,
        "download_failures": download_failures,
    }


def _report_from_session(session: dict[str, Any]) -> DiagnosisReport:
    return DiagnosisReport.from_dict(
        {
            "summary": session.get("summary", ""),
            "failure_surface": session.get("failure_surface", ""),
            "confidence": session.get("confidence", "unknown"),
            "root_cause": session.get("root_cause", ""),
            "evidence": session.get("evidence", []),
            "candidate_mechanisms": session.get("candidate_mechanisms", []),
            "recommendations": session.get("recommendations", []),
            "unknowns": session.get("unknowns", []),
        }
    )


def _format_attachment(item: dict[str, Any]) -> str:
    line = "" if item.get("line") is None else f":{item['line']}"
    return f"{item.get('path', '')}{line}"


def _line_at(text: str, number: int) -> str:
    lines = text.splitlines()
    if number < 1 or number > len(lines):
        return ""
    return lines[number - 1].strip()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
