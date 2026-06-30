from __future__ import annotations

import string
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from .config import Config, LlmConfig
from .evidence import EvidenceCollector
from .llm import OpenRouterClient
from .progress import ProgressEvent, ProgressSink, emit_progress
from .sessions import append_session_event, save_session_snapshot
from .tool_activity import analyze_tool_activity
from .triage_state import (
    BudgetName,
    BudgetProfile,
    TriageLoopOptions,
    resolve_triage_budget,
)


@dataclass(frozen=True)
class ArenaOptions:
    models: list[str]
    budget: BudgetName = "default"
    max_tool_rounds: int | None = None
    triage_options: TriageLoopOptions | None = None
    output: str | Path | None = None


ClientFactory = Callable[[LlmConfig], Any]


class ArenaRunner:
    def __init__(self, config: Config, *, client_factory: ClientFactory | None = None):
        self.config = config
        self.client_factory = client_factory or OpenRouterClient

    def run(
        self,
        uuid: str,
        options: ArenaOptions,
        *,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        models = _normalize_models(options.models)
        if len(models) < 2:
            raise ValueError("Arena runs require at least two models.")
        artifact_root = self.config.paths.artifact_root / uuid
        pack = EvidenceCollector(artifact_root, uuid).collect()
        evidence_text = pack.to_prompt_text()
        session_id = _arena_session_id(uuid)
        now = _now()
        triage_options = options.triage_options or _resolve_triage_options(
            self.config,
            options,
        )
        session = {
            "schema_version": 2,
            "session_id": session_id,
            "session_type": "arena",
            "uuid": uuid,
            "status": "running",
            "summary": f"Arena comparison for {uuid}",
            "created_at": now,
            "updated_at": now,
            "artifact_root": str(artifact_root),
            "budget": options.budget,
            "evidence": [asdict(item) for item in pack.evidence],
            "probe_results": [result.to_dict() for result in pack.probe_results],
            "failed_step": asdict(pack.failed_step),
            "run": asdict(pack.run),
            "contenders": [],
        }
        save_session_snapshot(self.config.paths.artifact_root, session)
        emit_progress(
            progress,
            ProgressEvent(
                run_id=session_id,
                run_type="arena",
                phase="arena_started",
                status="running",
                message=f"Arena started with {len(models)} contenders",
            ),
        )
        append_session_event(
            self.config.paths.artifact_root,
            session_id,
            {
                "event": "arena_started",
                "created_at": now,
                "uuid": uuid,
                "models": models,
            },
        )

        for index, model in enumerate(models):
            contender_id = _contender_id(index)
            append_session_event(
                self.config.paths.artifact_root,
                session_id,
                {
                    "event": "contender_started",
                    "created_at": _now(),
                    "contender_id": contender_id,
                    "model": model,
                },
            )
            emit_progress(
                progress,
                ProgressEvent(
                    run_id=session_id,
                    run_type="arena",
                    phase="contender_started",
                    status="running",
                    message=f"Contender {contender_id} started",
                    contender_id=contender_id,
                ),
            )
            contender = self._run_contender(
                model,
                contender_id,
                evidence_text=evidence_text,
                session_id=session_id,
                artifact_root=artifact_root,
                triage_options=triage_options,
                progress=progress,
            )
            session["contenders"].append(contender)
            emit_progress(
                progress,
                ProgressEvent(
                    run_id=session_id,
                    run_type="arena",
                    phase=(
                        "contender_completed"
                        if contender["status"] == "completed"
                        else "contender_failed"
                    ),
                    status=contender["status"],
                    message=(
                        f"Contender {contender_id} completed"
                        if contender["status"] == "completed"
                        else f"Contender {contender_id} failed"
                    ),
                    contender_id=contender_id,
                    warning=contender.get("error"),
                ),
            )
            append_session_event(
                self.config.paths.artifact_root,
                session_id,
                {
                    "event": (
                        "contender_completed"
                        if contender["status"] == "completed"
                        else "contender_failed"
                    ),
                    "created_at": _now(),
                    "contender_id": contender_id,
                    "model": model,
                    "status": contender["status"],
                },
            )
            session["updated_at"] = _now()
            save_session_snapshot(self.config.paths.artifact_root, session)

        completed = [
            item for item in session["contenders"] if item.get("status") == "completed"
        ]
        session["status"] = (
            "completed"
            if len(completed) == len(session["contenders"])
            else "completed_with_errors"
            if completed
            else "failed"
        )
        session["summary"] = _arena_summary(session)
        output = (
            Path(options.output)
            if options.output
            else _arena_output_path(uuid, session_id)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_arena_html(session), encoding="utf-8")
        session["output"] = str(output)
        session["updated_at"] = _now()
        save_session_snapshot(self.config.paths.artifact_root, session)
        emit_progress(
            progress,
            ProgressEvent(
                run_id=session_id,
                run_type="arena",
                phase="arena_completed",
                status=session["status"],
                message=session["summary"],
            ),
        )
        append_session_event(
            self.config.paths.artifact_root,
            session_id,
            {
                "event": "arena_completed",
                "created_at": session["updated_at"],
                "status": session["status"],
                "contenders": len(session["contenders"]),
            },
        )
        return session

    def retry_failed(
        self,
        session: dict[str, Any],
        options: ArenaOptions,
        *,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        uuid = str(session["uuid"])
        artifact_root = self.config.paths.artifact_root / uuid
        pack = EvidenceCollector(artifact_root, uuid).collect()
        evidence_text = pack.to_prompt_text()
        triage_options = options.triage_options or _resolve_triage_options(
            self.config,
            options,
        )
        updated = dict(session)
        updated["contenders"] = [dict(item) for item in session.get("contenders", [])]
        session_id = str(updated["session_id"])
        for index, contender in enumerate(updated["contenders"]):
            if contender.get("status") != "failed":
                continue
            contender_id = str(contender["contender_id"])
            model = str(contender["model"])
            emit_progress(
                progress,
                ProgressEvent(
                    run_id=session_id,
                    run_type="arena",
                    phase="contender_started",
                    status="running",
                    message=f"Contender {contender_id} started",
                    contender_id=contender_id,
                ),
            )
            replacement = self._run_contender(
                model,
                contender_id,
                evidence_text=evidence_text,
                session_id=session_id,
                artifact_root=artifact_root,
                triage_options=triage_options,
                progress=progress,
            )
            updated["contenders"][index] = replacement
            emit_progress(
                progress,
                ProgressEvent(
                    run_id=session_id,
                    run_type="arena",
                    phase=(
                        "contender_completed"
                        if replacement["status"] == "completed"
                        else "contender_failed"
                    ),
                    status=replacement["status"],
                    message=(
                        f"Contender {contender_id} completed"
                        if replacement["status"] == "completed"
                        else f"Contender {contender_id} failed"
                    ),
                    contender_id=contender_id,
                    warning=replacement.get("error"),
                ),
            )
        completed = [
            item for item in updated["contenders"] if item.get("status") == "completed"
        ]
        updated["status"] = (
            "completed"
            if len(completed) == len(updated["contenders"])
            else "completed_with_errors"
            if completed
            else "failed"
        )
        updated["summary"] = _arena_summary(updated)
        output = Path(updated.get("output") or _arena_output_path(uuid, session_id))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_arena_html(updated), encoding="utf-8")
        updated["output"] = str(output)
        updated["updated_at"] = _now()
        save_session_snapshot(self.config.paths.artifact_root, updated)
        append_session_event(
            self.config.paths.artifact_root,
            session_id,
            {
                "event": "arena_retry_completed",
                "created_at": updated["updated_at"],
                "status": updated["status"],
            },
        )
        emit_progress(
            progress,
            ProgressEvent(
                run_id=session_id,
                run_type="arena",
                phase="arena_completed",
                status=updated["status"],
                message=updated["summary"],
            ),
        )
        return updated

    def _run_contender(
        self,
        model: str,
        contender_id: str,
        *,
        evidence_text: str,
        session_id: str,
        artifact_root: Path,
        triage_options: TriageLoopOptions,
        progress: ProgressSink | None = None,
    ) -> dict[str, Any]:
        llm_config = _llm_config_for_model(self.config.llm, model)
        client = self.client_factory(llm_config)
        contender_session_id = f"{session_id}-{contender_id}"
        base = {
            "contender_id": contender_id,
            "model": model,
            "started_at": _now(),
        }
        try:
            emit_progress(
                progress,
                ProgressEvent(
                    run_id=session_id,
                    run_type="arena",
                    phase="model_request",
                    status="running",
                    message=f"Contender {contender_id} request sent",
                    contender_id=contender_id,
                ),
            )
            report = client.diagnose(
                evidence_text,
                session_id=contender_session_id,
                artifact_root=artifact_root,
                max_tool_rounds=triage_options.max_rounds,
                max_tool_result_chars=triage_options.max_tool_result_chars,
                triage_options=triage_options,
                progress=_arena_progress_proxy(progress),
                run_type="arena",
                contender_id=contender_id,
            )
        except Exception as exc:
            return {
                **base,
                "status": "failed",
                "completed_at": _now(),
                "error": str(exc),
                "exchanges": list(getattr(client, "exchanges", [])),
            }
        emit_progress(
            progress,
            ProgressEvent(
                run_id=session_id,
                run_type="arena",
                phase="completed",
                status="completed",
                message=f"Contender {contender_id} response complete",
                contender_id=contender_id,
            ),
        )
        contender = {
            **base,
            "status": "completed",
            "completed_at": _now(),
            "report": asdict(report),
            "exchanges": list(getattr(client, "exchanges", [])),
        }
        contender["tool_activity"] = analyze_tool_activity({
            "uuid": contender_session_id,
            "model": model,
            "exchanges": contender["exchanges"],
        })
        contender["trace_path"] = f".sunbeam-triage/sessions/{session_id}.json"
        return contender


def render_arena_html(session: dict[str, Any]) -> str:
    reveal_models = "verdict" in session
    body = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Arena: {escape(str(session.get('uuid', '')))}</title>",
        "<style>",
        ARENA_CSS,
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Arena: {escape(str(session.get('uuid', '')))}</h1>",
        f"<p>Status: {escape(str(session.get('status', '')))}</p>",
    ]
    verdict = session.get("verdict")
    if isinstance(verdict, dict):
        body.append(
            f"<section><h2>Verdict</h2><p>Winner: "
            f"{escape(str(verdict.get('winner', '')))}</p>"
            f"<p>{escape(str(verdict.get('notes', '')))}</p></section>"
        )
    body.append("<section><h2>Contenders</h2>")
    body.extend(
        _contender_html(contender, reveal_models=reveal_models)
        for contender in session.get("contenders", [])
    )
    body.extend(["</section>", "</body>", "</html>"])
    return "\n".join(body)


def _contender_html(contender: dict[str, Any], *, reveal_models: bool) -> str:
    contender_id = str(contender.get("contender_id", ""))
    title = f"Contender {contender_id}"
    if reveal_models:
        title += f" - {contender.get('model', '')}"
    report = contender.get("report", {})
    if not isinstance(report, dict):
        report = {}
    return "\n".join([
        '<article class="contender">',
        f"<h3>{escape(title)}</h3>",
        f"<p>Status: {escape(str(contender.get('status', '')))}</p>",
        f"<p><strong>Summary:</strong> {escape(str(report.get('summary', '')))}</p>",
        (
            f"<p><strong>Root cause:</strong> "
            f"{escape(str(report.get('root_cause', '')))}</p>"
        ),
        (
            f"<p><strong>Confidence:</strong> "
            f"{escape(str(report.get('confidence', '')))}</p>"
        ),
        f"<p>{escape(str(contender.get('error', '')))}</p>"
        if contender.get("error")
        else "",
        "</article>",
    ])


def _resolve_triage_options(config: Config, options: ArenaOptions) -> TriageLoopOptions:
    return resolve_triage_budget(
        BudgetProfile(
            quick_max_rounds=config.triage.quick_max_rounds,
            default_max_rounds=config.triage.default_max_rounds,
            hard_max_rounds=config.triage.hard_max_rounds,
            stall_limit=config.triage.stall_limit,
            min_evidence_items=config.triage.min_evidence_items,
            max_tool_result_chars=config.triage.max_tool_result_chars,
        ),
        budget=options.budget,
        max_tool_rounds=options.max_tool_rounds,
    )


def _llm_config_for_model(config: LlmConfig, model: str) -> LlmConfig:
    return LlmConfig(
        base_url=config.base_url,
        model=model,
        api_key_env=config.api_key_env,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
    )


def _normalize_models(models: list[str]) -> list[str]:
    return [model.strip() for model in models if model.strip()]


def _contender_id(index: int) -> str:
    alphabet = string.ascii_uppercase
    if index < len(alphabet):
        return alphabet[index]
    return f"M{index + 1}"


def _arena_session_id(uuid: str) -> str:
    return f"arena-{uuid}-{_now().replace(':', '').replace('-', '')}"


def _arena_output_path(uuid: str, session_id: str) -> Path:
    return Path(f"arena-{uuid}-{session_id}.html")


def _arena_summary(session: dict[str, Any]) -> str:
    completed = sum(
        1
        for contender in session["contenders"]
        if contender.get("status") == "completed"
    )
    return f"{completed}/{len(session['contenders'])} contenders completed"


def _arena_progress_proxy(progress: ProgressSink | None) -> ProgressSink | None:
    if progress is None:
        return None

    def proxy(event: ProgressEvent) -> None:
        if event.phase not in {"model_request", "completed"}:
            progress(event)

    return proxy


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


ARENA_CSS = """
body {
  color: #1f2937;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
  margin: 2rem auto;
  max-width: 1100px;
  padding: 0 1rem;
}
.contender {
  border: 1px solid #d1d5db;
  border-radius: 8px;
  margin: 1rem 0;
  padding: 1rem;
}
""".strip()
