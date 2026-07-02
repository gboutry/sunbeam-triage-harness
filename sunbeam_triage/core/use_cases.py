from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .arena import ArenaOptions, ArenaRunner
from .config import Config, LlmConfig
from .evidence import EvidenceCollector
from .llm import DiagnosisReport, OpenRouterClient
from .progress import ProgressEvent, ProgressSink, emit_progress
from .redaction import redact_data, redact_text
from .render import render_html
from .sessions import append_session_event, load_session_record, save_session_snapshot
from .swift import SwiftConfig, SwiftMirror
from .triage_state import (
    BudgetName,
    BudgetProfile,
    parse_budget_name,
    resolve_triage_budget,
)


@dataclass(frozen=True)
class DiagnosisRunRequest:
    uuid: str
    model: str
    budget: str = "default"


@dataclass(frozen=True)
class DiagnosisRunResult:
    session: dict[str, Any]
    selected_uuid: str
    clear_attachments: bool = True
    error: str | None = None


@dataclass(frozen=True)
class FollowupRequest:
    session: dict[str, Any]
    prompt: str
    attachments: list[dict[str, Any]]


@dataclass(frozen=True)
class FollowupResult:
    session: dict[str, Any]
    answer: str
    clear_attachments: bool = True


@dataclass(frozen=True)
class ArenaRunRequest:
    uuid: str
    models: list[str]
    budget: str = "default"


@dataclass(frozen=True)
class ArenaRetryRequest:
    session_id: str


@dataclass(frozen=True)
class ArenaRunResult:
    session: dict[str, Any]
    selected_arena_id: str


@dataclass(frozen=True)
class ArenaVerdictRequest:
    session: dict[str, Any]
    winner: str
    notes: str
    rubric: dict[str, dict[str, int]]


ClientFactory = Callable[[LlmConfig], Any]
MirrorFactory = Callable[[SwiftConfig, Path], Any]
EvidenceCollectorFactory = Callable[[Path, str], Any]


class TriageUseCases:
    def __init__(
        self,
        config: Config,
        *,
        client_factory: ClientFactory | None = None,
        mirror_factory: MirrorFactory | None = None,
        evidence_collector_factory: EvidenceCollectorFactory | None = None,
    ):
        self.config = config
        self.client_factory = client_factory or OpenRouterClient
        self.mirror_factory = mirror_factory or SwiftMirror
        self.evidence_collector_factory = (
            evidence_collector_factory or EvidenceCollector
        )

    def run_diagnosis(
        self,
        request: DiagnosisRunRequest,
        *,
        progress: ProgressSink | None = None,
        progress_events: list[dict[str, Any]] | None = None,
    ) -> DiagnosisRunResult:
        if not request.uuid:
            raise ValueError("Enter a Solutions Run UUID.")
        run_config = _config_with_model(self.config, request.model)
        uuid = request.uuid
        artifact_root = run_config.paths.artifact_root / uuid
        llm_client = self.client_factory(run_config.llm)
        download_failures: list[dict[str, Any]] = []

        try:
            _emit_use_case_progress(
                progress,
                uuid,
                "diagnosis",
                "download",
                "running",
                "Downloading artifacts",
            )

            def show_download(event: dict[str, Any]) -> None:
                _emit_use_case_progress(
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

            manifest = self.mirror_factory(
                run_config.swift,
                run_config.paths.artifact_root,
            ).mirror_uuid(
                uuid,
                progress=show_download,
                continue_on_error=True,
            )
            _emit_use_case_progress(
                progress,
                uuid,
                "diagnosis",
                "download",
                "completed",
                f"Downloaded or reused {len(manifest.objects)} objects",
            )
            if manifest.failures:
                download_failures = [asdict(item) for item in manifest.failures]

            _emit_use_case_progress(
                progress,
                uuid,
                "diagnosis",
                "evidence",
                "running",
                "Collecting evidence",
            )
            pack = self.evidence_collector_factory(artifact_root, uuid).collect()
            _emit_use_case_progress(
                progress,
                uuid,
                "diagnosis",
                "probe",
                "completed",
                f"Ran {len(pack.probe_results)} deterministic probes",
            )

            triage_options = _resolve_options(run_config, request.budget)
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

            session = session_from_diagnosis(
                uuid=uuid,
                model=run_config.llm.model,
                artifact_root=artifact_root,
                output=output,
                failed_step=pack.failed_step.name,
                report=report,
                exchanges=llm_client.exchanges,
                download_failures=download_failures,
                probe_results=pack.probe_results,
                progress_events=progress_events,
            )
            persist_diagnosis_session(run_config.paths.artifact_root, session)
            return DiagnosisRunResult(session=session, selected_uuid=uuid)
        except Exception as exc:
            _emit_use_case_progress(
                progress,
                uuid,
                "diagnosis",
                "failed",
                "failed",
                str(exc),
                warning=str(exc),
            )
            session = {
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
            }
            persist_diagnosis_session(run_config.paths.artifact_root, session)
            return DiagnosisRunResult(
                session=session,
                selected_uuid=uuid,
                error=str(exc),
                clear_attachments=False,
            )

    def send_followup(
        self,
        request: FollowupRequest,
        *,
        progress: ProgressSink | None = None,
        progress_events: list[dict[str, Any]] | None = None,
    ) -> FollowupResult:
        session = dict(request.session)
        pack = self.evidence_collector_factory(
            Path(session["artifact_root"]),
            session["uuid"],
        ).collect()
        report = report_from_session(session)
        context = build_followup_context(
            pack,
            report,
            attachments=request.attachments,
        )
        run_config = _config_with_model(self.config, str(session["model"]))
        llm_client = self.client_factory(run_config.llm)
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in session.get("chat", [])
            if item.get("role") in {"user", "assistant"}
        ]
        messages = [*history, {"role": "user", "content": request.prompt}]
        answer = llm_client.chat(
            context,
            messages,
            session_id=session["uuid"],
            artifact_root=Path(session["artifact_root"]),
            progress=progress,
        )

        session.setdefault("chat", []).extend([
            {"role": "user", "content": request.prompt, "created_at": _now()},
            {"role": "assistant", "content": answer, "created_at": _now()},
        ])
        session.setdefault("exchanges", []).extend(llm_client.exchanges)
        session.setdefault("progress_events", []).extend(progress_events or [])
        session["updated_at"] = _now()
        persist_diagnosis_session(self.config.paths.artifact_root, session)
        return FollowupResult(session=session, answer=answer)

    def run_arena(
        self,
        request: ArenaRunRequest,
        *,
        progress: ProgressSink | None = None,
        progress_events: list[dict[str, Any]] | None = None,
    ) -> ArenaRunResult:
        if not request.uuid:
            raise ValueError("Enter a Solutions Run UUID.")
        if len(request.models) < 2:
            raise ValueError("Enter at least two contender models.")
        _emit_use_case_progress(
            progress,
            request.uuid,
            "arena",
            "download",
            "running",
            "Downloading artifacts",
        )
        self.mirror_factory(
            self.config.swift, self.config.paths.artifact_root
        ).mirror_uuid(
            request.uuid,
            continue_on_error=True,
        )
        _emit_use_case_progress(
            progress,
            request.uuid,
            "arena",
            "arena_running",
            "running",
            f"Running {len(request.models)} contenders",
        )
        session = ArenaRunner(self.config, client_factory=self.client_factory).run(
            request.uuid,
            ArenaOptions(
                models=request.models,
                budget=_parse_budget(request.budget),
            ),
            progress=progress,
        )
        session["progress_events"] = list(progress_events or [])
        save_session_snapshot(self.config.paths.artifact_root, session)
        return ArenaRunResult(
            session=session,
            selected_arena_id=str(session["session_id"]),
        )

    def retry_failed_arena(
        self,
        request: ArenaRetryRequest,
        *,
        progress: ProgressSink | None = None,
        progress_events: list[dict[str, Any]] | None = None,
    ) -> ArenaRunResult:
        loaded = load_session_record(
            self.config.paths.artifact_root, request.session_id
        )
        if not loaded:
            raise ValueError("Arena record is missing.")
        arena = loaded["snapshot"]
        models = [str(item.get("model", "")) for item in arena.get("contenders", [])]
        updated = ArenaRunner(
            self.config, client_factory=self.client_factory
        ).retry_failed(
            arena,
            ArenaOptions(
                models=models,
                budget=_parse_budget(str(arena.get("budget", "default"))),
            ),
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
        save_session_snapshot(self.config.paths.artifact_root, updated)
        return ArenaRunResult(
            session=updated,
            selected_arena_id=str(updated["session_id"]),
        )

    def save_arena_verdict(self, request: ArenaVerdictRequest) -> dict[str, Any]:
        updated = dict(request.session)
        updated["status"] = "judged"
        updated["updated_at"] = _now()
        updated["verdict"] = {
            "winner": request.winner,
            "notes": request.notes,
            "rubric": request.rubric,
            "created_at": _now(),
        }
        save_session_snapshot(self.config.paths.artifact_root, updated)
        append_session_event(
            self.config.paths.artifact_root,
            str(updated["session_id"]),
            {
                "event": "arena_verdict_saved",
                "created_at": updated["updated_at"],
                "winner": request.winner,
            },
        )
        return updated


def session_from_diagnosis(
    *,
    uuid: str,
    model: str,
    artifact_root: Path,
    output: Path,
    failed_step: str,
    report: DiagnosisReport,
    exchanges: list[dict[str, Any]],
    download_failures: list[dict[str, Any]],
    probe_results: Any | None = None,
    progress_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return redact_data({
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
        "probe_results": [
            result.to_dict() if hasattr(result, "to_dict") else result
            for result in list(probe_results or [])
        ],
        "progress_events": list(progress_events or []),
    })


def persist_diagnosis_session(artifact_root: Path, session: dict[str, Any]) -> None:
    snapshot = {
        **session,
        "schema_version": 2,
        "session_id": str(session["uuid"]),
        "session_type": "diagnosis",
        "status": "error" if session.get("error") else "completed",
    }
    save_session_snapshot(artifact_root, snapshot)


def report_from_session(session: dict[str, Any]) -> DiagnosisReport:
    return DiagnosisReport.from_dict({
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
    })


def build_followup_context(
    pack,
    report: DiagnosisReport,
    *,
    attachments: list[dict[str, Any]] | None = None,
) -> str:
    parts = [
        "You are answering a follow-up question about this active diagnosis.",
        f"Solutions Run UUID: {pack.uuid}",
        f"Run ID: {pack.run.run_id}",
        f"Branch: {pack.run.branch}",
        f"Workflow: {pack.run.workflow}",
        f"Failed Step: {pack.failed_step.name}",
        f"Diagnosis Summary: {report.summary}",
        f"Failure Surface: {report.failure_surface}",
        f"Confidence: {report.confidence}",
        f"Triage Confidence: {report.triage_confidence}",
        f"Stop Reason: {report.stop_reason}",
        f"Root Cause: {report.root_cause}",
        "",
        "Model Evidence:",
    ]
    for item in report.evidence:
        line = "" if item.line is None else f":{item.line}"
        parts.append(f"- {item.path}{line}: {item.excerpt}")
    parts.extend(["", "Harness Evidence:"])
    for item in pack.evidence:
        line = "" if item.line is None else f":{item.line}"
        parts.append(f"- [{item.kind}] {item.path}{line}: {item.excerpt}")
    probe_lines = _probe_context_lines(pack)
    if probe_lines:
        parts.extend(["", "Deterministic Probes:", *probe_lines])
    if report.recommendations:
        parts.extend(["", "Recommendations:"])
        parts.extend(f"- {item}" for item in report.recommendations)
    if report.unknowns:
        parts.extend(["", "Unknowns:"])
        parts.extend(f"- {item}" for item in report.unknowns)
    if report.failure_timeline:
        parts.extend(["", "Failure Timeline:"])
        parts.extend(
            (f"- {item.timestamp} {item.source} {item.location}: {item.event}")
            for item in report.failure_timeline
        )
    if report.cascading_errors:
        parts.extend(["", "Cascading Errors:"])
        for item in report.cascading_errors:
            line = "" if item.line is None else f":{item.line}"
            parts.append(f"- {item.path}{line}: {item.excerpt}")
    if report.alternatives_considered:
        parts.extend(["", "Alternatives Considered:"])
        parts.extend(
            f"- {item.hypothesis} ({item.status}): {item.reason}"
            for item in report.alternatives_considered
        )
    if report.missing_evidence:
        parts.extend(["", "Missing Evidence:"])
        parts.extend(f"- {item}" for item in report.missing_evidence)
    if attachments:
        parts.extend(["", "Attached Context:"])
        for item in attachments:
            line = "" if item.get("line") is None else f":{item['line']}"
            parts.append(f"- {item.get('path', '')}{line}: {item.get('text', '')}")
    return redact_text("\n".join(parts))


def _probe_context_lines(pack) -> list[str]:
    lines: list[str] = []
    for result in pack.probe_results:
        if result.status == "not_applicable":
            continue
        lines.append(f"- [{result.name}] {result.status}: {result.summary}")
        for finding in result.findings[:20]:
            line = "" if finding.line is None else f":{finding.line}"
            lines.append(
                f"  - [{finding.category}] {finding.path}{line}: {finding.excerpt}"
            )
        lines.extend(f"  - [missing] {missing}" for missing in result.missing_evidence)
    return lines


def _resolve_options(config: Config, budget: str):
    return resolve_triage_budget(
        BudgetProfile(
            quick_max_rounds=config.triage.quick_max_rounds,
            default_max_rounds=config.triage.default_max_rounds,
            hard_max_rounds=config.triage.hard_max_rounds,
            stall_limit=config.triage.stall_limit,
            min_evidence_items=config.triage.min_evidence_items,
            max_tool_result_chars=config.triage.max_tool_result_chars,
        ),
        budget=_parse_budget(budget),
    )


def _parse_budget(value: str) -> BudgetName:
    return parse_budget_name(value)


def _config_with_model(config: Config, model: str) -> Config:
    updated = deepcopy(config)
    if model:
        updated.llm.model = model
    return updated


def _emit_use_case_progress(
    progress: ProgressSink | None,
    run_id: str,
    run_type: str,
    phase: str,
    status: str,
    message: str,
    *,
    warning: str | None = None,
) -> None:
    emit_progress(
        progress,
        ProgressEvent(
            run_id=run_id,
            run_type=run_type,
            phase=phase,
            status=status,
            message=message,
            warning=warning,
        ),
    )


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
