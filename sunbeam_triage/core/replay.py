from __future__ import annotations

import json
import signal
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Config, LlmConfig
from .evaluation import EvaluationCase, manifest_sha256, score_session
from .evidence import EvidenceCollector
from .llm import OpenRouterClient
from .progress import ProgressSink
from .render import render_html
from .report_policy import apply_probe_report_policies
from .report_validation import validate_causal_report
from .tool_activity import analyze_tool_activity
from .triage_state import BudgetName, BudgetProfile, resolve_triage_budget
from .use_cases import session_from_diagnosis

ClientFactory = Callable[[LlmConfig], Any]


class ReplayAttemptError(RuntimeError):
    def __init__(self, message: str, exchanges: list[dict[str, Any]]):
        super().__init__(message)
        self.exchanges = exchanges


def replay_case(
    config: Config,
    case: EvaluationCase,
    *,
    attempt: int,
    output_dir: Path,
    budget: BudgetName = "default",
    client_factory: ClientFactory = OpenRouterClient,
    progress: ProgressSink | None = None,
    timeout_seconds: int = 480,
) -> dict[str, Any]:
    artifact_root = config.paths.artifact_root / case.uuid
    actual_manifest = manifest_sha256(artifact_root)
    if actual_manifest != case.manifest_sha256:
        raise RuntimeError(
            f"Artifact manifest mismatch for {case.uuid}: "
            f"expected {case.manifest_sha256}, got {actual_manifest}"
        )
    pack = EvidenceCollector(artifact_root, case.uuid).collect()
    client = client_factory(config.llm)
    options = resolve_triage_budget(
        BudgetProfile(
            quick_max_rounds=config.triage.quick_max_rounds,
            default_max_rounds=config.triage.default_max_rounds,
            hard_max_rounds=config.triage.hard_max_rounds,
            stall_limit=config.triage.stall_limit,
            min_evidence_items=config.triage.min_evidence_items,
            max_tool_result_chars=config.triage.max_tool_result_chars,
        ),
        budget=budget,
    )
    attempt_id = f"{case.uuid}--r{attempt}"
    html_path = output_dir / f"{attempt_id}.html"
    try:
        with _deadline(timeout_seconds, attempt_id):
            report = client.diagnose(
                pack.to_prompt_text(),
                session_id=attempt_id,
                artifact_root=artifact_root,
                max_tool_rounds=options.max_rounds,
                max_tool_result_chars=options.max_tool_result_chars,
                triage_options=options,
                run_type="replay",
                progress=progress,
            )
    except Exception as exc:
        raise ReplayAttemptError(str(exc), list(client.exchanges)) from exc
    report = apply_probe_report_policies(report, pack.probe_results, pack.evidence)
    report = validate_causal_report(report, pack.probe_results)
    html_path.write_text(render_html(pack, report), encoding="utf-8")
    session = session_from_diagnosis(
        uuid=case.uuid,
        model=config.llm.model,
        artifact_root=artifact_root,
        output=html_path,
        failed_step=pack.failed_step.name,
        report=report,
        exchanges=list(client.exchanges),
        download_failures=[],
        probe_results=pack.probe_results,
    )
    session.update({
        "schema_version": 2,
        "session_id": attempt_id,
        "session_type": "replay",
        "status": "completed",
        "attempt": attempt,
        "corpus_manifest_sha256": case.manifest_sha256,
        "score": score_session(case, session),
    })
    _write_json(output_dir / f"{attempt_id}.json", session)
    return session


def replay_corpus(
    config: Config,
    cases: list[EvaluationCase],
    *,
    repetitions: int,
    output_root: Path,
    budget: BudgetName = "default",
    client_factory: ClientFactory = OpenRouterClient,
    progress: ProgressSink | None = None,
    timeout_seconds: int = 480,
) -> dict[str, Any]:
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for case in cases:
            try:
                session = replay_case(
                    config,
                    case,
                    attempt=repetition,
                    output_dir=output_dir,
                    budget=budget,
                    client_factory=client_factory,
                    progress=progress,
                    timeout_seconds=timeout_seconds,
                )
                attempts.append({
                    "uuid": case.uuid,
                    "attempt": repetition,
                    "status": "completed",
                    "score": session["score"],
                    "session": f"{case.uuid}--r{repetition}.json",
                })
            except Exception as exc:
                exchanges = exc.exchanges if isinstance(exc, ReplayAttemptError) else []
                failure = {
                    "uuid": case.uuid,
                    "attempt": repetition,
                    "status": "error",
                    "error": str(exc),
                    "exchanges": exchanges,
                }
                failure["activity"] = analyze_tool_activity(failure)
                attempts.append(failure)
                _write_json(
                    output_dir / f"{case.uuid}--r{repetition}.error.json",
                    failure,
                )
    completed = [item for item in attempts if item["status"] == "completed"]
    summary = {
        "schema": "sunbeam-triage-replay-v1",
        "run_id": run_id,
        "model": config.llm.model,
        "budget": budget,
        "repetitions": repetitions,
        "case_count": len(cases),
        "attempt_count": len(attempts),
        "completed": len(completed),
        "passed": sum(bool(item["score"]["passed"]) for item in completed),
        "attempts": attempts,
    }
    _write_json(output_dir / "summary.json", summary)
    return {**summary, "output_dir": str(output_dir)}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def _deadline(seconds: int, attempt_id: str):
    if seconds <= 0:
        yield
        return

    def timed_out(_signum, _frame):
        raise TimeoutError(f"Replay attempt {attempt_id} exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
