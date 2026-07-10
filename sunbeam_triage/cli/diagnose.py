from __future__ import annotations

import argparse
import json
import sys

from ..core.config import Config
from ..core.llm import DiagnosisReport
from ..core.progress import ProgressEvent
from ..core.use_cases import DiagnosisRunRequest, TriageUseCases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a Sunbeam CI failure UUID.")
    parser.add_argument("uuid", help="Solutions Run UUID")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--output", help="Output HTML path or pattern")
    parser.add_argument("--artifact-root", help="Local artifact cache root")
    parser.add_argument("--model", help="OpenRouter model override")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-download cached objects"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use already mirrored artifacts; do not fetch Swift objects",
    )
    parser.add_argument(
        "--llm-json",
        help="Use a precomputed diagnosis JSON payload instead of calling OpenRouter",
    )
    parser.add_argument(
        "--budget",
        choices=["quick", "default", "hard"],
        default="default",
        help="Triage tool-round budget profile",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        help="Override the selected triage budget round count",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _log("stage", "config")
    config = Config.load(
        args.config,
        cli_model=args.model,
        cli_output=args.output,
        cli_artifact_root=args.artifact_root,
    )
    _log("model", config.llm.model)
    uuid = args.uuid
    if args.offline:
        _log("stage", "mirror skipped (offline)")
    else:
        _log("stage", f"mirror start uuid={uuid}")

    precomputed_report = None
    if args.llm_json:
        _log("stage", "diagnosis using supplied JSON")
        precomputed_report = DiagnosisReport.from_dict(json.loads(args.llm_json))
    else:
        _log("stage", f"diagnosis requesting model={config.llm.model}")

    progress_events: list[dict[str, object]] = []

    def show_progress(event: ProgressEvent) -> None:
        progress_events.append(event.to_trace())
        if event.phase == "evidence" and event.status == "running":
            _log("stage", f"evidence root={config.paths.artifact_root / uuid}")

    result = TriageUseCases(config).run_diagnosis(
        DiagnosisRunRequest(
            uuid=uuid,
            model=config.llm.model,
            budget=args.budget,
            offline=args.offline,
            refresh=args.refresh,
            max_tool_rounds=args.max_tool_rounds,
            precomputed_report=precomputed_report,
        ),
        progress=show_progress,
        progress_events=progress_events,
    )
    if result.error:
        _log("error", result.error)
        return 1

    _log(
        "result",
        (
            f"failed_step={result.session.get('failed_step', '')} "
            f"family={result.failed_step_family} "
            f"evidence_items={result.evidence_item_count}"
        )
    )
    _log(
        "result",
        f"confidence={result.session['confidence']} summary={result.session['summary']}",
    )

    output = result.session["output"]
    _log("stage", f"render output={output}")
    print(output)
    return 0


def _log(kind: str, message: str) -> None:
    print(f"[{kind}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
