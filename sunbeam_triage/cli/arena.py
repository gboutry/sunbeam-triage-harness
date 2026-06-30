from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.arena import ArenaOptions, ArenaRunner
from ..core.config import Config
from ..core.llm import DiagnosisReport
from ..core.sessions import export_judged_arenas
from ..core.swift import SwiftMirror


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or export diagnosis arenas.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a multi-model diagnosis arena.")
    run.add_argument("uuid", help="Solutions Run UUID")
    run.add_argument("--config", default="config.toml", help="Path to config.toml")
    run.add_argument("--models", help="Comma-separated contender model list")
    run.add_argument("--output", help="Combined arena HTML output path")
    run.add_argument("--offline", action="store_true", help="Use already mirrored artifacts")
    run.add_argument("--refresh", action="store_true", help="Re-download cached objects")
    run.add_argument(
        "--llm-json",
        action="append",
        default=[],
        help="Use a precomputed diagnosis JSON payload for the next contender",
    )
    run.add_argument(
        "--budget",
        choices=["quick", "default", "hard"],
        default="default",
        help="Triage tool-round budget profile",
    )
    run.add_argument(
        "--max-tool-rounds",
        type=int,
        help="Override the selected triage budget round count",
    )

    export = subparsers.add_parser("export", help="Export judged arenas as JSONL.")
    export.add_argument("--config", default="config.toml", help="Path to config.toml")
    export.add_argument("--output", required=True, help="Provider-neutral JSONL path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run_arena(args)
    if args.command == "export":
        return _export_arenas(args)
    raise AssertionError(args.command)


def _run_arena(args: argparse.Namespace) -> int:
    _log("stage", "config")
    config = Config.load(args.config)
    models = _models_from_args(args.models, config.arena.models)
    for model in models:
        _log("model", model)
    if not args.offline:
        _log("stage", f"mirror start uuid={args.uuid}")
        manifest = SwiftMirror(config.swift, config.paths.artifact_root).mirror_uuid(
            args.uuid,
            refresh=args.refresh,
        )
        _log("result", f"mirrored_objects={len(manifest.objects)} root={manifest.root}")
    else:
        _log("stage", "mirror skipped (offline)")
    factory = None
    if args.llm_json:
        factory = _precomputed_factory(args.llm_json)
    _log("stage", f"arena start contenders={len(models)}")
    session = ArenaRunner(config, client_factory=factory).run(
        args.uuid,
        ArenaOptions(
            models=models,
            budget=args.budget,
            max_tool_rounds=args.max_tool_rounds,
            output=args.output,
        ),
    )
    _log("result", f"status={session['status']} summary={session['summary']}")
    print(session["output"])
    return 0


def _export_arenas(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    count = export_judged_arenas(config.paths.artifact_root, Path(args.output))
    _log("result", f"exported={count}")
    print(args.output)
    return 0


def _models_from_args(raw_models: str | None, config_models: list[str]) -> list[str]:
    if raw_models:
        models = [model.strip() for model in raw_models.split(",") if model.strip()]
    else:
        models = [model.strip() for model in config_models if model.strip()]
    if len(models) < 2:
        raise SystemExit("Arena runs require at least two models.")
    return models


class _PrecomputedClient:
    def __init__(self, model: str, report: DiagnosisReport):
        self.model = model
        self.report = report
        self.exchanges: list[dict[str, Any]] = []

    def diagnose(self, evidence_text: str, **kwargs):
        self.exchanges.append(
            {
                "request": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": evidence_text}],
                    "session_id": kwargs.get("session_id"),
                },
                "response": {
                    "content": json.dumps(asdict(self.report), sort_keys=True),
                    "usage": {},
                },
            }
        )
        return self.report


def _precomputed_factory(payloads: list[str]):
    reports = [DiagnosisReport.from_dict(json.loads(payload)) for payload in payloads]
    index = {"value": 0}

    def factory(llm_config):
        if index["value"] >= len(reports):
            raise RuntimeError("Not enough --llm-json payloads for arena contenders.")
        report = reports[index["value"]]
        index["value"] += 1
        return _PrecomputedClient(llm_config.model, report)

    return factory


def _log(kind: str, message: str) -> None:
    print(f"[{kind}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
