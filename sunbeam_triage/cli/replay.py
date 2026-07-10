from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core.config import Config
from ..core.evaluation import load_evaluation_cases
from ..core.replay import replay_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a historical triage corpus.")
    parser.add_argument("corpus", type=Path, help="Evaluation corpus JSON")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--model", help="OpenRouter model override")
    parser.add_argument("--artifact-root", default="artifacts", type=Path)
    parser.add_argument(
        "--output-root",
        default="artifacts/.sunbeam-triage/replays",
        type=Path,
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--uuid",
        action="append",
        dest="uuids",
        help="Replay only this UUID; repeat to select multiple cases",
    )
    parser.add_argument(
        "--budget", choices=["quick", "default", "hard"], default="default"
    )
    parser.add_argument("--case-timeout-seconds", type=int, default=480)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    config = Config.load(
        args.config,
        cli_model=args.model,
        cli_artifact_root=args.artifact_root,
    )
    if not config.llm.api_key:
        raise SystemExit(f"Missing {config.llm.api_key_env}")

    def show_progress(event):
        round_text = (
            "" if event.round_number is None else f" round={event.round_number}"
        )
        tool_text = "" if event.tool_name is None else f" tool={event.tool_name}"
        target_text = "" if not event.target else f" target={event.target}"
        print(
            f"[{event.run_id}] {event.phase}{round_text}{tool_text}{target_text}: "
            f"{event.message}",
            file=sys.stderr,
            flush=True,
        )

    cases = load_evaluation_cases(args.corpus)
    if args.uuids:
        selected = set(args.uuids)
        cases = [case for case in cases if case.uuid in selected]
        missing = sorted(selected - {case.uuid for case in cases})
        if missing:
            raise SystemExit(f"UUID not present in corpus: {', '.join(missing)}")
    summary = replay_corpus(
        config,
        cases,
        repetitions=args.repetitions,
        output_root=args.output_root,
        budget=args.budget,
        progress=show_progress,
        timeout_seconds=args.case_timeout_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["completed"] == summary["attempt_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
