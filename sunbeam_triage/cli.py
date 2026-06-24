from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .evidence import EvidenceCollector
from .llm import DiagnosisReport, OpenRouterClient
from .render import render_html
from .swift import SwiftMirror


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a Sunbeam CI failure UUID.")
    parser.add_argument("uuid", help="Solutions Run UUID")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--output", help="Output HTML path or pattern")
    parser.add_argument("--artifact-root", help="Local artifact cache root")
    parser.add_argument("--model", help="OpenRouter model override")
    parser.add_argument("--refresh", action="store_true", help="Re-download cached objects")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use already mirrored artifacts; do not fetch Swift objects",
    )
    parser.add_argument(
        "--llm-json",
        help="Use a precomputed diagnosis JSON payload instead of calling OpenRouter",
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
    if not args.offline:
        _log("stage", f"mirror start uuid={uuid}")
        mirror = SwiftMirror(config.swift, config.paths.artifact_root)
        manifest = mirror.mirror_uuid(uuid, refresh=args.refresh)
        _log("result", f"mirrored_objects={len(manifest.objects)} root={manifest.root}")
    else:
        _log("stage", "mirror skipped (offline)")

    artifact_root = config.paths.artifact_root / uuid
    _log("stage", f"evidence root={artifact_root}")
    pack = EvidenceCollector(artifact_root, uuid).collect()
    _log(
        "result",
        (
            f"failed_step={pack.failed_step.name} "
            f"family={pack.failed_step.family} evidence_items={len(pack.evidence)}"
        ),
    )
    if args.llm_json:
        _log("stage", "diagnosis using supplied JSON")
        report = DiagnosisReport.from_dict(json.loads(args.llm_json))
    else:
        _log("stage", f"diagnosis requesting model={config.llm.model}")
        report = OpenRouterClient(config.llm).diagnose(
            pack.to_prompt_text(),
            session_id=uuid,
        )
    _log("result", f"confidence={report.confidence} summary={report.summary}")

    output = config.output_path(uuid)
    _log("stage", f"render output={output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(pack, report), encoding="utf-8")
    print(output)
    return 0


def _log(kind: str, message: str) -> None:
    print(f"[{kind}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
