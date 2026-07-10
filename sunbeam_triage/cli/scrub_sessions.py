from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.sessions import scrub_session_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redact secrets already stored in triage session records."
    )
    parser.add_argument("--artifact-root", default="artifacts", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = scrub_session_store(args.artifact_root, dry_run=args.dry_run)
    print(json.dumps({
        "scanned": result.scanned,
        "changed": result.changed,
        "unchanged": result.unchanged,
        "malformed": list(result.malformed),
        "dry_run": args.dry_run,
    }, indent=2, sort_keys=True))
    return 1 if result.malformed else 0


if __name__ == "__main__":
    raise SystemExit(main())
