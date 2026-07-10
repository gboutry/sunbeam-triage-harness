from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.evaluation import load_evaluation_cases, manifest_sha256, score_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score saved triage sessions.")
    parser.add_argument("corpus", type=Path, help="Evaluation corpus JSON")
    parser.add_argument("sessions", type=Path, help="Directory containing session JSON")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Optional artifact cache root used to reject corpus drift",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scores = []
    for case in load_evaluation_cases(args.corpus):
        if args.artifact_root is not None:
            actual_hash = manifest_sha256(args.artifact_root / case.uuid)
            if actual_hash != case.manifest_sha256:
                scores.append({
                    "uuid": case.uuid,
                    "passed": False,
                    "manifest_mismatch": True,
                    "expected_manifest_sha256": case.manifest_sha256,
                    "actual_manifest_sha256": actual_hash,
                })
                continue
        path = args.sessions / f"{case.uuid}.json"
        if not path.exists():
            scores.append({"uuid": case.uuid, "passed": False, "missing_session": True})
            continue
        session = json.loads(path.read_text(encoding="utf-8"))
        scores.append(score_session(case, session))
    summary = {
        "case_count": len(scores),
        "passed": sum(bool(score["passed"]) for score in scores),
        "scores": scores,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"passed={summary['passed']}/{summary['case_count']}")
        for score in scores:
            print(f"{score['uuid']} passed={score['passed']}")
    return 0 if summary["passed"] == summary["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
