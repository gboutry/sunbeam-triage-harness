from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.tool_activity import analyze_tool_activity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze saved Streamlit/OpenRouter tool-call rounds."
    )
    parser.add_argument("sessions", nargs="+", help="Session JSON files to analyze")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for item in args.sessions:
        path = Path(item)
        session = json.loads(path.read_text(encoding="utf-8"))
        analysis = analyze_tool_activity(session)
        warnings = ",".join(analysis["warnings"]) if analysis["warnings"] else "-"
        print(
            f"{analysis['uuid']} "
            f"model={analysis['model']} "
            f"exchanges={analysis['exchange_count']} "
            f"tool_calls={analysis['tool_call_count']} "
            f"tool_results={analysis['tool_result_count']} "
            f"tool_result_chars={analysis['tool_result_chars']} "
            f"session_tokens={analysis['total_tokens']} "
            f"session_cost_usd={analysis['total_cost']:.6g} "
            f"warnings={warnings}"
        )
        for row in analysis["rows"]:
            target = row["target"] or "-"
            print(
                f"  exchange={row['exchange']} "
                f"tool={row['tool_name']} "
                f"target={target} "
                f"result_chars={row['result_chars']} "
                f"exchange_tokens={row['total_tokens']}"
                f" exchange_cost_usd={row['cost']:.6g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
