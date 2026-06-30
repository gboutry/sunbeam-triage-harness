from __future__ import annotations

import argparse
import json
from pathlib import Path

from sunbeam_triage.core.step_profile_extractor import extract_profiles, profiles_to_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("steps_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = profiles_to_dict(extract_profiles(args.steps_dir))
    rendered = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
