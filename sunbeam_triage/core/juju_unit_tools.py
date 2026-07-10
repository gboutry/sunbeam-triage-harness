from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Any

from .redaction import redact_text

UNIT_TOKEN = re.compile(r"^(?P<unit>[a-z0-9-]+/\d+)\*?$")
ERROR_STATE = re.compile(r"\b(error|blocked|lost|unknown)\b", re.IGNORECASE)


def resolve_juju_unit(root: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    unit = str(arguments.get("unit", "")).rstrip("*")
    if not UNIT_TOKEN.match(unit):
        return {"ok": False, "error": "A Juju unit name such as app/0 is required."}
    records = _unit_records(root)
    record = records.get(unit)
    if record is None:
        return {"ok": False, "unit": unit, "error": "Juju unit was not found."}

    machine_id = str(record.get("machine_id", ""))
    machine = _machine_records(root).get(machine_id, {})
    hostname = str(machine.get("hostname", ""))
    archives = _matching_archives(root, hostname)
    member = f"var/log/juju/unit-{unit.replace('/', '-')}.log"
    suggested = [
        {"archive_path": archive, "member_path": member}
        for archive in archives
        if _archive_has_member(Path(root) / archive, member)
    ]
    return {
        "ok": True,
        "unit": unit,
        "principal": record.get("principal", ""),
        "machine_id": machine_id,
        "hostname": hostname,
        "addresses": machine.get("ip-addresses", []),
        "status_path": record.get("status_path", ""),
        "status_line": record.get("status_line"),
        "status_excerpt": redact_text(str(record.get("status_excerpt", ""))),
        "archive_paths": archives,
        "suggested_members": suggested,
        "missing_evidence": (
            []
            if suggested
            else ["No matching Juju unit log was found in the host sosreport."]
        ),
    }


def find_juju_error_units(root: Path) -> list[dict[str, Any]]:
    findings = []
    for unit, record in _unit_records(root).items():
        if not ERROR_STATE.search(str(record.get("status_excerpt", ""))):
            continue
        resolved = resolve_juju_unit(root, {"unit": unit})
        if resolved.get("ok"):
            findings.append(resolved)
    return findings


def _unit_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    generated = Path(root) / "generated/sunbeam"
    for status_path in sorted(generated.glob("juju_status_*.txt")):
        principal = ""
        principal_machine = ""
        with status_path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                match = UNIT_TOKEN.match(parts[0]) if parts else None
                if match is None:
                    continue
                unit = match.group("unit")
                subordinate = line[0].isspace()
                if not subordinate:
                    principal = unit
                    principal_machine = parts[3] if len(parts) > 3 else ""
                machine_id = principal_machine if subordinate else (
                    parts[3] if len(parts) > 3 else ""
                )
                records[unit] = {
                    "principal": principal if subordinate else unit,
                    "machine_id": machine_id if machine_id.isdigit() else "",
                    "status_path": status_path.relative_to(root).as_posix(),
                    "status_line": line_number,
                    "status_excerpt": stripped,
                }
    return records


def _machine_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    generated = Path(root) / "generated/sunbeam"
    for path in sorted(generated.glob("juju_machines_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        machines = data.get("machines", {}) if isinstance(data, dict) else {}
        if not isinstance(machines, dict):
            continue
        for machine_id, machine in machines.items():
            if isinstance(machine, dict):
                records[str(machine_id)] = machine
    return records


def _matching_archives(root: Path, hostname: str) -> list[str]:
    if not hostname:
        return []
    return [
        path.relative_to(root).as_posix()
        for path in sorted(
            (Path(root) / "generated/sunbeam").glob(f"sosreport-{hostname}-*.tar*")
        )
        if not path.name.endswith(".sha256")
    ]


def _archive_has_member(path: Path, expected: str) -> bool:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            return any(
                member.isfile()
                and (member.name == expected or member.name.endswith(f"/{expected}"))
                for member in archive.getmembers()
            )
    except (OSError, tarfile.TarError):
        return False
