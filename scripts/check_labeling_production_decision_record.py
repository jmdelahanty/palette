"""Check that the web-labeling production decision record is filled in."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_RECORD = Path("docs/web_labeling_production_decision_record.md")

REQUIRED_TEXT_FIELDS = (
    "decision_record_owner",
    "decision_date",
    "service_url",
    "auth_provider",
    "proxy_or_gateway",
    "trusted_user_header",
    "admin_users",
    "host",
    "service_account",
    "working_directory",
    "labeling_store_path",
    "palette_repo_path",
    "registry_path",
    "zarr_mounts",
    "backup_location",
    "service_bind_port",
    "external_url",
    "tls_termination",
    "allowed_clients_or_networks",
    "operator",
    "reviewer",
)

REQUIRED_YES_FIELDS = (
    "production_ready",
    "header_strip_rule_confirmed",
    "header_rewrite_rule_confirmed",
    "static_validation_passed",
    "focused_unit_tests_passed",
    "real_zarr_smoke_passed",
    "sidecar_backup_tested",
    "mutable_zarr_backup_tested",
    "admin_preflight_clean",
    "approved_for_labelers",
)

YES_NO_FIELDS = REQUIRED_YES_FIELDS + ("non_loopback_bind_required",)


def _parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*):\s*(.*)$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def _placeholder(value: str) -> bool:
    text = str(value or "").strip()
    return not text or text.lower() in {"todo", "tbd", "replace-me", "replace with value", "<todo>"}


def check_record(path: Path) -> dict[str, object]:
    fields = _parse_fields(path.read_text(encoding="utf-8"))
    missing_text = [field for field in REQUIRED_TEXT_FIELDS if _placeholder(fields.get(field, ""))]
    not_yes = [field for field in REQUIRED_YES_FIELDS if fields.get(field, "").strip().lower() != "yes"]
    invalid_yes_no = [
        field
        for field in YES_NO_FIELDS
        if fields.get(field, "").strip().lower() not in {"yes", "no"}
    ]
    return {
        "ok": not missing_text and not not_yes and not invalid_yes_no,
        "path": str(path),
        "missing_text_fields": missing_text,
        "required_yes_fields_not_yes": not_yes,
        "invalid_yes_no_fields": invalid_yes_no,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        default=str(DEFAULT_RECORD),
        help="Path to web_labeling_production_decision_record.md.",
    )
    args = parser.parse_args(argv)
    path = Path(args.record)
    if not path.is_file():
        print(json.dumps({"ok": False, "error": "record_not_found", "path": str(path)}, sort_keys=True))
        return 2
    result = check_record(path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
