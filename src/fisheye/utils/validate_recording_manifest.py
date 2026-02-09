#!/usr/bin/env python3
"""Validate recording_manifest.json files before registry ingestion."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:  # optional rich dependency
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - optional
    Console = None
    Table = None

from fisheye.registry.db import Registry, RegistryPaths

DEFAULT_ALLOWED_TYPES = {"behavior", "microscopy", "histology"}
DEFAULT_ALLOWED_SUBTYPES: Dict[str, Set[str]] = {
    "behavior": {"free", "embedded"},
    "microscopy": {"lightsheet", "confocal", "2p"},
    "histology": {"section", "wholemount"},
}
DEFAULT_ALLOWED_BEHAVIOR_MODES = {"free", "embedded", "none"}
REQUIRED_FIELDS = (
    "recording_type",
    "recording_subtype",
    "behavior_mode",
    "artifact_schema_id",
)


@dataclass(frozen=True)
class ManifestIssue:
    path: Path
    code: str
    detail: str


def _apply_default_fields(payload: Dict[str, object]) -> bool:
    """Apply missing manifest defaults in-place. Returns True if mutated."""
    changed = False
    recording_type_raw = payload.get("recording_type")
    recording_type = str(recording_type_raw).strip() if recording_type_raw is not None else ""
    if not recording_type:
        payload["recording_type"] = "behavior"
        recording_type = "behavior"
        changed = True

    recording_subtype_raw = payload.get("recording_subtype")
    recording_subtype = (
        str(recording_subtype_raw).strip() if recording_subtype_raw is not None else ""
    )
    if not recording_subtype:
        if recording_type == "behavior":
            payload["recording_subtype"] = "free"
            recording_subtype = "free"
            changed = True

    behavior_mode_raw = payload.get("behavior_mode")
    behavior_mode = str(behavior_mode_raw).strip() if behavior_mode_raw is not None else ""
    if not behavior_mode:
        if recording_type == "behavior":
            payload["behavior_mode"] = recording_subtype or "free"
        else:
            payload["behavior_mode"] = "none"
        changed = True

    artifact_schema_raw = payload.get("artifact_schema_id")
    artifact_schema = str(artifact_schema_raw).strip() if artifact_schema_raw is not None else ""
    if not artifact_schema:
        payload["artifact_schema_id"] = "behavior_v1"
        changed = True

    return changed


def _apply_defaults_to_manifest(manifest_path: Path) -> bool:
    """Patch one manifest file in-place with default required fields."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if not _apply_default_fields(payload):
        return False
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def _discover_manifest_paths(
    paths: List[Path],
    *,
    recursive: bool,
    manifest_name: str,
) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if path.is_file():
            found.append(path)
            continue
        if not path.exists():
            continue
        if recursive:
            found.extend(path.rglob(manifest_name))
        else:
            candidate = path / manifest_name
            if candidate.exists():
                found.append(candidate)
    deduped: List[Path] = []
    seen: Set[Path] = set()
    for path in found:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return sorted(deduped)


def _load_vocab_from_registry(registry_path: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    registry = Registry(registry_path)
    try:
        type_rows = registry.conn.execute(
            """
            SELECT recording_type
            FROM recording_type_vocab
            WHERE active = 1
            ORDER BY recording_type;
            """
        ).fetchall()
        subtype_rows = registry.conn.execute(
            """
            SELECT recording_type, recording_subtype
            FROM recording_subtype_vocab
            WHERE active = 1
            ORDER BY recording_type, recording_subtype;
            """
        ).fetchall()
    finally:
        registry.close()

    allowed_types = {
        str(row["recording_type"]).strip()
        for row in type_rows
        if row["recording_type"] is not None and str(row["recording_type"]).strip()
    }
    allowed_subtypes: Dict[str, Set[str]] = {}
    for row in subtype_rows:
        recording_type = str(row["recording_type"]).strip() if row["recording_type"] is not None else ""
        recording_subtype = (
            str(row["recording_subtype"]).strip() if row["recording_subtype"] is not None else ""
        )
        if not recording_type or not recording_subtype:
            continue
        allowed_subtypes.setdefault(recording_type, set()).add(recording_subtype)

    if not allowed_types:
        allowed_types = set(DEFAULT_ALLOWED_TYPES)
    if not allowed_subtypes:
        allowed_subtypes = {key: set(values) for key, values in DEFAULT_ALLOWED_SUBTYPES.items()}
    return allowed_types, allowed_subtypes


def _validate_manifest(
    manifest_path: Path,
    *,
    allowed_types: Set[str],
    allowed_subtypes: Dict[str, Set[str]],
) -> List[ManifestIssue]:
    issues: List[ManifestIssue] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(ManifestIssue(manifest_path, "json_parse_error", str(exc)))
        return issues
    if not isinstance(payload, dict):
        issues.append(ManifestIssue(manifest_path, "invalid_payload", "manifest root must be a JSON object"))
        return issues

    values: Dict[str, Optional[str]] = {}
    for field in REQUIRED_FIELDS:
        raw = payload.get(field)
        value = str(raw).strip() if raw is not None else ""
        values[field] = value or None
        if not value:
            issues.append(ManifestIssue(manifest_path, "missing_required_field", field))

    recording_type = values["recording_type"]
    recording_subtype = values["recording_subtype"]
    behavior_mode = values["behavior_mode"]

    if recording_type and recording_type not in allowed_types:
        issues.append(
            ManifestIssue(
                manifest_path,
                "invalid_recording_type",
                f"{recording_type} (allowed={','.join(sorted(allowed_types))})",
            )
        )

    if recording_type and recording_subtype:
        allowed_for_type = allowed_subtypes.get(recording_type)
        if allowed_for_type and recording_subtype not in allowed_for_type:
            issues.append(
                ManifestIssue(
                    manifest_path,
                    "invalid_recording_subtype",
                    (
                        f"type={recording_type} subtype={recording_subtype} "
                        f"(allowed={','.join(sorted(allowed_for_type))})"
                    ),
                )
            )

    if behavior_mode and behavior_mode not in DEFAULT_ALLOWED_BEHAVIOR_MODES:
        issues.append(
            ManifestIssue(
                manifest_path,
                "invalid_behavior_mode",
                f"{behavior_mode} (allowed={','.join(sorted(DEFAULT_ALLOWED_BEHAVIOR_MODES))})",
            )
        )

    if recording_type == "behavior" and recording_subtype and behavior_mode:
        if recording_subtype != behavior_mode:
            issues.append(
                ManifestIssue(
                    manifest_path,
                    "behavior_mode_mismatch",
                    f"recording_subtype={recording_subtype} behavior_mode={behavior_mode}",
                )
            )

    return issues


def _render_results(
    *,
    manifests: List[Path],
    issues: List[ManifestIssue],
    use_rich: bool,
    as_json: bool,
    manifests_updated: int,
) -> None:
    valid_count = len(manifests) - len({issue.path for issue in issues})
    if as_json:
        payload = {
            "manifests_scanned": len(manifests),
            "manifests_updated": manifests_updated,
            "valid_manifests": valid_count,
            "invalid_manifests": len(manifests) - valid_count,
            "issues_count": len(issues),
            "issues": [
                {
                    "path": str(issue.path),
                    "code": issue.code,
                    "detail": issue.detail,
                }
                for issue in issues
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if use_rich and Console is not None and Table is not None:
        console = Console()
        summary = Table(title="Recording Manifest Validation", show_lines=False)
        summary.add_column("Field")
        summary.add_column("Value")
        summary.add_row("Manifests scanned", str(len(manifests)))
        summary.add_row("Manifests updated", str(manifests_updated))
        summary.add_row("Valid manifests", str(valid_count))
        summary.add_row("Invalid manifests", str(len(manifests) - valid_count))
        summary.add_row("Issues", str(len(issues)))
        console.print(summary)
        if issues:
            table = Table(title="Manifest Issues", show_lines=False)
            table.add_column("Path", style="cyan")
            table.add_column("Code")
            table.add_column("Detail")
            for issue in issues:
                table.add_row(str(issue.path), issue.code, issue.detail)
            console.print(table)
        return

    print("Recording Manifest Validation")
    print(f"  manifests_scanned: {len(manifests)}")
    print(f"  manifests_updated: {manifests_updated}")
    print(f"  valid_manifests: {valid_count}")
    print(f"  invalid_manifests: {len(manifests) - valid_count}")
    print(f"  issues: {len(issues)}")
    for issue in issues:
        print(f"- {issue.path} [{issue.code}] {issue.detail}")


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Manifest file(s) and/or directories to scan.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover manifests under directory inputs.",
    )
    parser.add_argument(
        "--manifest-name",
        default="recording_manifest.json",
        help="Manifest filename to discover in directory inputs.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry path to load active recording vocab tables.",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich table output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    parser.add_argument(
        "--apply-defaults",
        action="store_true",
        help="Patch missing required manifest fields in-place before validation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    manifest_paths = _discover_manifest_paths(
        [Path(path).expanduser() for path in args.paths],
        recursive=bool(args.recursive),
        manifest_name=str(args.manifest_name),
    )
    if not manifest_paths:
        print("No manifest files found.")
        return 1

    allowed_types = set(DEFAULT_ALLOWED_TYPES)
    allowed_subtypes = {key: set(values) for key, values in DEFAULT_ALLOWED_SUBTYPES.items()}
    if args.registry:
        registry_path = Path(args.registry).expanduser()
    else:
        registry_path = RegistryPaths.from_env(Path.cwd()).path
    if registry_path.exists():
        try:
            allowed_types, allowed_subtypes = _load_vocab_from_registry(registry_path)
        except Exception:
            # Keep default vocab when registry lookup fails.
            pass

    manifests_updated = 0
    if args.apply_defaults:
        for manifest_path in manifest_paths:
            if _apply_defaults_to_manifest(manifest_path):
                manifests_updated += 1

    issues: List[ManifestIssue] = []
    for manifest_path in manifest_paths:
        issues.extend(
            _validate_manifest(
                manifest_path,
                allowed_types=allowed_types,
                allowed_subtypes=allowed_subtypes,
            )
        )

    use_rich = not args.no_rich and not args.json and Console is not None and Table is not None
    _render_results(
        manifests=manifest_paths,
        issues=issues,
        use_rich=use_rich,
        as_json=bool(args.json),
        manifests_updated=manifests_updated,
    )
    return 2 if issues else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
