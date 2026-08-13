#!/usr/bin/env python3
"""Refresh derivable context fields in existing recording_manifest.json files.

Defaults to dry-run; pass --apply to write changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from fisheye.shared.type_conversions import normalize_attr as _normalize_attr
from fisheye.utils.organize_recordings import (
    _external_ipc_output_for_camera,
    _external_ipc_video_streams_payload,
    _read_camera_context,
    _runtime_snapshot_software_version,
)


REFRESH_FIELDS = (
    "protocol_name",
    "protocol_name_from_definition",
    "dish_design",
    "genotype",
    "dpf_at_acquisition",
    "software_version",
)
PLACEHOLDER_METADATA_VALUES = {"unknown", "none", "null", "n/a", "na"}


@dataclass(frozen=True)
class ArtifactCopyPlan:
    source: Path
    dest: Path
    manifest_section: str
    manifest_path: str


@dataclass(frozen=True)
class ManifestRefreshPlan:
    manifest_path: Path
    h5_path: Optional[Path]
    updates: dict[str, dict[str, Any]]
    preserved_existing: dict[str, dict[str, Any]]
    unavailable: list[str]
    file_additions: dict[str, list[str]] = field(default_factory=dict)
    artifact_copies: list[ArtifactCopyPlan] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_updates(self) -> bool:
        return bool(self.updates or self.file_additions or self.artifact_copies)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return not text or text.lower() in PLACEHOLDER_METADATA_VALUES
    return False


def _iter_manifests(
    roots: Sequence[Path],
    *,
    recursive: bool,
    manifest_name: str,
) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        path = root.expanduser()
        if path.is_file():
            if path.name != manifest_name:
                continue
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                yield path
            continue
        if not path.exists():
            continue
        direct = path / manifest_name
        if direct.is_file():
            key = str(direct.resolve())
            if key not in seen:
                seen.add(key)
                yield direct
        candidates = path.rglob(manifest_name) if recursive else path.glob(f"*/{manifest_name}")
        for manifest_path in candidates:
            if not manifest_path.is_file():
                continue
            key = str(manifest_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield manifest_path


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    return payload


def _manifest_file_entries(payload: Mapping[str, Any]) -> list[str]:
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return []
    entries: list[str] = []
    for section in ("raw", "cams", "derived"):
        values = files.get(section)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str):
                continue
            text = value.strip()
            if text:
                entries.append(text)
    return entries


def _resolve_h5_path(recording_dir: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    candidates: list[Path] = []
    for rel_path in _manifest_file_entries(payload):
        if rel_path.lower().endswith(".h5"):
            candidates.append(recording_dir / rel_path)
    candidates.extend(sorted((recording_dir / "raw").glob("*.h5")))

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    existing = [path for path in unique if path.is_file()]
    if len(existing) == 1:
        return existing[0]

    recording_name = _normalize_attr(payload.get("recording_name"))
    if recording_name:
        for path in existing:
            if path.stem == recording_name:
                return path
    return None


def _runtime_snapshot_candidates(recording_dir: Path) -> list[Path]:
    return [
        recording_dir / "raw" / "recording_snapshot_runtime.json",
        recording_dir / "raw" / "recording_snapshot.json",
        recording_dir / "derived" / "recording_snapshot.json",
        recording_dir / "recording_snapshot.json",
    ]


def _session_candidates(recording_dir: Path) -> list[Path]:
    return [
        recording_dir / "raw" / "recording_session.json",
        recording_dir / "recording_session.json",
    ]


def _load_optional_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _derived_manifest_context(recording_dir: Path, payload: Mapping[str, Any]) -> tuple[Optional[Path], dict[str, Any], list[str]]:
    unavailable: list[str] = []
    h5_path = _resolve_h5_path(recording_dir, payload)
    derived: dict[str, Any] = {}
    if h5_path is None:
        unavailable.append("h5")
    else:
        _, h5_meta = _read_camera_context(h5_path)
        derived.update({key: h5_meta.get(key) for key in REFRESH_FIELDS if h5_meta.get(key)})

    if not derived.get("software_version"):
        for snapshot_path in _runtime_snapshot_candidates(recording_dir):
            value = _runtime_snapshot_software_version(snapshot_path)
            if value:
                derived["software_version"] = value
                break

    for field in REFRESH_FIELDS:
        if not derived.get(field):
            unavailable.append(field)
    return h5_path, derived, unavailable


def _manifest_file_set(payload: Mapping[str, Any], section: str) -> set[str]:
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return set()
    values = files.get(section)
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str)}


def _existing_or_planned_file_addition(
    *,
    recording_dir: Path,
    source: Path,
    rel_dest: str,
    section: str,
    payload: Mapping[str, Any],
    copies: list[ArtifactCopyPlan],
    additions: dict[str, list[str]],
    unavailable: list[str],
) -> None:
    dest = recording_dir / rel_dest
    if not dest.exists():
        if source.exists():
            copies.append(
                ArtifactCopyPlan(
                    source=source,
                    dest=dest,
                    manifest_section=section,
                    manifest_path=rel_dest,
                )
            )
        else:
            unavailable.append(f"artifact:{source.name}")
            return
    if rel_dest not in _manifest_file_set(payload, section):
        additions.setdefault(section, []).append(rel_dest)


def _source_dir_from_manifest(payload: Mapping[str, Any]) -> Optional[Path]:
    text = _normalize_attr(payload.get("source_dir"))
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.exists() else None


def _first_manifest_file(
    payload: Mapping[str, Any],
    *,
    section: str,
    suffix: str,
    contains: Optional[str] = None,
) -> Optional[str]:
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return None
    values = files.get(section)
    if not isinstance(values, list):
        return None
    for value in values:
        if not isinstance(value, str):
            continue
        if not value.endswith(suffix):
            continue
        if contains and contains not in value:
            continue
        return value
    return None


def _derive_video_streams(
    recording_dir: Path,
    payload: Mapping[str, Any],
) -> Optional[dict[str, object]]:
    camera_id = _normalize_attr(payload.get("camera_id"))
    full_video = _first_manifest_file(payload, section="cams", suffix=".mp4")
    if not camera_id or not full_video:
        return None

    session: dict[str, Any] = {}
    for candidate in _session_candidates(recording_dir):
        session = _load_optional_json_object(candidate)
        if session:
            break

    outputs = _external_ipc_output_for_camera(session, camera_id)
    full_output = outputs.get("full") if isinstance(outputs.get("full"), dict) else {}
    crop_output = outputs.get("crop") if isinstance(outputs.get("crop"), dict) else {}
    if not crop_output and _first_manifest_file(
        payload, section="derived", suffix="_crop_external.mp4"
    ):
        crop_output = {"output_kind": "crop"}

    cam_base = Path(full_video).stem
    full_metadata = recording_dir / "cams" / f"{cam_base}_external_meta.csv"
    compat_metadata = recording_dir / "cams" / f"{cam_base}_meta.csv"
    full_frame_clock_metadata = None
    if full_metadata.is_file():
        full_frame_clock_metadata = f"cams/{full_metadata.name}"
    elif compat_metadata.is_file():
        full_frame_clock_metadata = f"cams/{compat_metadata.name}"

    return _external_ipc_video_streams_payload(
        camera_id=camera_id,
        cam_base=cam_base,
        full_output=full_output,
        crop_output=crop_output,
        full_frame_clock_metadata=full_frame_clock_metadata,
        has_full_summary=(
            recording_dir / "cams" / f"{cam_base}_external_summary.json"
        ).is_file(),
        has_full_status=(
            recording_dir
            / "derived"
            / "external_recorder"
            / f"{cam_base}_external_status.json"
        ).is_file(),
    )


def _external_ipc_artifact_context(
    recording_dir: Path,
    payload: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], list[ArtifactCopyPlan], list[str]]:
    updates: dict[str, dict[str, Any]] = {}
    additions: dict[str, list[str]] = {}
    copies: list[ArtifactCopyPlan] = []
    unavailable: list[str] = []

    source_dir = _source_dir_from_manifest(payload)
    if source_dir is None:
        unavailable.append("source_dir")
    else:
        for source_name, rel_dest, section in (
            ("_citrus_transfer_complete.json", "raw/transfer_complete.json", "raw"),
            (
                "orange_local_control.events.jsonl",
                "raw/orange_local_control.events.jsonl",
                "raw",
            ),
        ):
            _existing_or_planned_file_addition(
                recording_dir=recording_dir,
                source=source_dir / source_name,
                rel_dest=rel_dest,
                section=section,
                payload=payload,
                copies=copies,
                additions=additions,
                unavailable=unavailable,
            )

        threading_candidates = list(source_dir.glob("*threading_startup*.json"))
        citrus_dir = source_dir / "citrus"
        if citrus_dir.exists():
            threading_candidates.extend(citrus_dir.glob("*threading_startup*.json"))
        if not threading_candidates:
            unavailable.append("artifact:*threading_startup*.json")
        for source in sorted(set(threading_candidates)):
            _existing_or_planned_file_addition(
                recording_dir=recording_dir,
                source=source,
                rel_dest=f"derived/citrus/{source.name}",
                section="derived",
                payload=payload,
                copies=copies,
                additions=additions,
                unavailable=unavailable,
            )

    if not isinstance(payload.get("video_streams"), Mapping):
        video_streams = _derive_video_streams(recording_dir, payload)
        if video_streams:
            updates["video_streams"] = {"old": payload.get("video_streams"), "new": video_streams}
        else:
            unavailable.append("video_streams")

    return updates, additions, copies, unavailable


def build_refresh_plan(
    manifest_path: Path,
    *,
    overwrite_existing: bool = False,
    refresh_external_ipc_artifacts: bool = False,
) -> ManifestRefreshPlan:
    try:
        payload = _load_manifest(manifest_path)
    except Exception as exc:
        return ManifestRefreshPlan(
            manifest_path=manifest_path,
            h5_path=None,
            updates={},
            file_additions={},
            artifact_copies=[],
            preserved_existing={},
            unavailable=[],
            error=f"failed to read manifest: {exc}",
        )

    recording_dir = manifest_path.parent
    h5_path, derived, unavailable = _derived_manifest_context(recording_dir, payload)
    updates: dict[str, dict[str, Any]] = {}
    preserved_existing: dict[str, dict[str, Any]] = {}
    for field in REFRESH_FIELDS:
        new_value = derived.get(field)
        old_value = payload.get(field)
        if _is_empty(new_value):
            if field in payload and isinstance(old_value, str) and old_value.strip().lower() in PLACEHOLDER_METADATA_VALUES:
                updates[field] = {"old": old_value, "new": None}
            continue
        if _is_empty(old_value):
            updates[field] = {"old": old_value, "new": new_value}
        elif overwrite_existing and old_value != new_value:
            updates[field] = {"old": old_value, "new": new_value}
        elif old_value != new_value:
            preserved_existing[field] = {"old": old_value, "new": new_value}

    file_additions: dict[str, list[str]] = {}
    artifact_copies: list[ArtifactCopyPlan] = []
    if (
        refresh_external_ipc_artifacts
        and payload.get("recording_backend") == "external_ipc"
    ):
        (
            artifact_updates,
            file_additions,
            artifact_copies,
            artifact_unavailable,
        ) = _external_ipc_artifact_context(recording_dir, payload)
        updates.update(artifact_updates)
        unavailable.extend(artifact_unavailable)

    return ManifestRefreshPlan(
        manifest_path=manifest_path,
        h5_path=h5_path,
        updates=updates,
        file_additions=file_additions,
        artifact_copies=artifact_copies,
        preserved_existing=preserved_existing,
        unavailable=unavailable,
    )


def apply_refresh_plan(plan: ManifestRefreshPlan) -> bool:
    if plan.error or not plan.has_updates:
        return False
    for copy_plan in plan.artifact_copies:
        if copy_plan.dest.exists():
            continue
        copy_plan.dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(copy_plan.source), str(copy_plan.dest))
    payload = _load_manifest(plan.manifest_path)
    for field, change in plan.updates.items():
        payload[field] = change["new"]
    files = payload.setdefault("files", {})
    if isinstance(files, dict):
        for section, additions in plan.file_additions.items():
            values = files.setdefault(section, [])
            if not isinstance(values, list):
                continue
            for rel_path in additions:
                if rel_path not in values:
                    values.append(rel_path)
    plan.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _print_plan(plan: ManifestRefreshPlan, *, apply: bool, show_unchanged: bool) -> None:
    if plan.error:
        print(f"ERROR {plan.manifest_path}: {plan.error}")
        return
    if plan.has_updates:
        action = "UPDATED" if apply else "WOULD UPDATE"
        fields = ",".join(plan.updates)
        file_count = sum(len(values) for values in plan.file_additions.values())
        print(
            f"{action} {plan.manifest_path}: "
            f"fields={fields or '-'} file_additions={file_count} copies={len(plan.artifact_copies)}"
        )
        return
    if show_unchanged:
        print(f"UNCHANGED {plan.manifest_path}")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Recording folders, roots, or recording_manifest.json paths.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan path roots for recording manifests.",
    )
    parser.add_argument(
        "--manifest-name",
        default="recording_manifest.json",
        help="Manifest filename to refresh (default: recording_manifest.json).",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite non-empty manifest values when H5/runtime metadata differs.",
    )
    parser.add_argument(
        "--refresh-external-ipc-artifacts",
        action="store_true",
        help=(
            "For external_ipc manifests, copy missing small session/control "
            "artifacts, add manifest files entries, and add video_streams."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write manifest updates (default is dry-run).",
    )
    parser.add_argument(
        "--show-unchanged",
        action="store_true",
        help="Print manifests that have no updates.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifests = list(
        _iter_manifests(
            list(args.paths),
            recursive=bool(args.recursive),
            manifest_name=str(args.manifest_name),
        )
    )
    if not manifests:
        print("No manifests found.")
        return 1

    plans = [
        build_refresh_plan(
            path,
            overwrite_existing=bool(args.overwrite_existing),
            refresh_external_ipc_artifacts=bool(args.refresh_external_ipc_artifacts),
        )
        for path in manifests
    ]
    applied = 0
    for plan in plans:
        if args.apply and apply_refresh_plan(plan):
            applied += 1
        if not args.json:
            _print_plan(plan, apply=bool(args.apply), show_unchanged=bool(args.show_unchanged))

    summary = {
        "manifests_found": len(plans),
        "would_update": sum(1 for plan in plans if plan.has_updates),
        "updated": applied,
        "failed": sum(1 for plan in plans if plan.error),
        "fields": {
            field: sum(1 for plan in plans if field in plan.updates)
            for field in (*REFRESH_FIELDS, "video_streams")
        },
        "file_additions": sum(
            sum(len(values) for values in plan.file_additions.values())
            for plan in plans
        ),
        "artifact_copies": sum(len(plan.artifact_copies) for plan in plans),
        "preserved_existing": sum(1 for plan in plans if plan.preserved_existing),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("\nSummary:")
        for key, value in summary.items():
            if key == "fields":
                print("  fields:")
                for field, count in value.items():
                    print(f"    {field}: {count}")
            else:
                print(f"  {key}: {value}")
        if not args.apply:
            print("Dry-run only. Re-run with --apply to write changes.")
    return 2 if summary["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
