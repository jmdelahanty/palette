"""Register the recovered training archives without making them selectable.

``fisheye.training.recover_merged_training_recording`` rebuilt one training
derivative per recording whose ``/nvme1`` archives were deleted
(``<recording>/zarr/<recording>_recovered_training.zarr`` under the recordings
root). They are deliberately selector-ineligible and do not claim the deleted
originals' identity (docs/diagnostics/merged_pose_detect_recovery_full_2026_09_14.md),
so the generic scan (which trusts the archive's ``zarr_purpose: training``) is
not used. Each archive is registered with:

- ``zarr_use='recovered_training'``, ``artifact_kind='recovered_training_derivative'``,
  ``zarr_origin='derived'`` (never ``zarr_use='training'``: training selection
  filters on that), and the original's session and recording identity;
- a ``recovered_from`` lineage edge to its original training dataset, which
  stays ``status='missing'``.

Dry run by default; ``--apply`` writes through ``publish_registry_shadow``
(backup, local candidate, publish only if the registry did not change).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Sequence

RECOVERY_SCHEMAS = (
    "palette.training.merged_pose_detect_recovery.v1",
    "palette.training.merged_pose_detect_recovery_source.v1",
)
ZARR_USE = "recovered_training"
ARTIFACT_KIND = "recovered_training_derivative"
RELATIONSHIP = "recovered_from"
DEFAULT_RECORDINGS_ROOT = Path("/groups/johnson/johnsonlab/jeremy/recordings")


@dataclass(frozen=True)
class Planned:
    dataset_id: str
    zarr_path: str
    recording_id: str
    session_uuid: str
    parent_dataset_id: str
    recovery_schema_id: str
    recovery_mode: str | None
    recovered_created_utc: str | None


def _read_attrs(path: Path) -> dict[str, Any]:
    import zarr

    return dict(zarr.open_group(str(path), mode="r").attrs)


def plan(registry: Path, recordings_root: Path) -> tuple[list[Planned], list[dict]]:
    from fisheye.registry.recording_identity_authority import canonical_dataset_path_hash

    con = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    planned: list[Planned] = []
    skipped: list[dict] = []
    try:
        for archive in sorted(recordings_root.glob("*/zarr/*_recovered_training.zarr")):
            path = archive.resolve()
            try:
                attrs = _read_attrs(path)
            except Exception as exc:  # unreadable archive: report, never guess
                skipped.append({"zarr_path": str(path), "reason": f"unreadable: {exc}"})
                continue
            if attrs.get("schema_id") not in RECOVERY_SCHEMAS:
                skipped.append({"zarr_path": str(path), "reason": f"not a recovery archive: {attrs.get('schema_id')!r}"})
                continue
            if attrs.get("stage_selector_eligible") is not False:
                skipped.append({"zarr_path": str(path), "reason": "archive is not marked selector-ineligible"})
                continue
            if con.execute("SELECT 1 FROM datasets WHERE path_hash = ? OR zarr_path = ?",
                           (canonical_dataset_path_hash(path), str(path))).fetchone():
                skipped.append({"zarr_path": str(path), "reason": "already registered"})
                continue
            recording_id = str(attrs.get("recording_id") or "")
            originals = con.execute(
                "SELECT dataset_id, session_uuid FROM datasets "
                "WHERE recording_id = ? AND zarr_use = 'training' AND status = 'missing'",
                (recording_id,),
            ).fetchall()
            if len(originals) != 1:
                skipped.append({"zarr_path": str(path), "reason": f"{len(originals)} missing original training datasets for recording {recording_id!r}"})
                continue
            original = originals[0]
            planned.append(Planned(
                dataset_id=f"{original['session_uuid']}:z{canonical_dataset_path_hash(path)[:12]}",
                zarr_path=str(path),
                recording_id=recording_id,
                session_uuid=str(original["session_uuid"]),
                parent_dataset_id=str(original["dataset_id"]),
                recovery_schema_id=str(attrs["schema_id"]),
                recovery_mode=attrs.get("recovery_mode"),
                recovered_created_utc=attrs.get("created_utc"),
            ))
    finally:
        con.close()
    return planned, skipped


def apply_plan(candidate: Path, planned: Sequence[Planned]) -> dict:
    from fisheye.registry.db import Registry

    registry = Registry(candidate)
    try:
        for item in planned:
            registry.upsert_dataset(
                item.dataset_id,
                session_uuid=item.session_uuid,
                zarr_path=Path(item.zarr_path),
                recording_id=item.recording_id,
                artifact_kind=ARTIFACT_KIND,
                zarr_origin="derived",
                zarr_use=ZARR_USE,
            )
            registry.replace_dataset_lineage(
                child_dataset_id=item.dataset_id,
                parent_dataset_ids=[item.parent_dataset_id],
                relationship_type=RELATIONSHIP,
                metadata={"recovery_schema_id": item.recovery_schema_id,
                          "recovery_mode": item.recovery_mode,
                          "recovered_created_utc": item.recovered_created_utc},
            )
        registry.conn.commit()
    finally:
        registry.close()
    return {"operation": "register_recovered_training_archives", "registered": len(planned)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path, help="Backup path (required with --apply).")
    args = parser.parse_args(argv)
    planned, skipped = plan(args.registry, args.recordings_root)
    print(json.dumps({"planned": [asdict(p) for p in planned], "skipped": skipped}, indent=1))
    print(f"planned={len(planned)} skipped={len(skipped)}", file=sys.stderr)
    if not args.apply or not planned:
        return 0
    if args.backup is None:
        parser.error("--apply requires --backup")
    from fisheye.registry.shadow_publish import publish_registry_shadow

    publication = publish_registry_shadow(
        canonical_registry=args.registry, backup_path=args.backup,
        mutate=lambda candidate: apply_plan(candidate, planned),
    )
    print(json.dumps(dict(publication.mutation_result)), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
