"""Place validated merged-source recoveries beside their source recordings.

The recovery collection remains an immutable record of the initial outputs.
This relocation records both paths without changing any recovered Zarr payload.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from fisheye.training.recover_merged_training_recording import (
    validate_recovered_recording,
)


RECEIPT_NAME = "merged_pose_detect_recovery_relocation_v001.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_index_sha256(attrs: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            attrs["array_sha256"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def build_relocation_plan(
    *,
    collection: dict[str, Any],
    collection_dir: Path,
    detect_manifest: dict[str, Any],
    recordings_root: Path,
) -> list[dict[str, Any]]:
    """Bind every recovered ID to its original recording directory name."""
    if (
        collection.get("schema_id") != "palette.training.merged_recovery_collection.v1"
        or collection.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Unsupported or selector-eligible recovery collection")
    if collection.get("source_detect_set_id") != detect_manifest.get("set_id"):
        raise ValueError("Detection source set disagrees with recovery collection")
    source_rows = detect_manifest["merged_export"]["source_datasets"]
    source_names: dict[str, str] = {}
    for row in source_rows:
        recording_id = str(row["dataset_id"]).split(":", 1)[0]
        name = str(row["name"])
        if (
            recording_id in source_names
            or not recording_id
            or "/" in recording_id
            or recording_id in {".", ".."}
            or not name.startswith(recording_id + "_")
            or "/" in name
            or name in {".", ".."}
        ):
            raise ValueError(f"Unsafe or duplicate source recording: {recording_id}")
        source_names[recording_id] = name
    entries = collection.get("recordings")
    if not isinstance(entries, list) or len(entries) != len(source_names):
        raise ValueError("Recovery collection does not cover the detection cohort")
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in entries:
        recording_id = str(row["recording_id"])
        if recording_id not in source_names or recording_id in seen:
            raise ValueError(
                f"Unknown or duplicate recovered recording: {recording_id}"
            )
        seen.add(recording_id)
        old = collection_dir / f"{recording_id}_recovered_training.zarr"
        if Path(row["path"]) != old:
            raise ValueError(f"Recovery collection path disagrees: {recording_id}")
        name = source_names[recording_id]
        new = recordings_root / name / "zarr" / f"{name}_recovered_training.zarr"
        plan.append({**row, "source_path": str(old), "destination_path": str(new)})
    if seen != set(source_names):
        raise ValueError("Recovery collection is missing detection recordings")
    for key in ("pose_rows", "detect_rows", "detect_only_rows"):
        if sum(int(row[key]) for row in plan) != int(collection[key]):
            raise ValueError(f"Recovery collection {key} total disagrees")
    if sum(int(row["pose_rows"]) > 0 for row in plan) != int(
        collection["recordings_with_pose"]
    ):
        raise ValueError("Recovery collection pose-recording total disagrees")
    return plan


def relocate_full_recovery(
    *,
    collection_path: Path,
    detect_manifest_path: Path,
    recordings_root: Path,
) -> dict[str, Any]:
    """Validate, atomically rename each archive, then publish a path receipt.

    A partly completed move can be resumed: an archive must exist at exactly
    one of its recorded old or new paths and must match its content declaration.
    """
    collection = json.loads(collection_path.read_text())
    detect_manifest = json.loads(detect_manifest_path.read_text())
    if collection.get("source_detect_manifest_sha256") != _sha256_file(
        detect_manifest_path
    ):
        raise ValueError("Detection manifest bytes disagree with recovery receipt")
    plan = build_relocation_plan(
        collection=collection,
        collection_dir=collection_path.parent,
        detect_manifest=detect_manifest,
        recordings_root=recordings_root,
    )
    receipt_path = recordings_root / "_index" / RECEIPT_NAME
    if receipt_path.exists():
        raise FileExistsError(f"Relocation receipt already exists: {receipt_path}")
    for row in plan:
        source = Path(row["source_path"])
        destination = Path(row["destination_path"])
        if source.is_symlink() or destination.is_symlink():
            raise ValueError(f"Refusing symlinked recovery archive: {source}")
        if source.exists() == destination.exists():
            raise ValueError(f"Recovery must exist at exactly one path: {source}")
        current = source if source.exists() else destination
        attrs = validate_recovered_recording(
            current,
            expected_recording_id=row["recording_id"],
            require_source_only=True,
        )
        if (
            int(attrs["pose_source_row_count"]) != int(row["pose_rows"])
            or int(attrs["detect_source_row_count"]) != int(row["detect_rows"])
            or int(attrs["detect_only_row_count"]) != int(row["detect_only_rows"])
            or _digest_index_sha256(attrs) != row["array_digest_index_sha256"]
        ):
            raise ValueError(f"Recovered archive disagrees with collection: {current}")
    for ordinal, row in enumerate(plan, start=1):
        source = Path(row["source_path"])
        destination = Path(row["destination_path"])
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Relocation destination appeared: {destination}")
            if source.stat().st_dev != destination.parent.stat().st_dev:
                raise ValueError(f"Relocation would cross filesystems: {source}")
            original_inode = source.stat().st_ino
            os.rename(source, destination)
            if destination.stat().st_ino != original_inode:
                raise ValueError(
                    f"Archive identity changed during relocation: {destination}"
                )
        attrs = validate_recovered_recording(
            destination,
            expected_recording_id=row["recording_id"],
            require_source_only=True,
        )
        if _digest_index_sha256(attrs) != row["array_digest_index_sha256"]:
            raise ValueError(f"Relocated archive digest index changed: {destination}")
        print(f"[{ordinal}/{len(plan)}] {destination}", flush=True)
    receipt = {
        "schema_id": "palette.training.merged_recovery_relocation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_collection_path": str(collection_path),
        "source_collection_sha256": _sha256_file(collection_path),
        "source_detect_manifest_path": str(detect_manifest_path),
        "source_detect_manifest_sha256": _sha256_file(detect_manifest_path),
        "recording_count": len(plan),
        "recordings_with_pose": collection["recordings_with_pose"],
        "pose_rows": collection["pose_rows"],
        "detect_rows": collection["detect_rows"],
        "detect_only_rows": collection["detect_only_rows"],
        "stage_selector_eligible": False,
        "recordings": plan,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        raise FileExistsError(f"Relocation receipt appeared: {receipt_path}")
    temporary = receipt_path.with_name(f".{receipt_path.name}.partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--detect-manifest", required=True, type=Path)
    parser.add_argument("--recordings-root", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = relocate_full_recovery(
        collection_path=args.collection,
        detect_manifest_path=args.detect_manifest,
        recordings_root=args.recordings_root,
    )
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "recordings"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
