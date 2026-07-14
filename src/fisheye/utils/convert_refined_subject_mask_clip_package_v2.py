#!/usr/bin/env python3
"""Convert a completed v1 refined-mask clip package to global encoded chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import Any, Sequence

from fisheye.shared.refined_subject_mask_encoded_chunks import (
    ENCODED_MASK_PAYLOAD_NAME,
    ENCODED_PACKAGE_SCHEMA_ID,
    build_global_encoded_mask_payload,
)
from fisheye.utils.finalize_subject_mask_clip_package import PACKAGE_SCHEMA_ID, _write_package


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    members = tar.getmembers()
    for member in members:
        resolved = (root / member.name).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Unsafe package member path: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"Refusing package link member: {member.name!r}")
    tar.extractall(target, members=members)


def convert_package(
    *,
    source_package: Path,
    output_package: Path,
    grid_manifest: Path,
    staging_root: Path | None = None,
    copy_workers: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_package = source_package.expanduser().resolve()
    output_package = output_package.expanduser().resolve()
    if source_package == output_package:
        raise ValueError("v2 conversion requires a distinct output package path.")
    if not source_package.is_file():
        raise FileNotFoundError(source_package)

    temp_parent = staging_root.expanduser().resolve() if staging_root is not None else None
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="palette_mask_package_v2_", dir=temp_parent) as tmp:
        extracted = Path(tmp)
        with tarfile.open(source_package, "r:gz") as tar:
            _safe_extract(tar, extracted)
        manifest_path = extracted / "package.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_id") not in {PACKAGE_SCHEMA_ID, ENCODED_PACKAGE_SCHEMA_ID}:
            raise ValueError(f"Unsupported source package schema: {manifest.get('schema_id')!r}")
        run_group_path = PurePosixPath(str(manifest.get("run_group_path") or ""))
        if len(run_group_path.parts) != 2 or run_group_path.parts[0] != "refined_subject_masks_runs":
            raise ValueError(f"Invalid package run_group_path={str(run_group_path)!r}")
        run_name = run_group_path.parts[1]
        run_path = extracted / run_group_path.parts[0] / run_name
        encoded_path = extracted / ENCODED_MASK_PAYLOAD_NAME
        encoded_summary = build_global_encoded_mask_payload(
            run_path=run_path,
            grid_manifest_path=grid_manifest,
            payload_path=encoded_path,
            copy_workers=int(copy_workers),
        )
        metadata = {
            key: value
            for key, value in manifest.items()
            if key not in {"schema_id", "created_at_utc", "package_path", "run_group_path"}
        }
        metadata.update(
            {
                "converted_from_package": str(source_package),
                "converted_from_schema_id": str(manifest.get("schema_id") or ""),
                "encoded_global_masks_roi": encoded_summary,
            }
        )
        artifact = _write_package(
            staged_zarr=extracted,
            refined_run=run_name,
            package_path=output_package,
            metadata=metadata,
            overwrite=bool(overwrite),
            schema_id=ENCODED_PACKAGE_SCHEMA_ID,
            encoded_payload_path=encoded_path,
        )
    return {
        "status": "ok",
        "schema_id": ENCODED_PACKAGE_SCHEMA_ID,
        "source_package": str(source_package),
        "output_package": str(output_package),
        "artifact": artifact,
        "encoded_global_masks_roi": encoded_summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--output-package", required=True, type=Path)
    parser.add_argument("--grid-manifest", required=True, type=Path)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = convert_package(
        source_package=args.source_package,
        output_package=args.output_package,
        grid_manifest=args.grid_manifest,
        staging_root=args.staging_root,
        copy_workers=int(args.copy_workers),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
