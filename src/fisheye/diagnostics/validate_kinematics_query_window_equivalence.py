"""Prove one bounded kinematics query export equals its unbounded v1 slice."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

from fisheye.analytics_exports.kinematics_samples import (
    KINEMATICS_PROJECTION_SCHEMA_VERSION,
    KINEMATICS_PROJECTION_SCHEMA_VERSION_V2,
    KINEMATICS_SCIENTIFIC_DTYPES,
    _NUMPY_DTYPES,
    _ProjectedPayloadHasher,
    kinematics_projection_contract,
    validate_kinematics_samples_export_payload,
)
from fisheye.analytics_exports.contracts import KINEMATICS_SAMPLES_TABLE
from fisheye.analytics_exports.publication import (
    export_manifest_path,
    manifest_selected_part_files_from_payload,
    sha256_file,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

EVIDENCE_SCHEMA_ID = "palette.kinematics_query.frame_window_equivalence"
EVIDENCE_SCHEMA_VERSION = 1


def _read_manifest(export_root: Path, export_run_id: str) -> tuple[Path, dict[str, Any]]:
    path = export_manifest_path(export_root, export_run_id).resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Export manifest is missing or unsafe: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {raw}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"Export manifest is not one JSON object: {path}")
    return path, value


def _decoded_receipt(
    *,
    export_root: Path,
    manifest: Mapping[str, Any],
    frame_start: int | None = None,
    frame_stop_exclusive: int | None = None,
) -> dict[str, Any]:
    parts = manifest_selected_part_files_from_payload(
        export_root,
        manifest,
        KINEMATICS_SAMPLES_TABLE,
        allow_legacy_layout=False,
    )
    dataset = ds.dataset([str(path) for path in parts], format="parquet")
    filter_expression = None
    if frame_start is not None or frame_stop_exclusive is not None:
        if frame_start is None or frame_stop_exclusive is None:
            raise ValueError("Decoded frame filter is incomplete.")
        axis = ds.field("source_acquisition_frame_index")
        filter_expression = (axis >= frame_start) & (axis < frame_stop_exclusive)
    hasher = _ProjectedPayloadHasher()
    scanner = dataset.scanner(
        columns=list(KINEMATICS_SCIENTIFIC_DTYPES),
        filter=filter_expression,
        batch_size=65_536,
    )
    for batch in scanner.to_batches():
        columns = {
            name: np.asarray(
                batch.column(batch.schema.get_field_index(name)).to_pylist(),
                dtype=_NUMPY_DTYPES[dtype_name],
            )
            for name, dtype_name in KINEMATICS_SCIENTIFIC_DTYPES.items()
        }
        hasher.update(columns)
    return hasher.finish()


def _require_benchmark_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if not any("benchmark" in component.lower() for component in output.parts):
        raise ValueError("Equivalence evidence must be benchmark-namespaced.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace immutable evidence: {output}")
    return output


def _write_strict_json(path: Path, value: Mapping[str, Any]) -> None:
    json.dumps(value, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary evidence path exists: {temporary}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_kinematics_query_window_equivalence(
    *,
    full_export_root: Path,
    full_export_run_id: str,
    bounded_export_root: Path,
    bounded_export_run_id: str,
    output: Path,
) -> dict[str, Any]:
    """Validate and persist an immutable decoded-equality receipt."""

    destination = _require_benchmark_output(output)
    full_root = full_export_root.expanduser().resolve()
    bounded_root = bounded_export_root.expanduser().resolve()
    if full_root == bounded_root:
        raise ValueError("Full and bounded exports must have distinct roots.")
    full_path, full = _read_manifest(full_root, full_export_run_id)
    bounded_path, bounded = _read_manifest(bounded_root, bounded_export_run_id)
    full_file_sha = sha256_file(full_path)
    bounded_file_sha = sha256_file(bounded_path)
    full_validation = validate_kinematics_samples_export_payload(full_root, full)
    bounded_validation = validate_kinematics_samples_export_payload(
        bounded_root, bounded
    )
    full_envelope = full["kinematics_samples_export"]
    bounded_envelope = bounded["kinematics_samples_export"]
    full_projection = full_envelope["projection_contract"]
    bounded_projection = bounded_envelope["projection_contract"]
    if (
        full_projection.get("schema_version")
        != KINEMATICS_PROJECTION_SCHEMA_VERSION
        or bounded_projection.get("schema_version")
        != KINEMATICS_PROJECTION_SCHEMA_VERSION_V2
    ):
        raise ValueError("Expected one unbounded v1 and one bounded v2 projection.")
    if full_envelope["source_binding"] != bounded_envelope["source_binding"]:
        raise ValueError("Full and bounded exports bind different source authority.")
    expected_bounded = kinematics_projection_contract(
        source_sample_rate_hz=float(full_projection["source_sample_rate_hz"]),
        requested_sample_rate_hz=float(full_projection["requested_sample_rate_hz"]),
        source_frame_start=bounded_projection.get("source_frame_start"),
        source_frame_stop_exclusive=bounded_projection.get(
            "source_frame_stop_exclusive"
        ),
    )
    if dict(bounded_projection) != expected_bounded:
        raise ValueError("Bounded projection is not the exact v2 form of v1.")
    frame_start = int(bounded_projection["source_frame_start"])
    frame_stop = int(bounded_projection["source_frame_stop_exclusive"])
    filtered_full = _decoded_receipt(
        export_root=full_root,
        manifest=full,
        frame_start=frame_start,
        frame_stop_exclusive=frame_stop,
    )
    recomputed_bounded = _decoded_receipt(
        export_root=bounded_root,
        manifest=bounded,
    )
    expected_bounded_payload = bounded_envelope["projected_payload"]
    if recomputed_bounded != expected_bounded_payload:
        raise ValueError("Bounded decoded payload differs from its manifest receipt.")
    if filtered_full != recomputed_bounded:
        raise ValueError("Bounded payload differs from the same v1 frame slice.")
    if (
        sha256_file(full_path) != full_file_sha
        or sha256_file(bounded_path) != bounded_file_sha
    ):
        raise RuntimeError("An export manifest changed during equality validation.")
    git = get_git_info(Path(__file__).resolve().parents[3])
    body = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "palette_git": git,
        "evidence_eligible": git.get("is_dirty") is False,
        "full_export": {
            "root": str(full_root),
            "run_id": full_export_run_id,
            "manifest_path": str(full_path),
            "manifest_file_sha256": full_file_sha,
            "validation": full_validation,
        },
        "bounded_export": {
            "root": str(bounded_root),
            "run_id": bounded_export_run_id,
            "manifest_path": str(bounded_path),
            "manifest_file_sha256": bounded_file_sha,
            "validation": bounded_validation,
        },
        "frame_interval": {
            "start": frame_start,
            "stop_exclusive": frame_stop,
            "frame_count": frame_stop - frame_start,
        },
        "logical_equality": {
            "equal": True,
            "receipt": recomputed_bounded,
        },
        "manifest_nonmutation": True,
        "production_state_changes": [],
        "promotion_authorized": False,
    }
    document = {
        "schema_id": EVIDENCE_SCHEMA_ID,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "payload": body,
        "payload_digest": canonical_json_sha256(body),
    }
    _write_strict_json(destination, document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export-root", type=Path, required=True)
    parser.add_argument("--full-export-run-id", required=True)
    parser.add_argument("--bounded-export-root", type=Path, required=True)
    parser.add_argument("--bounded-export-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_kinematics_query_window_equivalence(
        full_export_root=args.full_export_root,
        full_export_run_id=args.full_export_run_id,
        bounded_export_root=args.bounded_export_root,
        bounded_export_run_id=args.bounded_export_run_id,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["payload"]["status"],
                "output": str(args.output.expanduser().resolve()),
                "payload_digest": result["payload_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EVIDENCE_SCHEMA_ID",
    "EVIDENCE_SCHEMA_VERSION",
    "validate_kinematics_query_window_equivalence",
]
