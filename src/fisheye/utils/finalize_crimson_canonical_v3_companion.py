#!/usr/bin/env python3
"""Issue a full Crimson handoff with only canonical detection replaced by v3.

The original seven-surface handoff and all of its stores remain immutable.
This benchmark-only gate validates one separately published canonical-v3 store,
proves that its decoded contract and physical plan equal the original v2
canonical companion, and emits a new full handoff that reuses the other six
artifact records verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import sha256_file, utc_now
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION,
    validate_canonical_detection_run_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.finalize_crimson_storage_candidate import (
    HANDOFF_SCHEMA_ID,
    HANDOFF_SCHEMA_VERSION,
    _artifact,
    _git_state,
    _read_json,
    _require_commit,
    _require_sha256,
)


COMPANION_SCHEMA_ID = "palette.crimson.canonical_v3_companion_handoff"
COMPANION_SCHEMA_VERSION = 1
_EXPECTED_ARTIFACTS = {
    "canonical_detection",
    "refined_detection",
    "crop_geometry",
    "raw_keypoints",
    "keypoint_quality",
    "refined_keypoints",
    "body_frame",
}


def _manifest_for_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    archive = Path(str(artifact.get("server_path") or "")).expanduser().resolve()
    run_path = str(artifact.get("run_path") or "").strip().strip("/")
    if not archive.is_dir() or not run_path:
        raise ValueError("Canonical artifact path is incomplete.")
    declaration = _read_json(archive / run_path / "zarr.json")
    attributes = declaration.get("attributes")
    manifest = attributes.get("run_manifest") if isinstance(attributes, Mapping) else None
    if not isinstance(manifest, Mapping):
        raise ValueError("Canonical artifact lacks its persisted run_manifest.")
    return dict(manifest)


def _payload(manifest: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} manifest lacks its exact payload.")
    return payload


def _require_equal_contract(
    source_manifest: Mapping[str, Any],
    companion_manifest: Mapping[str, Any],
) -> dict[str, object]:
    if source_manifest.get("schema_version") != (
        CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Base canonical companion must carry native manifest v2.")
    if companion_manifest.get("schema_version") != (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Replacement canonical companion must carry manifest v3.")
    errors = validate_canonical_detection_run_manifest(companion_manifest)
    if errors:
        raise ValueError("Canonical-v3 manifest is invalid: " + "; ".join(errors))
    source = _payload(source_manifest, name="source")
    companion = _payload(companion_manifest, name="companion")
    if companion.get("source_evidence_kind") != "native_detection":
        raise ValueError("Canonical-v3 companion is not bound to native detection evidence.")
    coordinate = companion.get("coordinate_contract")
    if not isinstance(coordinate, Mapping):
        raise ValueError("Canonical-v3 companion lacks its coordinate catalog.")

    compared_fields = (
        "logical_schema",
        "storage_plan",
        "logical_content",
        "source_evidence",
    )
    differences = [
        name for name in compared_fields if source.get(name) != companion.get(name)
    ]
    source_publication = source.get("publication")
    companion_publication = companion.get("publication")
    if not isinstance(source_publication, Mapping) or not isinstance(
        companion_publication, Mapping
    ):
        differences.append("publication")
    elif source_publication.get("metadata_declarations_digest") != (
        companion_publication.get("metadata_declarations_digest")
    ):
        differences.append("publication.metadata_declarations_digest")
    if differences:
        raise ValueError(
            "Canonical-v3 companion differs from v2 outside the manifest/catalog "
            f"envelope: {sorted(set(differences))!r}."
        )
    logical = companion["logical_content"]
    return {
        "base_manifest_schema_version": source_manifest["schema_version"],
        "companion_manifest_schema_version": companion_manifest["schema_version"],
        "logical_content_digest": logical["digest"],
        "storage_plan_digest": canonical_json_sha256(companion["storage_plan"]),
        "source_evidence_digest": canonical_json_sha256(companion["source_evidence"]),
        "metadata_declarations_digest": companion_publication[
            "metadata_declarations_digest"
        ],
        "coordinate_catalog_digest": coordinate["digest"],
        "equal_fields": list(compared_fields),
    }


def finalize_crimson_canonical_v3_companion(
    *,
    base_handoff_path: Path,
    expected_base_handoff_sha256: str,
    canonical_result_path: Path,
    canonical_archive: Path,
    canonical_run: str,
    crimson_validation_commit: str,
    expected_palette_commit: str,
    output: Path,
) -> dict[str, object]:
    """Validate the replacement and write one new complete seven-surface handoff."""

    base_path = base_handoff_path.expanduser().resolve()
    expected_base_sha = _require_sha256(
        expected_base_handoff_sha256,
        name="base handoff",
    )
    if sha256_file(base_path) != expected_base_sha:
        raise ValueError("Base handoff bytes differ from the expected immutable pin.")
    base = _read_json(base_path)
    if (
        base.get("schema_id") != HANDOFF_SCHEMA_ID
        or base.get("schema_version") != HANDOFF_SCHEMA_VERSION
    ):
        raise ValueError("Base handoff schema is not the frozen full candidate contract.")
    base_payload = base.get("payload")
    if not isinstance(base_payload, Mapping):
        raise ValueError("Base handoff lacks its payload.")
    if base.get("payload_digest") != canonical_json_sha256(base_payload):
        raise ValueError("Base handoff payload digest is invalid.")
    if (
        base_payload.get("status") != "complete"
        or base_payload.get("benchmark_only") is not True
        or base_payload.get("selector_eligible") is not False
        or base_payload.get("registry_registered") is not False
        or base_payload.get("production_state_changes") != []
    ):
        raise ValueError("Base handoff is not complete benchmark-only evidence.")
    artifacts = base_payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _EXPECTED_ARTIFACTS:
        raise ValueError("Base handoff does not contain the exact seven artifacts.")
    base_canonical = artifacts["canonical_detection"]
    if not isinstance(base_canonical, Mapping):
        raise ValueError("Base canonical artifact record is malformed.")

    result_path = canonical_result_path.expanduser().resolve()
    result = _read_json(result_path)
    if (
        result.get("status") != "complete"
        or result.get("selector_eligible") is not False
        or result.get("registry_registered") is not False
        or result.get("production_state_changes") != []
        or result.get("coordinate_catalog") is not True
        or result.get("run_manifest_schema_version")
        != CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Canonical-v3 adapter result is incomplete or unsafe.")
    archive = canonical_archive.expanduser().resolve()
    if Path(str(result.get("output_archive") or "")).resolve() != archive:
        raise ValueError("Canonical-v3 result names a different output archive.")
    if result.get("output_run_id") != canonical_run:
        raise ValueError("Canonical-v3 result names a different run.")

    dimensions = base_canonical.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("Base canonical artifact lacks dimensions.")
    replacement = _artifact(
        stage="canonical_detection",
        archive=archive,
        run_id=canonical_run,
        expected_n_frames=int(dimensions["n_frames"]),
        expected_n_instances=int(dimensions["n_instances"]),
    )
    base_manifest = _manifest_for_artifact(base_canonical)
    replacement_manifest = _manifest_for_artifact(replacement)
    equivalence = _require_equal_contract(base_manifest, replacement_manifest)
    if replacement["logical_content_digest"] != base_canonical.get(
        "logical_content_digest"
    ):
        raise ValueError("Canonical-v3 artifact logical digest differs from v2.")
    if replacement["dimensions"] != dict(dimensions):
        raise ValueError("Canonical-v3 artifact dimensions differ from v2.")

    destination = output.expanduser().resolve()
    if destination.name != "handoff_manifest.json":
        raise ValueError("Companion handoff must be named handoff_manifest.json.")
    if destination.exists():
        raise FileExistsError(f"Companion handoff already exists: {destination}")
    if ".palette_benchmarks" not in destination.parts:
        raise ValueError("Companion handoff must remain below .palette_benchmarks.")
    if destination.parent != archive.parent:
        raise ValueError("Companion handoff and canonical store must share one root.")

    palette = _git_state()
    expected_palette = _require_commit(expected_palette_commit)
    if palette["commit"] != expected_palette or palette["worktree_clean"] is not True:
        raise ValueError("Palette runtime revision is not the expected clean commit.")

    resolved_artifacts = {name: dict(value) for name, value in artifacts.items()}
    resolved_artifacts["canonical_detection"] = replacement
    payload = dict(base_payload)
    payload.update(
        {
            "candidate_id": f"{base_payload['candidate_id']}_canonical_v3",
            "artifacts": resolved_artifacts,
            "palette": palette,
            "execution": {
                "hostname": socket.gethostname(),
                "lsb_job_id": os.environ.get("LSB_JOBID"),
            },
            "receipts": {
                **dict(base_payload.get("receipts") or {}),
                "base_handoff": {
                    "path": str(base_path),
                    "sha256": expected_base_sha,
                    "payload_digest": base["payload_digest"],
                },
                "canonical_v3_companion": {
                    "adapter_result_path": str(result_path),
                    "adapter_result_sha256": sha256_file(result_path),
                    "adapter_receipt_path": result.get("adapter_receipt_path"),
                    "adapter_receipt_digest": result.get("adapter_receipt_digest"),
                },
            },
            "crimson_validation": {
                "commit": _require_commit(crimson_validation_commit),
                "finding": "canonical_v2_rejected_v3_required_other_six_surfaces_passed",
            },
            "canonical_v3_companion": {
                "schema_id": COMPANION_SCHEMA_ID,
                "schema_version": COMPANION_SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "base_canonical": dict(base_canonical),
                "equivalence": equivalence,
                "only_replaced_artifact": "canonical_detection",
                "unchanged_artifacts": sorted(
                    _EXPECTED_ARTIFACTS - {"canonical_detection"}
                ),
            },
        }
    )
    handoff: dict[str, object] = {
        "schema_id": HANDOFF_SCHEMA_ID,
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, handoff)
    return handoff


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-handoff", type=Path, required=True)
    parser.add_argument("--expected-base-handoff-sha256", required=True)
    parser.add_argument("--canonical-result", type=Path, required=True)
    parser.add_argument("--canonical-archive", type=Path, required=True)
    parser.add_argument("--canonical-run", required=True)
    parser.add_argument("--crimson-validation-commit", required=True)
    parser.add_argument("--expected-palette-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = finalize_crimson_canonical_v3_companion(
        base_handoff_path=args.base_handoff,
        expected_base_handoff_sha256=args.expected_base_handoff_sha256,
        canonical_result_path=args.canonical_result,
        canonical_archive=args.canonical_archive,
        canonical_run=args.canonical_run,
        crimson_validation_commit=args.crimson_validation_commit,
        expected_palette_commit=args.expected_palette_commit,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPANION_SCHEMA_ID",
    "COMPANION_SCHEMA_VERSION",
    "finalize_crimson_canonical_v3_companion",
]
