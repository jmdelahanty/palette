#!/usr/bin/env python3
"""Validate and publish the final handoff for one Crimson storage candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any, Mapping, Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.clipped_keypoint_finalization import (
    validate_clipped_keypoint_finalization_receipt,
)
from fisheye.shared.zarr.clipped_refined_detection_finalization import (
    validate_clipped_refined_detection_finalization_receipt,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


HANDOFF_SCHEMA_ID = "palette.crimson.storage_candidate_handoff"
HANDOFF_SCHEMA_VERSION = 1
_HEX = frozenset("0123456789abcdef")
_FAMILIES = {
    "canonical_detection": "detect_runs",
    "refined_detection": "refined_detect_runs",
    "crop_geometry": "crop_runs",
    "raw_keypoints": "keypoints_runs",
    "keypoint_quality": "keypoint_quality_runs",
    "refined_keypoints": "refined_keypoints_runs",
    "body_frame": "analysis/body_frame_runs",
}


def _read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return text


def _require_commit(value: object) -> str:
    text = str(value).strip().lower()
    if len(text) != 40 or any(character not in _HEX for character in text):
        raise ValueError("crimson_contract_commit must be a full Git commit.")
    return text


def _git_state() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]

    def run(*args: str) -> str:
        return subprocess.run(
            ("git", "-C", str(repository), *args),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "repository": str(repository),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_clean": run("status", "--short") == "",
    }


def _metadata_inventory(root: Path) -> dict[str, object]:
    files = 0
    apparent_bytes = 0
    metadata_files = 0
    metadata_digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Candidate Zarr contains a symlink: {path}")
        if not path.is_file():
            continue
        files += 1
        apparent_bytes += int(path.stat().st_size)
        if path.name != "zarr.json":
            continue
        metadata_files += 1
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        metadata_digest.update(len(relative).to_bytes(8, "little"))
        metadata_digest.update(relative)
        metadata_digest.update(len(payload).to_bytes(8, "little"))
        metadata_digest.update(payload)
    return {
        "file_count": files,
        "metadata_file_count": metadata_files,
        "apparent_bytes": apparent_bytes,
        "metadata_sha256": metadata_digest.hexdigest(),
    }


def _consolidated_declaration(root: Path, run_path: str) -> Mapping[str, Any]:
    document = _read_json(root / "zarr.json")
    consolidated = document.get("consolidated_metadata")
    if not isinstance(consolidated, Mapping) or consolidated.get("kind") != "inline":
        raise ValueError(f"Candidate store lacks inline consolidated metadata: {root}")
    metadata = consolidated.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Candidate store has malformed consolidated metadata: {root}")
    declaration = metadata.get(run_path)
    if not isinstance(declaration, Mapping):
        raise ValueError(f"Consolidated metadata omits {run_path!r} in {root}.")
    return declaration


def _dimensions(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Run manifest lacks its payload.")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, Mapping):
        logical = payload.get("logical_schema")
        dimensions = logical.get("dimensions") if isinstance(logical, Mapping) else None
    if not isinstance(dimensions, Mapping):
        raise ValueError("Run manifest lacks exact dimensions.")
    return dimensions


def _artifact(
    *,
    stage: str,
    archive: Path,
    run_id: str,
    expected_n_frames: int,
    expected_n_instances: int,
) -> dict[str, object]:
    root = archive.expanduser().resolve()
    family = _FAMILIES[stage]
    run_path = f"{family}/{run_id}"
    direct_path = root / run_path / "zarr.json"
    direct = _read_json(direct_path)
    attributes = direct.get("attributes")
    manifest = attributes.get("run_manifest") if isinstance(attributes, Mapping) else None
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{stage} run lacks its exact run_manifest: {direct_path}")
    dimensions = _dimensions(manifest)
    observed_frames = dimensions.get("n_frames")
    observed_instances = dimensions.get("n_instances")
    if observed_frames != expected_n_frames:
        raise ValueError(
            f"{stage} n_frames mismatch: expected {expected_n_frames}, "
            f"found {observed_frames}."
        )
    if stage not in {"canonical_detection"} and observed_instances != (
        expected_n_instances
    ):
        raise ValueError(
            f"{stage} n_instances mismatch: expected {expected_n_instances}, "
            f"found {observed_instances}."
        )
    consolidated = _consolidated_declaration(root, run_path)
    consolidated_attributes = consolidated.get("attributes")
    consolidated_manifest = (
        consolidated_attributes.get("run_manifest")
        if isinstance(consolidated_attributes, Mapping)
        else None
    )
    if consolidated_manifest != manifest:
        raise ValueError(f"{stage} direct/consolidated run manifests differ.")
    payload = manifest["payload"]
    publication = payload.get("publication")
    if isinstance(publication, Mapping) and publication.get(
        "stage_selector_eligible"
    ) is not False:
        raise ValueError(f"{stage} unexpectedly became selector eligible.")
    logical = payload.get("logical_content")
    return {
        "stage": stage,
        "server_path": str(root),
        "macos_path": _macos_path(root),
        "run_id": run_id,
        "run_path": run_path,
        "manifest_payload_digest": manifest.get("payload_digest"),
        "manifest_digest": canonical_json_sha256(manifest),
        "logical_content_digest": (
            logical.get("digest") if isinstance(logical, Mapping) else None
        ),
        "dimensions": dict(dimensions),
        "direct_consolidated_manifest_equal": True,
        "inventory": _metadata_inventory(root),
    }


def _macos_path(path: Path) -> str:
    text = str(path)
    prefix = "/groups/johnson/johnsonlab/jeremy/"
    if text.startswith(prefix):
        return "/Volumes/johnsonlab/jeremy/" + text[len(prefix) :]
    return text


def _complete_result(path: Path, *, label: str) -> dict[str, Any]:
    result = _read_json(path.expanduser().resolve())
    if result.get("status") != "complete":
        raise ValueError(f"{label} result is not complete.")
    if result.get("selector_eligible") is not False:
        raise ValueError(f"{label} result became selector eligible.")
    if result.get("registry_registered") is not False:
        raise ValueError(f"{label} result became registry registered.")
    if result.get("production_state_changes") != []:
        raise ValueError(f"{label} reports production-state changes.")
    return result


def finalize_crimson_storage_candidate(
    *,
    candidate_id: str,
    classification: str,
    expected_n_frames: int,
    expected_n_instances: int,
    canonical_archive: Path,
    canonical_run: str,
    refined_result_path: Path,
    crop_result_path: Path,
    keypoint_result_path: Path,
    crimson_contract_commit: str,
    crimson_contract_sha256: str,
    expected_palette_commit: str,
    output: Path,
) -> dict[str, object]:
    """Reopen every final receipt and make the candidate visible to Crimson."""

    candidate = str(candidate_id).strip()
    if not candidate or "/" in candidate:
        raise ValueError("candidate_id must be one path-safe component.")
    if classification not in {"integration_fixture", "full_duration_fixture"}:
        raise ValueError("Unsupported candidate classification.")
    if type(expected_n_frames) is not int or expected_n_frames <= 0:
        raise ValueError("expected_n_frames must be a positive exact integer.")
    if type(expected_n_instances) is not int or expected_n_instances <= 0:
        raise ValueError("expected_n_instances must be a positive exact integer.")
    destination = output.expanduser().resolve()
    if destination.name != "handoff_manifest.json":
        raise ValueError("Candidate handoff must be named handoff_manifest.json.")
    if destination.exists():
        raise FileExistsError(f"Candidate handoff already exists: {destination}")
    if ".palette_benchmarks" not in destination.parts:
        raise ValueError("Crimson candidates must remain below .palette_benchmarks.")

    refined_result = _complete_result(refined_result_path, label="refined detection")
    crop_result = _complete_result(crop_result_path, label="crop geometry")
    keypoint_result = _complete_result(keypoint_result_path, label="keypoint chain")

    refined_receipt = _read_json(Path(refined_result["finalization_receipt_path"]))
    refined_errors = validate_clipped_refined_detection_finalization_receipt(
        refined_receipt
    )
    if refined_errors:
        raise ValueError("Invalid refined finalization receipt: " + "; ".join(refined_errors))
    keypoint_receipt = _read_json(Path(keypoint_result["finalization_receipt_path"]))
    keypoint_errors = validate_clipped_keypoint_finalization_receipt(keypoint_receipt)
    if keypoint_errors:
        raise ValueError("Invalid keypoint finalization receipt: " + "; ".join(keypoint_errors))

    artifacts: dict[str, dict[str, object]] = {}
    artifacts["canonical_detection"] = _artifact(
        stage="canonical_detection",
        archive=canonical_archive,
        run_id=canonical_run,
        expected_n_frames=expected_n_frames,
        expected_n_instances=expected_n_instances,
    )
    artifacts["refined_detection"] = _artifact(
        stage="refined_detection",
        archive=Path(refined_result["output_archive"]),
        run_id=str(refined_result["output_run_id"]),
        expected_n_frames=expected_n_frames,
        expected_n_instances=expected_n_instances,
    )
    artifacts["crop_geometry"] = _artifact(
        stage="crop_geometry",
        archive=Path(crop_result["output_archive"]),
        run_id=str(crop_result["output_run_id"]),
        expected_n_frames=expected_n_frames,
        expected_n_instances=expected_n_instances,
    )
    outputs = keypoint_receipt["payload"]["outputs"]
    for stage in (
        "raw_keypoints",
        "keypoint_quality",
        "refined_keypoints",
        "body_frame",
    ):
        binding = outputs[stage]
        artifacts[stage] = _artifact(
            stage=stage,
            archive=Path(binding["path"]),
            run_id=str(binding["run_id"]),
            expected_n_frames=expected_n_frames,
            expected_n_instances=expected_n_instances,
        )

    palette = _git_state()
    expected_palette = _require_commit(expected_palette_commit)
    if palette["commit"] != expected_palette:
        raise ValueError(
            "Palette runtime commit differs from the candidate plan: "
            f"expected {expected_palette}, found {palette['commit']}."
        )
    if palette["worktree_clean"] is not True:
        raise ValueError("Palette runtime worktree must be clean for final handoff.")
    payload: dict[str, object] = {
        "status": "complete",
        "candidate_id": candidate,
        "classification": classification,
        "benchmark_only": True,
        "selector_eligible": False,
        "registry_registered": False,
        "production_state_changes": [],
        "dimensions": {
            "n_frames": expected_n_frames,
            "n_instances": expected_n_instances,
        },
        "artifacts": artifacts,
        "video_copy_included": False,
        "analysis_crop_pixels_included": False,
        "pixel_materialization_role": "external_compute_input_only",
        "receipts": {
            "refined_detection": {
                "path": str(Path(refined_result["finalization_receipt_path"]).resolve()),
                "payload_digest": refined_receipt["payload_digest"],
            },
            "keypoints": {
                "path": str(Path(keypoint_result["finalization_receipt_path"]).resolve()),
                "payload_digest": keypoint_receipt["payload_digest"],
            },
        },
        "crimson_contract": {
            "commit": _require_commit(crimson_contract_commit),
            "document_sha256": _require_sha256(
                crimson_contract_sha256, name="crimson contract document"
            ),
        },
        "palette": palette,
        "execution": {
            "hostname": socket.gethostname(),
            "lsb_job_id": os.environ.get("LSB_JOBID"),
        },
        "promotion_semantics": (
            "integration_only_not_full_duration_evidence"
            if classification == "integration_fixture"
            else "full_duration_candidate_requires_crimson_gate"
        ),
    }
    handoff = {
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
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--expected-n-frames", type=int, required=True)
    parser.add_argument("--expected-n-instances", type=int, required=True)
    parser.add_argument("--canonical-archive", type=Path, required=True)
    parser.add_argument("--canonical-run", required=True)
    parser.add_argument("--refined-result", type=Path, required=True)
    parser.add_argument("--crop-result", type=Path, required=True)
    parser.add_argument("--keypoint-result", type=Path, required=True)
    parser.add_argument("--crimson-contract-commit", required=True)
    parser.add_argument("--crimson-contract-sha256", required=True)
    parser.add_argument("--expected-palette-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handoff = finalize_crimson_storage_candidate(
        candidate_id=args.candidate_id,
        classification=args.classification,
        expected_n_frames=args.expected_n_frames,
        expected_n_instances=args.expected_n_instances,
        canonical_archive=args.canonical_archive,
        canonical_run=args.canonical_run,
        refined_result_path=args.refined_result,
        crop_result_path=args.crop_result,
        keypoint_result_path=args.keypoint_result,
        crimson_contract_commit=args.crimson_contract_commit,
        crimson_contract_sha256=args.crimson_contract_sha256,
        expected_palette_commit=args.expected_palette_commit,
        output=args.output,
    )
    print(json.dumps(handoff, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HANDOFF_SCHEMA_ID",
    "HANDOFF_SCHEMA_VERSION",
    "finalize_crimson_storage_candidate",
]
