"""Assemble clipped detection artifacts and atomically publish canonical v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.detection.clipped_native_artifact_io import (
    load_clipped_detection_artifact_members,
)
from fisheye.detection.clipped_native_binding import (
    bind_clipped_detection_artifacts,
)
from fisheye.detection.native_canonical_candidate import (
    write_native_clipped_detection_candidate,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.analysis_workflows.native_canonical_detection_publication import (
    publish_native_canonical_detection_candidate,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
)
from fisheye.utils.run_detection_artifact import (
    FRAME_MAPPING_MODE_CHOICES,
    FRAME_MAPPING_MODE_INDEXED,
)


CLIPPED_NATIVE_DETECTION_ASSEMBLY_SCHEMA_ID = (
    "palette.clipped_detection.native_assembly"
)
CLIPPED_NATIVE_DETECTION_ASSEMBLY_SCHEMA_VERSION = 1


def _model_digests(run_provenance: Mapping[str, Any]) -> set[str]:
    artifacts = run_provenance.get("input_artifacts")
    if not isinstance(artifacts, list):
        return set()
    return {
        str(item.get("sha256") or "")
        for item in artifacts
        if isinstance(item, Mapping) and item.get("role") == "detect_model"
    }


def assemble_and_publish_clipped_native_detection(
    *,
    analysis_zarr: Path,
    work_unit_reports: Sequence[Path],
    recording_frame_index: Path | None,
    frame_mapping_mode: str = FRAME_MAPPING_MODE_INDEXED,
    recording_identity: str,
    n_frames: int,
    source_width: int,
    source_height: int,
    run_id: str,
    candidate_zarr: Path,
    producer_id: str,
    producer_version: str,
    source_frame_authority: Mapping[str, str],
    source_pixel_authority: Mapping[str, str],
    model_artifact_sha256: str,
    workflow_id: str,
    result_json: Path | None = None,
    copy_backend: str = "python",
) -> dict[str, object]:
    """Run the complete artifact -> native-v2 -> atomic publication boundary."""

    if not work_unit_reports:
        raise ValueError("At least one work-unit report is required.")
    if frame_mapping_mode not in FRAME_MAPPING_MODE_CHOICES:
        raise ValueError(
            f"frame_mapping_mode must be one of {FRAME_MAPPING_MODE_CHOICES}."
        )
    if frame_mapping_mode == FRAME_MAPPING_MODE_INDEXED:
        if recording_frame_index is None:
            raise ValueError(
                "recording_frame_index mapping requires --recording-frame-index."
            )
    elif recording_frame_index is not None:
        raise ValueError(
            "--recording-frame-index is only valid with recording_frame_index mapping."
        )
    members, member_evidence = load_clipped_detection_artifact_members(
        work_unit_reports,
        analysis_zarr=analysis_zarr,
        recording_frame_index=recording_frame_index,
        recording_identity=recording_identity,
        n_frames=n_frames,
        source_width=source_width,
        source_height=source_height,
    )
    for evidence in member_evidence:
        provenance = evidence.get("run_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Detection artifact has no run provenance.")
        digests = _model_digests(provenance)
        if digests != {model_artifact_sha256}:
            raise ValueError(
                "Detection artifact model digest differs from the pinned model: "
                f"observed={sorted(digests)!r}."
            )

    bound = bind_clipped_detection_artifacts(
        members,
        recording_identity=recording_identity,
        n_frames=int(n_frames),
        source_width=int(source_width),
        source_height=int(source_height),
    )
    provenance_members = [
        {
            name: evidence[name]
            for name in (
                "report_path",
                "report_sha256",
                "receipt_path",
                "receipt_sha256",
                "artifact_group_path",
                "artifact_manifest_sha256",
                "run_group_tree_sha256",
            )
        }
        for evidence in member_evidence
    ]
    run_provenance: dict[str, object] = {
        "schema_id": CLIPPED_NATIVE_DETECTION_ASSEMBLY_SCHEMA_ID,
        "schema_version": CLIPPED_NATIVE_DETECTION_ASSEMBLY_SCHEMA_VERSION,
        "workflow_id": str(workflow_id),
        "recording_identity": str(recording_identity),
        "producer": {"id": str(producer_id), "version": str(producer_version)},
        "parameters": {
            "n_frames": int(n_frames),
            "source_width": int(source_width),
            "source_height": int(source_height),
            "member_count": len(members),
            "frame_mapping_mode": str(frame_mapping_mode),
        },
        "input_artifacts": [
            {"role": "detect_model", "sha256": str(model_artifact_sha256)}
        ],
        "members": provenance_members,
        "members_digest": canonical_json_sha256(provenance_members),
    }
    candidate = write_native_clipped_detection_candidate(
        bound,
        destination=candidate_zarr,
        run_id=run_id,
        recording_identity=recording_identity,
        producer_id=producer_id,
        producer_version=producer_version,
        source_frame_authority=source_frame_authority,
        source_pixel_authority=source_pixel_authority,
        model_artifact_sha256=model_artifact_sha256,
        run_provenance=run_provenance,
        coordinate_catalog=True,
        publication_selector_eligible=True,
    )
    publication = publish_native_canonical_detection_candidate(
        analysis_zarr=analysis_zarr,
        candidate_zarr=candidate.output_path,
        run_id=candidate.run_id,
        recording_identity=recording_identity,
        expected_manifest_schema_version=(
            CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        ),
        activate=True,
        copy_backend=copy_backend,
    )
    result: dict[str, object] = {
        "schema_id": CLIPPED_NATIVE_DETECTION_ASSEMBLY_SCHEMA_ID,
        "schema_version": CLIPPED_NATIVE_DETECTION_ASSEMBLY_SCHEMA_VERSION,
        "status": "complete",
        "workflow_id": str(workflow_id),
        "recording_identity": str(recording_identity),
        "work_unit_count": len(members),
        "candidate": dict(candidate.receipt),
        "publication": publication,
        "canonical_group_path": publication["group_path"],
        "native_run_manifest_schema_version": (
            CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        ),
        "logical_schema_version": 1,
        "selector_eligible": True,
        "registry_updated": False,
    }
    if result_json is not None:
        write_json_atomic(result_json.expanduser().resolve(), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument(
        "--work-unit-report",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--recording-frame-index", type=Path)
    parser.add_argument(
        "--frame-mapping-mode",
        choices=FRAME_MAPPING_MODE_CHOICES,
        default=FRAME_MAPPING_MODE_INDEXED,
    )
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--n-frames", required=True, type=int)
    parser.add_argument("--source-width", required=True, type=int)
    parser.add_argument("--source-height", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-zarr", required=True, type=Path)
    parser.add_argument("--producer-id", required=True)
    parser.add_argument("--producer-version", required=True)
    parser.add_argument("--source-frame-record-ref", required=True)
    parser.add_argument("--source-frame-record-sha256", required=True)
    parser.add_argument("--source-pixel-record-ref", required=True)
    parser.add_argument("--source-pixel-record-sha256", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = assemble_and_publish_clipped_native_detection(
        analysis_zarr=args.analysis_zarr,
        work_unit_reports=args.work_unit_report,
        recording_frame_index=args.recording_frame_index,
        frame_mapping_mode=args.frame_mapping_mode,
        recording_identity=args.recording_identity,
        n_frames=args.n_frames,
        source_width=args.source_width,
        source_height=args.source_height,
        run_id=args.run_id,
        candidate_zarr=args.candidate_zarr,
        producer_id=args.producer_id,
        producer_version=args.producer_version,
        source_frame_authority={
            "record_ref": args.source_frame_record_ref,
            "record_sha256": args.source_frame_record_sha256,
        },
        source_pixel_authority={
            "record_ref": args.source_pixel_record_ref,
            "record_sha256": args.source_pixel_record_sha256,
        },
        model_artifact_sha256=args.model_artifact_sha256,
        workflow_id=args.workflow_id,
        result_json=args.result_json,
        copy_backend=args.copy_backend,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
