from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from fisheye.diagnostics.zarr_detection_schema_inventory import (
    DETECTION_OUTPUT,
    DETECTION_SUMMARY_OUTPUT,
    render_detection_inventory,
)
from fisheye.diagnostics.zarr_storage_census import build_census

REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _build() -> dict[str, object]:
    return build_census(REPO_ROOT)


def test_detection_inventory_separates_declarations_from_runtime_variants() -> None:
    document = _build()["detection_document"]
    summary = document["summary"]

    assert summary["declared_variant_count"] == 4
    assert summary["declared_array_binding_count"] == 35
    assert summary["current_runtime_variant_count"] == 8
    assert summary["current_runtime_array_binding_count"] == 103

    variants = {row["variant_id"]: row for row in document["current_runtime_variants"]}
    canonical = {
        row["name"]: row for row in variants["current.detect_yolo_canonical"]["arrays"]
    }
    assert canonical["bbox_norm_coords"]["dtype"] == "float64"
    assert canonical["source_acquisition_frame_index"]["dtype"] == "int64"
    assert canonical["instance_key"]["required"] is True

    artifact = {
        row["name"]: row
        for row in variants["current.detection_artifact_unbound"]["arrays"]
    }
    assert "artifact_row_id" in artifact
    assert "instance_key" not in artifact

    artifact_classification = variants["current.detection_artifact_unbound"][
        "classification"
    ]
    assert artifact_classification == {
        "publication_role": "quarantined_evidence",
        "authority": "noncanonical_unbound",
        "mutability": "immutable_after_build",
        "selector_eligible": False,
        "row_identity": "run_local_noncanonical",
        "storage_disposition": "shard_if_retained",
        "implementation_priority": "deferred_compatibility_diagnostic",
        "future_facing": False,
    }


def test_detection_inventory_exposes_known_declaration_conflicts() -> None:
    document = _build()["detection_document"]
    conflicts = {row["conflict_id"]: row for row in document["conflicts"]}

    assert len(conflicts) == document["summary"]["unresolved_conflict_count"] == 13
    bbox = conflicts["current.detect_yolo_canonical:bbox_norm_coords:dtype"]
    assert bbox["declared"] == "float32"
    assert bbox["observed_current"] == "float64"
    assert "current.detect_quality_collection_snapshot:missing_stage_spec" in conflicts
    assert "current.refined_detect_dense_authoring_root:missing_stage_spec" in conflicts
    assert (
        "current.refined_detect_clipped_collection_snapshot:missing_stage_spec"
        in conflicts
    )
    assert (
        "current.refined_detect_instances_projection:instance_key_origin_codes:missing_declaration"
        in conflicts
    )


def test_detection_inventory_records_accepted_float32_geometry_target() -> None:
    document = _build()["detection_document"]

    assert document["schema_version"] == 3
    assert document["accepted_target_decisions"] == [
        {
            "decision_id": "canonical_detection_continuous_geometry_dtype.v1",
            "status": "accepted",
            "scope": [
                "detect_runs/<run>/bbox_norm_coords",
                "detect_runs/<run>/bbox_img_xyxy",
                "detect_runs/<run>/centers_img_xy",
            ],
            "canonical_dtype": "float32",
            "current_runtime_dtype": "float64",
            "current_runtime_disposition": "explicit_legacy_transition",
            "rationale": (
                "Prefer the interoperable precision-safe baseline while canonical "
                "storage schemas and consumers are being completed."
            ),
            "deferred_representations": [
                "float16",
                "uint16_normalized",
                "uint16_fixed_point",
            ],
            "revisit_after": "canonical_storage_specs_complete",
            "change_policy": (
                "A smaller representation requires a new schema version or "
                "representation ID plus quantified error and downstream-behavior "
                "benchmarks."
            ),
        }
    ]


def test_detection_inventory_records_dated_sleepyfish_physical_layouts() -> None:
    document = _build()["detection_document"]
    observations = {
        row["observation_id"]: row for row in document["dated_physical_observations"]
    }
    raw = observations["sleepyfish_cam2010095_latest_raw_detect_20260723"]
    quality = observations["sleepyfish_cam2010095_quality_snapshot_20260723"]
    refined = observations["sleepyfish_cam2010095_refined_snapshot_20260723"]

    assert len(raw["arrays"]) == 6
    assert not any(row["sharded"] for row in raw["arrays"])
    assert len(quality["arrays"]) == 3
    assert all(row["sharded"] for row in quality["arrays"])
    assert len(refined["arrays"]) == 36
    assert all(row["sharded"] for row in refined["arrays"])
    assert len(refined["metadata_tree_sha256"]) == 64
    refined_arrays = {row["path"]: row for row in refined["arrays"]}
    assert refined_arrays["instances/frame_indices"]["dtype"] == "int64"
    assert refined_arrays["instances/frame_indices"]["inner_chunk_shape"] == [16384]
    assert refined_arrays["instances/frame_indices"]["outer_shape"] == [131072]


def test_detection_writer_sites_receive_stage_affiliation() -> None:
    sites = _build()["writer_document"]["sites"]
    assert any(
        row["file"].endswith("detection/detect_yolo.py")
        and row["declaring_stage"] == "detect"
        for row in sites
    )
    assert any(
        row["file"].endswith("refinement/detect_quality_collection.py")
        and row["declaring_stage"] == "detect_quality"
        for row in sites
    )


def test_checked_in_detection_inventory_matches_generator() -> None:
    document = _build()["detection_document"]
    expected_json = json.dumps(document, indent=2, sort_keys=True) + "\n"
    expected_markdown = render_detection_inventory(document)

    assert (REPO_ROOT / DETECTION_OUTPUT).read_text(encoding="utf-8") == expected_json
    assert (REPO_ROOT / DETECTION_SUMMARY_OUTPUT).read_text(
        encoding="utf-8"
    ) == expected_markdown
