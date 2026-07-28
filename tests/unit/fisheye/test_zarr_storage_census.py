from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from fisheye.diagnostics.zarr_storage_census import (
    DETECTION_OUTPUT,
    DETECTION_SUMMARY_OUTPUT,
    SCHEMA_OUTPUT,
    SUMMARY_OUTPUT,
    WRITER_OUTPUT,
    build_census,
    render_detection_inventory,
    render_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _build() -> dict[str, object]:
    return build_census(REPO_ROOT)


def test_array_spec_census_matches_current_repository_baseline() -> None:
    census = _build()
    schema = census["schema_document"]
    summary = schema["summary"]

    assert summary["array_spec_declaration_count"] == 450
    assert summary["array_spec_unique_leaf_name_count"] == 270
    assert summary["array_spec_unique_signature_count"] == 343
    assert summary["array_spec_ambiguous_leaf_name_count"] == 43

    ambiguous = {row["array_name"]: row for row in schema["array_spec_ambiguities"]}
    assert "frame_indices" in ambiguous
    assert "masks_roi" in ambiguous
    assert all(row["review_status"] == "unresolved" for row in ambiguous.values())


def test_stage_bound_and_unbound_array_specs_remain_distinct() -> None:
    occurrences = _build()["schema_document"]["occurrences"]

    keypoints_img = next(
        row
        for row in occurrences
        if row["source_kind"] == "array_spec_stage_binding"
        and row["path_pattern"] == "keypoints_runs/<run>/keypoints_img"
    )
    assert keypoints_img["dtype"] == "float64"
    assert keypoints_img["shape_template"] == ["n_rois", "n_keypoints", 2]
    assert keypoints_img["contract_mapping_status"] == "exact"
    assert keypoints_img["canonical_contract_id"] == "palette.array.keypoints_img"

    standalone_analysis = [
        row
        for row in occurrences
        if row["source_kind"] == "array_spec_unbound_declaration"
        and row["file"].endswith("analysis_stage_arrays.py")
    ]
    assert standalone_analysis
    assert all(
        str(row["path_pattern"]).startswith("<declaration:")
        for row in standalone_analysis
    )

    legacy_eye = next(
        row
        for row in occurrences
        if row["source_kind"] == "array_spec_stage_binding"
        and row["path_pattern"] == "eye_masks_runs/<run>/masks_roi"
    )
    assert legacy_eye["status"] == "legacy_only"
    assert legacy_eye["contract_mapping_status"] == "candidate"


def test_training_required_array_declarations_are_schema_evidence() -> None:
    schema = _build()["schema_document"]
    declarations = [
        row
        for row in schema["occurrences"]
        if row["source_kind"] == "reader_required_array_declaration"
        and row["file"].endswith("export_subject_mask_training_zarr.py")
    ]
    assert declarations
    paths = {row["path_pattern"] for row in declarations}
    assert "crop_runs/<run>/roi_images" in paths
    assert "subject_mask_runs/<run>/masks_roi" in paths
    assert "source_index/source_dataset_idx" in paths
    assert all(row["required"] is True for row in declarations)


def test_writer_census_covers_direct_wrapped_training_and_manual_writers() -> None:
    writer_document = _build()["writer_document"]
    sites = writer_document["sites"]
    summary = writer_document["summary"]

    assert summary["direct_zarr_api_site_count"] > 0
    assert summary["writer_wrapper_call_site_count"] > 0
    assert summary["manual_zarr_metadata_site_count"] == 1

    assert any(
        row["file"].endswith("export_keypoint_training_zarr.py")
        and row["api_method"] == "require_array"
        and row["surface_class"] == "training"
        for row in sites
    )
    assert any(
        row["file"].endswith("export_acquisition_crop_pose_training_zarr.py")
        and row["call_kind"] == "writer_wrapper_call"
        and row["array_name"] == "keypoints_img"
        and row["dtype"] == "float64"
        for row in sites
    )
    manual = next(row for row in sites if row["call_kind"] == "manual_zarr_metadata")
    assert manual["array_name"] == "encoded_global_masks_roi"
    assert manual["zarr_format_expression"] == "3"
    assert manual["compressors_expression"] == "bytes + zstd(level=0, checksum=False)"


def test_writer_census_covers_derived_caches_and_publication_copy() -> None:
    sites = _build()["writer_document"]["sites"]

    mask_store_names = {
        row["array_name"]
        for row in sites
        if row["file"].endswith("shared/mask_store.py")
    }
    assert {"masks_packed", "counts", "indptr", "present"} <= mask_store_names

    contour_names = {
        row["array_name"]
        for row in sites
        if row["file"].endswith("refined_subject_component_contours.py")
    }
    assert {"ptr", "len", "points_xy"} <= contour_names

    assert any(
        row["writer_symbol"]
        == "fisheye.shared.zarr_sharded_copy:copy_completed_run_to_sharded"
        for row in sites
    )


def test_writer_records_retain_physical_and_classification_fields() -> None:
    sites = _build()["writer_document"]["sites"]
    required_fields = {
        "site_id",
        "file",
        "line",
        "enclosing_symbol",
        "call_kind",
        "path_pattern",
        "declaring_stage",
        "declaring_stage_basis",
        "array_name_expression",
        "dtype_expression",
        "shape_expression",
        "chunks_expression",
        "shards_expression",
        "compressor_expression",
        "compressors_expression",
        "filters_expression",
        "serializer_expression",
        "zarr_format_expression",
        "surface_class",
        "status",
        "access_pattern",
        "write_lifecycle",
        "consumer",
        "contract_mapping_status",
    }
    assert sites
    assert all(required_fields <= set(row) for row in sites)
    assert len({row["site_id"] for row in sites}) == len(sites)


def test_generated_artifacts_are_deterministic() -> None:
    # Use the real repository as input, but direct generated output to a copied
    # root is not supported because source-relative paths are part of the scan.
    # Determinism is therefore checked in memory and serialization is exercised
    # against the checked-in root in the separate --check invariant.
    first = _build()
    second = build_census(REPO_ROOT)
    assert first == second
    assert "What `ArraySpec` Represents Today" in render_summary(first)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_checked_in_census_artifacts_match_generator() -> None:
    census = _build()
    expected = {
        SCHEMA_OUTPUT: json.dumps(census["schema_document"], indent=2, sort_keys=True)
        + "\n",
        WRITER_OUTPUT: json.dumps(census["writer_document"], indent=2, sort_keys=True)
        + "\n",
        DETECTION_OUTPUT: json.dumps(
            census["detection_document"], indent=2, sort_keys=True
        )
        + "\n",
        SUMMARY_OUTPUT: render_summary(census),
        DETECTION_SUMMARY_OUTPUT: render_detection_inventory(
            census["detection_document"]
        ),
    }
    for relative, content in expected.items():
        assert (REPO_ROOT / relative).read_text(encoding="utf-8") == content
