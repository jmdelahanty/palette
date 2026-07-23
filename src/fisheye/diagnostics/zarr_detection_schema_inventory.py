"""Detection-family projection of the repository-wide Zarr census.

This module keeps three kinds of evidence separate:

* declarative ``ArraySpec``/``StageSpec`` bindings;
* arrays emitted or required by current writer/validator code; and
* dated read-only observations of representative physical archives.

The result is an inventory and review queue, not an accepted storage contract.
No Zarr store is opened while generating the checked-in artifacts.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping, Sequence

DETECTION_OUTPUT = Path("docs/diagnostics/zarr_detection_schema_inventory.json")
DETECTION_SUMMARY_OUTPUT = Path("docs/diagnostics/zarr_detection_schema_inventory.md")

_OWNED_STAGES = frozenset({"detect", "detect_quality", "refined_detect"})
_OWNED_FAMILY_ROLES = (
    ("refined_detect_runs", "refined_detection"),
    ("detect_quality_runs", "detection_quality"),
    ("detection_artifact_runs", "unbound_detection_artifact"),
    ("/quality_reports/", "detection_quality"),
    ("detect_runs", "canonical_detection"),
)
_DOWNSTREAM_LINEAGE_ARRAYS = frozenset(
    {
        "artifact_row_id",
        "detection_indices",
        "detection_source",
        "instance_key",
        "instance_key_origin_codes",
        "refined_row_ids",
        "resolved_refined_row_id",
        "source_clip_detect_row_index",
        "source_detect_row_index",
        "source_refined_row_ids",
        "source_resolved_refined_row_id",
    }
)
_DETECTION_MODULE_MARKERS = (
    "/detection/",
    "/inference/predict_detections.py",
    "/refinement/detect_quality",
    "/refinement/refine_detect.py",
    "/shared/refined_detect",
    "/shared/detect_reason_codec.py",
    "/shared/detection_producer_lifecycle.py",
    "/tune/detect_",
    "/training/train_detection.py",
    "/training/zarr_yolo_dataset_loader.py",
    "/utils/export_detect_training_zarr.py",
    "/utils/import_acquisition_detections_to_detect_run.py",
    "/utils/materialize_clipped_detect_quality_source.py",
    "/utils/publish_clipped_refined_detect_snapshot.py",
)


def _array(
    name: str,
    dtype: str,
    shape: Sequence[str | int],
    *,
    required: bool = True,
    source: str,
    line: int,
    note: str = "",
) -> dict[str, object]:
    return {
        "name": name,
        "dtype": dtype,
        "shape_template": list(shape),
        "required": required,
        "access_pattern": _access_pattern(name),
        "evidence": {"file": source, "line": line},
        "note": note,
    }


def _access_pattern(name: str) -> str:
    leaf = name.rsplit("/", 1)[-1]
    if leaf in {"frame_counts", "n_detections", "frame_offsets", "quality_flags"}:
        return "windowed_frame_axis"
    if leaf in {
        "bbox_img_xyxy",
        "bbox_norm_coords",
        "centers_img_xy",
        "centers_px",
        "reason_bytes",
    }:
        return "per_detection_row"
    return "windowed_row_axis"


def _current_runtime_variants(
    declarations: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    yolo_source = "src/fisheye/detection/detect_yolo.py"
    refined_source = "src/fisheye/shared/refined_detect_curation.py"
    quality_source = "src/fisheye/refinement/detect_quality_collection.py"

    canonical_arrays = [
        _array(
            "frame_indices", "int32", ("n_detections",), source=yolo_source, line=3183
        ),
        _array(
            "bbox_norm_coords",
            "float64",
            ("n_detections", 4),
            source=yolo_source,
            line=3184,
        ),
        _array("scores", "float32", ("n_detections",), source=yolo_source, line=3185),
        _array("class_ids", "int32", ("n_detections",), source=yolo_source, line=3186),
        _array(
            "instance_key", "uint64", ("n_detections",), source=yolo_source, line=3237
        ),
        _array(
            "source_acquisition_frame_index",
            "int64",
            ("n_detections",),
            source=yolo_source,
            line=3284,
        ),
        _array(
            "bbox_img_xyxy",
            "float64",
            ("n_detections", 4),
            source=yolo_source,
            line=3288,
        ),
        _array(
            "centers_img_xy",
            "float64",
            ("n_detections", 2),
            source=yolo_source,
            line=3288,
        ),
        _array("frame_counts", "int32", ("n_frames",), source=yolo_source, line=3191),
        _array(
            "n_detections",
            "int32",
            ("n_frames",),
            source=yolo_source,
            line=1433,
            note="Current compatibility alias written alongside frame_counts.",
        ),
    ]
    artifact_arrays = [
        _array(
            "frame_indices", "int32", ("n_detections",), source=yolo_source, line=1584
        ),
        _array(
            "bbox_norm_coords",
            "float64",
            ("n_detections", 4),
            source=yolo_source,
            line=1585,
        ),
        _array("scores", "float32", ("n_detections",), source=yolo_source, line=1586),
        _array("class_ids", "int32", ("n_detections",), source=yolo_source, line=1587),
        _array(
            "artifact_row_id",
            "uint64",
            ("n_detections",),
            source=yolo_source,
            line=1588,
        ),
        _array("frame_counts", "int32", ("n_frames",), source=yolo_source, line=1614),
        _array("n_detections", "int32", ("n_frames",), source=yolo_source, line=1615),
    ]
    quality_collection_arrays = [
        _array("quality_flags", "int8", ("n_frames",), source=quality_source, line=936),
        _array(
            "detection_quality_labels",
            "int8",
            ("n_detections",),
            source=quality_source,
            line=943,
        ),
        _array(
            "instance_key", "uint64", ("n_detections",), source=quality_source, line=950
        ),
    ]
    dense_refined_arrays = [
        _array(
            "refined_row_ids", "int64", ("n_curated",), source=refined_source, line=1937
        ),
        _array(
            "frame_indices", "int32", ("n_curated",), source=refined_source, line=1938
        ),
        _array("entity_ids", "int32", ("n_curated",), source=refined_source, line=1939),
        _array(
            "bbox_img_xyxy",
            "float64",
            ("n_curated", 4),
            source=refined_source,
            line=1997,
        ),
        _array(
            "bbox_norm_coords",
            "float64",
            ("n_curated", 4),
            source=refined_source,
            line=1940,
        ),
        _array(
            "status_codes", "int8", ("n_curated",), source=refined_source, line=1941
        ),
        _array(
            "source_kind_codes",
            "int8",
            ("n_curated",),
            source=refined_source,
            line=1942,
        ),
        _array(
            "manual_edit_flags",
            "bool",
            ("n_curated",),
            source=refined_source,
            line=1968,
        ),
        _array(
            "source_detect_row_index",
            "int32",
            ("n_curated",),
            source=refined_source,
            line=1948,
        ),
        _array(
            "review_state_codes",
            "int8",
            ("n_curated",),
            source=refined_source,
            line=1943,
        ),
        _array(
            "keypoints_state_codes",
            "int8",
            ("n_curated",),
            source=refined_source,
            line=1944,
        ),
        _array(
            "subject_mask_state_codes",
            "int8",
            ("n_curated",),
            source=refined_source,
            line=1945,
        ),
        _array(
            "eye_mask_state_codes",
            "int8",
            ("n_curated",),
            source=refined_source,
            line=1946,
        ),
        _array(
            "swim_bladder_state_codes",
            "int8",
            ("n_curated",),
            source=refined_source,
            line=1947,
        ),
        _array(
            "confidence_scores",
            "float32",
            ("n_curated",),
            source=refined_source,
            line=1982,
        ),
        _array("class_ids", "int32", ("n_curated",), source=refined_source, line=1989),
        _array(
            "detection_source",
            "int8",
            ("n_curated",),
            required=False,
            source=refined_source,
            line=1975,
        ),
        _array(
            "reason_bytes",
            "uint8",
            ("n_curated", "reason_width"),
            source="src/fisheye/shared/detect_reason_codec.py",
            line=67,
        ),
        _array(
            "review_notes",
            "utf8",
            ("n_curated",),
            required=False,
            source=refined_source,
            line=2045,
        ),
    ]

    instances = [
        {**dict(item), "access_pattern": _access_pattern(str(item["name"]))}
        for item in declarations["declared.refined_detect.instances"]["arrays"]
    ]
    instances.append(
        _array(
            "instance_key_origin_codes",
            "int8",
            ("n_instances",),
            required=False,
            source=refined_source,
            line=1756,
        )
    )
    source_detections = [
        {**dict(item), "access_pattern": _access_pattern(str(item["name"]))}
        for item in declarations["declared.refined_detect.source_detections"]["arrays"]
    ]

    clipped_observation = next(
        item
        for item in _sleepyfish_observations()
        if item["observation_id"] == "sleepyfish_cam2010095_refined_snapshot_20260723"
    )
    clipped_arrays: list[dict[str, object]] = []
    for observed in clipped_observation["arrays"]:
        path = str(observed["path"])
        shape = list(observed["shape"])
        if path == "instances/frame_counts":
            template: list[str | int] = ["n_frames"]
        elif path == "instances/frame_offsets":
            template = ["n_frame_offsets"]
        elif path.startswith("instances/"):
            template = ["n_instances", *shape[1:]]
        else:
            template = ["n_source_detections", *shape[1:]]
        clipped_arrays.append(
            _array(
                path,
                str(observed["dtype"]),
                template,
                source="src/fisheye/utils/publish_clipped_refined_detect_snapshot.py",
                line=396,
                note=(
                    "Current clipped-collection publication plan; exact dtype is also "
                    "confirmed by the dated Sleepyfish observation."
                ),
            )
        )

    return [
        {
            "variant_id": "current.detect_yolo_canonical",
            "path_pattern": "detect_runs/<run>/",
            "role": "canonical_detection",
            "lifecycle": "build_then_immutable",
            "producer": "fisheye.detection.detect_yolo:_write_detection_output_arrays",
            "declaration_variant_id": "declared.detect",
            "arrays": canonical_arrays,
        },
        {
            "variant_id": "current.detection_artifact_unbound",
            "path_pattern": "detection_artifact_runs/<run>/",
            "role": "unbound_detection_artifact",
            "lifecycle": "build_then_immutable_nonselector",
            "producer": "detect_yolo or detect_traditional",
            "declaration_variant_id": "declared.detect",
            "classification": {
                "publication_role": "quarantined_evidence",
                "authority": "noncanonical_unbound",
                "mutability": "immutable_after_build",
                "selector_eligible": False,
                "row_identity": "run_local_noncanonical",
                "storage_disposition": "shard_if_retained",
                "implementation_priority": "deferred_compatibility_diagnostic",
                "future_facing": False,
            },
            "arrays": artifact_arrays,
        },
        {
            "variant_id": "current.detect_quality_nested",
            "path_pattern": "detect_runs/<run>/quality_reports/<qrun>/",
            "role": "detection_quality",
            "lifecycle": "build_then_immutable",
            "producer": "fisheye.refinement.detect_quality",
            "declaration_variant_id": "declared.detect_quality_nested",
            "arrays": [
                {**dict(item), "access_pattern": _access_pattern(str(item["name"]))}
                for item in declarations["declared.detect_quality_nested"]["arrays"]
            ],
        },
        {
            "variant_id": "current.detect_quality_collection_snapshot",
            "path_pattern": "detect_quality_runs/<run>/",
            "role": "detection_quality",
            "lifecycle": "immutable_sharded_snapshot",
            "producer": "fisheye.refinement.detect_quality_collection",
            "declaration_variant_id": None,
            "arrays": quality_collection_arrays,
        },
        {
            "variant_id": "current.refined_detect_dense_authoring_root",
            "path_pattern": "refined_detect_runs/<run>/",
            "role": "refined_detection_authoring",
            "lifecycle": "editable_random_update_and_projection_sync",
            "producer": "fisheye.shared.refined_detect_curation:_write_dense_curated_root_arrays",
            "declaration_variant_id": None,
            "arrays": dense_refined_arrays,
        },
        {
            "variant_id": "current.refined_detect_instances_projection",
            "path_pattern": "refined_detect_runs/<run>/instances/",
            "role": "refined_detection_projection",
            "lifecycle": "derived_projection_or_immutable_snapshot",
            "producer": "fisheye.shared.refined_detect_curation:_write_instances_arrays",
            "declaration_variant_id": "declared.refined_detect.instances",
            "arrays": sorted(instances, key=lambda item: str(item["name"])),
        },
        {
            "variant_id": "current.refined_detect_source_projection",
            "path_pattern": "refined_detect_runs/<run>/source_detections/",
            "role": "refined_detection_projection",
            "lifecycle": "derived_projection_or_immutable_snapshot",
            "producer": "fisheye.shared.refined_detect_curation:_write_source_detections_arrays",
            "declaration_variant_id": "declared.refined_detect.source_detections",
            "arrays": sorted(source_detections, key=lambda item: str(item["name"])),
        },
        {
            "variant_id": "current.refined_detect_clipped_collection_snapshot",
            "path_pattern": "refined_detect_runs/<run>/{instances,source_detections}/",
            "role": "refined_detection_publication",
            "lifecycle": "immutable_sharded_snapshot",
            "producer": "fisheye.utils.publish_clipped_refined_detect_snapshot",
            "declaration_variant_id": None,
            "arrays": sorted(clipped_arrays, key=lambda item: str(item["name"])),
        },
    ]


def _declared_variants(
    schema_document: Mapping[str, object],
) -> list[dict[str, object]]:
    occurrences = schema_document["occurrences"]
    assert isinstance(occurrences, Sequence)
    keys = (
        ("declared.detect", "detect", None),
        ("declared.detect_quality_nested", "detect_quality", None),
        ("declared.refined_detect.instances", "refined_detect", "instances"),
        (
            "declared.refined_detect.source_detections",
            "refined_detect",
            "source_detections",
        ),
    )
    variants: list[dict[str, object]] = []
    for variant_id, stage, subgroup in keys:
        rows = [
            row
            for row in occurrences
            if isinstance(row, Mapping)
            and row.get("source_kind") == "array_spec_stage_binding"
            and row.get("declaring_stage") == stage
            and row.get("subgroup") == subgroup
        ]
        if not rows:
            raise RuntimeError(f"Detection declaration variant {variant_id} is empty.")
        path = str(rows[0]["path_pattern"]).rsplit("/", 1)[0] + "/"
        arrays = [
            {
                "name": row["array_name"],
                "dtype": row["dtype"],
                "shape_template": row["shape_template"],
                "required": row["required"],
                "access_pattern": row["access_pattern"],
                "evidence": {"file": row["file"], "line": row["line"]},
                "note": row["description"],
            }
            for row in sorted(rows, key=lambda item: str(item["array_name"]))
        ]
        variants.append(
            {
                "variant_id": variant_id,
                "path_pattern": path,
                "stage": stage,
                "subgroup": subgroup,
                "stage_spec_symbol": rows[0]["stage_spec_symbol"],
                "arrays": arrays,
            }
        )
    return variants


def _affiliation(record: Mapping[str, object]) -> tuple[str, str] | None:
    path = str(record.get("path_pattern") or "").lower()
    stage = str(record.get("declaring_stage") or "")
    file = "/" + str(record.get("file") or "").lower().lstrip("/")
    name = record.get("array_name")

    for token, role in _OWNED_FAMILY_ROLES:
        if token in path:
            return role, f"path contains {token}"
    if stage in _OWNED_STAGES:
        return stage, f"declaring stage is {stage}"
    if any(marker in file for marker in _DETECTION_MODULE_MARKERS):
        if "training" in file:
            return "detection_training", "detection-specific training module"
        return "detection_implementation", "detection-specific implementation module"
    if isinstance(name, str) and name in _DOWNSTREAM_LINEAGE_ARRAYS:
        return "downstream_detection_lineage", "shared detection identity/lineage leaf"
    return None


def _project_evidence(
    records: Sequence[object],
    *,
    writer: bool,
) -> list[dict[str, object]]:
    projected: list[dict[str, object]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        affiliation = _affiliation(raw)
        if affiliation is None:
            continue
        role, basis = affiliation
        row = {
            "evidence_id": raw.get("site_id") if writer else raw.get("occurrence_id"),
            "evidence_kind": "writer_site" if writer else raw.get("source_kind"),
            "affiliation_role": role,
            "affiliation_basis": basis,
            "file": raw.get("file"),
            "line": raw.get("line"),
            "path_pattern": raw.get("path_pattern"),
            "declaring_stage": raw.get("declaring_stage"),
            "array_name": raw.get("array_name"),
            "array_name_expression": raw.get("array_name_expression"),
            "dtype": raw.get("dtype"),
            "dtype_expression": raw.get("dtype_expression"),
            "shape_template": raw.get("shape_template"),
            "shape_expression": raw.get("shape_expression"),
            "status": raw.get("status"),
        }
        if writer:
            row.update(
                {
                    "call_kind": raw.get("call_kind"),
                    "writer_symbol": raw.get("writer_symbol"),
                    "chunks_expression": raw.get("chunks_expression"),
                    "shards_expression": raw.get("shards_expression"),
                    "compressor_expression": raw.get("compressor_expression"),
                    "compressors_expression": raw.get("compressors_expression"),
                    "serializer_expression": raw.get("serializer_expression"),
                    "zarr_format_expression": raw.get("zarr_format_expression"),
                }
            )
        projected.append(row)
    return sorted(
        projected,
        key=lambda item: (
            str(item["affiliation_role"]),
            str(item["file"]),
            int(item["line"] or 0),
            str(item["array_name"]),
        ),
    )


def _compare_variants(
    declarations: Mapping[str, Mapping[str, object]],
    runtime_variants: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    for runtime in runtime_variants:
        runtime_id = str(runtime["variant_id"])
        declaration_id = runtime.get("declaration_variant_id")
        if not isinstance(declaration_id, str):
            conflicts.append(
                {
                    "conflict_id": f"{runtime_id}:missing_stage_spec",
                    "severity": "high",
                    "variant_id": runtime_id,
                    "array_name": None,
                    "field": "variant",
                    "declared": None,
                    "observed_current": runtime["path_pattern"],
                    "kind": "runtime_variant_missing_stage_spec",
                    "resolution_status": "unresolved",
                }
            )
            continue
        declared_arrays = {
            str(item["name"]): item
            for item in declarations[declaration_id]["arrays"]
            if isinstance(item, Mapping)
        }
        for current in runtime["arrays"]:
            assert isinstance(current, Mapping)
            name = str(current["name"])
            declared = declared_arrays.get(name)
            if declared is None:
                conflicts.append(
                    {
                        "conflict_id": f"{runtime_id}:{name}:missing_declaration",
                        "severity": "high",
                        "variant_id": runtime_id,
                        "array_name": name,
                        "field": "array_presence",
                        "declared": None,
                        "observed_current": True,
                        "kind": "writer_array_missing_declaration",
                        "resolution_status": "unresolved",
                    }
                )
                continue
            for field in ("dtype", "shape_template", "required"):
                if declared.get(field) != current.get(field):
                    conflicts.append(
                        {
                            "conflict_id": f"{runtime_id}:{name}:{field}",
                            "severity": (
                                "high"
                                if field in {"dtype", "shape_template"}
                                else "medium"
                            ),
                            "variant_id": runtime_id,
                            "array_name": name,
                            "field": field,
                            "declared": declared.get(field),
                            "observed_current": current.get(field),
                            "kind": "declaration_writer_mismatch",
                            "resolution_status": "unresolved",
                        }
                    )
    return sorted(conflicts, key=lambda item: str(item["conflict_id"]))


def _downstream_lineage_summary(
    schema_evidence: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in schema_evidence:
        if row["affiliation_role"] != "downstream_detection_lineage":
            continue
        grouped[str(row["array_name"])].append(row)
    output: list[dict[str, object]] = []
    for name, rows in sorted(grouped.items()):
        output.append(
            {
                "array_name": name,
                "occurrence_count": len(rows),
                "declaring_stages": sorted(
                    {
                        str(row["declaring_stage"])
                        for row in rows
                        if row.get("declaring_stage")
                    }
                ),
                "dtypes": sorted(
                    {str(row["dtype"]) for row in rows if row.get("dtype")}
                ),
                "path_patterns": sorted({str(row["path_pattern"]) for row in rows}),
            }
        )
    return output


def _observed_array(
    path: str,
    dtype: str,
    shape: Sequence[int],
    outer_shape: Sequence[int],
    inner_shape: Sequence[int] | None,
) -> dict[str, object]:
    return {
        "path": path,
        "dtype": dtype,
        "shape": list(shape),
        "outer_shape": list(outer_shape),
        "inner_chunk_shape": list(inner_shape) if inner_shape is not None else None,
        "sharded": inner_shape is not None,
        "zarr_format": 3,
        "codec_chain": (
            ["sharding_indexed", "bytes", "zstd(level=0)"]
            if inner_shape is not None
            else ["bytes", "zstd(level=0)"]
        ),
    }


def _sleepyfish_observations() -> list[dict[str, object]]:
    archive = (
        "/groups/johnson/johnsonlab/jeremy/recordings/"
        "sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/"
        "sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr"
    )
    raw = [
        _observed_array("bbox_norm_coords", "float64", (1187087, 4), (1024, 4), None),
        _observed_array("class_ids", "int32", (1187087,), (1024,), None),
        _observed_array("frame_counts", "int32", (1188000,), (1024,), None),
        _observed_array("frame_indices", "int32", (1187087,), (1024,), None),
        _observed_array("n_detections", "int32", (1188000,), (1024,), None),
        _observed_array("scores", "float32", (1187087,), (1024,), None),
    ]
    quality = [
        _observed_array(
            "detection_quality_labels", "int8", (1186376,), (131072,), (16384,)
        ),
        _observed_array("instance_key", "uint64", (1186376,), (131072,), (16384,)),
        _observed_array("quality_flags", "int8", (1188000,), (131072,), (16384,)),
    ]
    refined_rows = (
        ("instances/bbox_img_xyxy", "float64", (1169010, 4), (131072, 4), (1024, 4)),
        ("instances/bbox_norm_coords", "float64", (1169010, 4), (131072, 4), (1024, 4)),
        ("instances/class_ids", "int32", (1169010,), (131072,), (1024,)),
        ("instances/confidence_scores", "float32", (1169010,), (131072,), (1024,)),
        ("instances/frame_counts", "int32", (1188000,), (131072,), (16384,)),
        ("instances/frame_indices", "int64", (1169010,), (131072,), (16384,)),
        ("instances/frame_offsets", "int64", (1188001,), (131072,), (16384,)),
        ("instances/instance_key", "uint64", (1169010,), (131072,), (16384,)),
        (
            "instances/instance_key_origin_codes",
            "int8",
            (1169010,),
            (131072,),
            (16384,),
        ),
        ("instances/manual_edit_flags", "bool", (1169010,), (131072,), (1024,)),
        ("instances/reason_bytes", "uint8", (1169010, 64), (131072, 64), (1024, 64)),
        ("instances/refined_row_ids", "int64", (1169010,), (131072,), (16384,)),
        (
            "instances/source_clip_detect_row_index",
            "int64",
            (1169010,),
            (131072,),
            (16384,),
        ),
        ("instances/source_clip_indices", "int64", (1169010,), (131072,), (16384,)),
        (
            "instances/source_clip_local_frame_indices",
            "int64",
            (1169010,),
            (131072,),
            (16384,),
        ),
        ("instances/source_detect_row_index", "int32", (1169010,), (131072,), (16384,)),
        ("instances/source_frame_indices", "int64", (1169010,), (131072,), (16384,)),
        ("instances/source_kind_codes", "int8", (1169010,), (131072,), (1024,)),
        (
            "instances/source_recording_frame_ids",
            "int64",
            (1169010,),
            (131072,),
            (16384,),
        ),
        ("instances/source_refined_row_ids", "int64", (1169010,), (131072,), (16384,)),
        (
            "source_detections/bbox_img_xyxy",
            "float64",
            (1186376, 4),
            (131072, 4),
            (1024, 4),
        ),
        (
            "source_detections/bbox_norm_coords",
            "float64",
            (1186376, 4),
            (131072, 4),
            (1024, 4),
        ),
        ("source_detections/class_ids", "int32", (1186376,), (131072,), (1024,)),
        (
            "source_detections/confidence_scores",
            "float32",
            (1186376,),
            (131072,),
            (1024,),
        ),
        ("source_detections/decision_codes", "int8", (1186376,), (131072,), (1024,)),
        ("source_detections/frame_indices", "int64", (1186376,), (131072,), (16384,)),
        ("source_detections/instance_key", "uint64", (1186376,), (131072,), (16384,)),
        (
            "source_detections/reason_bytes",
            "uint8",
            (1186376, 64),
            (131072, 64),
            (1024, 64),
        ),
        (
            "source_detections/resolved_refined_row_id",
            "int64",
            (1186376,),
            (131072,),
            (16384,),
        ),
        (
            "source_detections/source_clip_detect_row_index",
            "int64",
            (1186376,),
            (131072,),
            (16384,),
        ),
        (
            "source_detections/source_clip_indices",
            "int64",
            (1186376,),
            (131072,),
            (16384,),
        ),
        (
            "source_detections/source_clip_local_frame_indices",
            "int64",
            (1186376,),
            (131072,),
            (16384,),
        ),
        (
            "source_detections/source_detect_row_index",
            "int32",
            (1186376,),
            (131072,),
            (16384,),
        ),
        (
            "source_detections/source_frame_indices",
            "int64",
            (1186376,),
            (131072,),
            (16384,),
        ),
        (
            "source_detections/source_recording_frame_ids",
            "int64",
            (1186376,),
            (131072,),
            (16384,),
        ),
        (
            "source_detections/source_resolved_refined_row_id",
            "int64",
            (1186376,),
            (131072,),
            (16384,),
        ),
    )
    refined = [_observed_array(*row) for row in refined_rows]
    return [
        {
            "observation_id": "sleepyfish_cam2010095_latest_raw_detect_20260723",
            "observed_at": "2026-07-23",
            "source_archive": archive,
            "run_path": "detect_runs/detect_2026-05-14_15-39-11",
            "selector_state": "detect_runs/latest",
            "completion_status": "completion metadata absent",
            "status": "historical_current_archive_observation",
            "root_metadata_sha256": "65e6f8a5d691aad0148e1f08d3c7c0e052fdf789167b49e47f2c1168cf3b7c3b",
            "metadata_tree_sha256": "b6e00d08c0ff7eca41b30e7dbc7c73964a5a252bb1b5a8844fcbadb7872b048d",
            "arrays": raw,
        },
        {
            "observation_id": "sleepyfish_cam2010095_quality_snapshot_20260723",
            "observed_at": "2026-07-23",
            "source_archive": archive,
            "run_path": "detect_quality_runs/detect_quality_sleepyfish_source_collection_v2_20260715_01",
            "selector_state": "latest and latest_complete",
            "completion_status": "complete",
            "status": "current_immutable_snapshot_observation",
            "root_metadata_sha256": "1b1adb3b24c58fc1e227430a1edd4b4f7c61c2a92a17178493dfb2ffc500a9c7",
            "metadata_tree_sha256": "5d6c9541183136842dd3d88cd1861a7e562185c9363e919a745b2efb4229fe71",
            "arrays": quality,
        },
        {
            "observation_id": "sleepyfish_cam2010095_refined_snapshot_20260723",
            "observed_at": "2026-07-23",
            "source_archive": archive,
            "run_path": "refined_detect_runs/refined_detect_sleepyfish_allclips_sharded_20260715_01",
            "selector_state": "latest and latest_complete",
            "completion_status": "complete",
            "status": "current_immutable_snapshot_observation",
            "root_metadata_sha256": "180e9f54b4b3369721f6e2a483d5d1bd1fc9357c760ac4da63340793d7c94b05",
            "metadata_tree_sha256": "ce71eb8872d3e07dff9f055f26d0a2cfd88ada5afdb1cb64c118aa16df0e3a13",
            "arrays": refined,
        },
    ]


def build_detection_document(
    schema_document: Mapping[str, object],
    writer_document: Mapping[str, object],
) -> dict[str, object]:
    """Build the deterministic detection-family projection."""

    declared = _declared_variants(schema_document)
    declared_by_id = {str(item["variant_id"]): item for item in declared}
    runtime = _current_runtime_variants(declared_by_id)
    conflicts = _compare_variants(declared_by_id, runtime)
    schema_evidence = _project_evidence(
        schema_document["occurrences"],  # type: ignore[arg-type]
        writer=False,
    )
    writer_evidence = _project_evidence(
        writer_document["sites"],  # type: ignore[arg-type]
        writer=True,
    )
    downstream = _downstream_lineage_summary(schema_evidence)
    observations = _sleepyfish_observations()
    evidence_roles: dict[str, int] = defaultdict(int)
    for row in schema_evidence:
        evidence_roles[str(row["affiliation_role"])] += 1
    accepted_target_decisions = [
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
    return {
        "schema_id": "palette.zarr_detection_schema_inventory",
        "schema_version": 3,
        "status": "review_inventory_not_accepted_contract",
        "generation_policy": (
            "deterministic projection of the static repository census plus explicitly "
            "dated read-only physical observations"
        ),
        "scope": {
            "owned_families": [token for token, _ in _OWNED_FAMILY_ROLES],
            "owned_stages": sorted(_OWNED_STAGES),
            "downstream_lineage_arrays": sorted(_DOWNSTREAM_LINEAGE_ARRAYS),
        },
        "summary": {
            "declared_variant_count": len(declared),
            "declared_array_binding_count": sum(
                len(item["arrays"]) for item in declared
            ),
            "current_runtime_variant_count": len(runtime),
            "current_runtime_array_binding_count": sum(
                len(item["arrays"]) for item in runtime
            ),
            "unresolved_conflict_count": len(conflicts),
            "schema_evidence_count": len(schema_evidence),
            "writer_evidence_count": len(writer_evidence),
            "schema_evidence_by_role": dict(sorted(evidence_roles.items())),
            "downstream_lineage_leaf_count": len(downstream),
            "physical_observation_count": len(observations),
            "physical_observed_array_binding_count": sum(
                len(item["arrays"]) for item in observations
            ),
        },
        "declared_variants": declared,
        "current_runtime_variants": runtime,
        "accepted_target_decisions": accepted_target_decisions,
        "conflicts": conflicts,
        "downstream_lineage": downstream,
        "dated_physical_observations": observations,
        "schema_evidence": schema_evidence,
        "writer_evidence": writer_evidence,
    }


def _markdown_table(
    rows: Sequence[Sequence[object]], headers: Sequence[str]
) -> list[str]:
    def escape(value: object) -> str:
        if isinstance(value, (list, tuple)):
            value = json.dumps(value, separators=(",", ":"))
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(escape(value) for value in row) + " |" for row in rows
    )
    return output


def render_detection_inventory(document: Mapping[str, object]) -> str:
    """Render a concise review document while JSON retains every evidence row."""

    summary = document["summary"]
    assert isinstance(summary, Mapping)
    variants = document["current_runtime_variants"]
    conflicts = document["conflicts"]
    observations = document["dated_physical_observations"]
    downstream = document["downstream_lineage"]
    target_decisions = document["accepted_target_decisions"]
    assert isinstance(variants, Sequence)
    assert isinstance(conflicts, Sequence)
    assert isinstance(observations, Sequence)
    assert isinstance(downstream, Sequence)
    assert isinstance(target_decisions, Sequence)
    lines = [
        "# Palette Detection-Family Zarr Schema Inventory",
        "",
        "Status: generated review inventory; not an accepted storage contract",
        "",
        "This report keeps declarations, current runtime evidence, and dated physical "
        "observations separate. A disagreement is a review item, not something the "
        "generator resolves by guessing.",
        "",
        "## Result",
        "",
        f"- `{summary['declared_variant_count']}` declared variants with "
        f"`{summary['declared_array_binding_count']}` array bindings;",
        f"- `{summary['current_runtime_variant_count']}` current runtime variants with "
        f"`{summary['current_runtime_array_binding_count']}` array bindings;",
        f"- `{summary['unresolved_conflict_count']}` unresolved declaration/runtime conflicts;",
        f"- `{summary['downstream_lineage_leaf_count']}` detection-lineage leaf names propagated outside detection-owned groups;",
        f"- `{summary['physical_observation_count']}` dated Sleepyfish observations with "
        f"`{summary['physical_observed_array_binding_count']}` physical array bindings.",
        "",
        "The current `DETECT_SPEC` is therefore orientation evidence, not an "
        "authoritative schema. In particular, raw bounding boxes are `float64` in "
        "the current writer and the observed archive, while the declaration says "
        "`float32`.",
        "The completed Sleepyfish refined-detection snapshot contains `36` sharded "
        "arrays, including publication lineage that is absent from the `25` subgroup "
        "bindings in the current StageSpec.",
        "`detection_artifact_runs` is classified as immutable, selector-ineligible "
        "quarantined evidence with run-local identity. It remains in the inventory "
        "for compatibility and diagnostics, but it is deferred from the future-facing "
        "storage implementation and benchmark waves.",
        "",
        "## Accepted Future-Facing Decisions",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                (
                    item["decision_id"],
                    item["canonical_dtype"],
                    item["current_runtime_dtype"],
                    item["current_runtime_disposition"],
                    item["revisit_after"],
                )
                for item in target_decisions
                if isinstance(item, Mapping)
            ],
            (
                "Decision",
                "Canonical target",
                "Current runtime",
                "Current disposition",
                "Revisit",
            ),
        )
    )
    lines.extend(
        [
            "",
            "Canonical detection bounding boxes and centers use exact `float32` in "
            "the first storage contract. Current `float64` writers and archives "
            "remain explicit transition evidence until migrated; they do not change "
            "the accepted target. `float16` and quantized `uint16` representations "
            "are deferred and require a new version plus numerical and behavioral "
            "validation.",
            "",
        "## Current Runtime Variants",
        "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                (
                    item["variant_id"],
                    item["path_pattern"],
                    len(item["arrays"]),
                    item["lifecycle"],
                    item["declaration_variant_id"],
                )
                for item in variants
                if isinstance(item, Mapping)
            ],
            ("Variant", "Path", "Arrays", "Lifecycle", "Compared declaration"),
        )
    )
    lines.extend(["", "## Unresolved Conflicts", ""])
    lines.extend(
        _markdown_table(
            [
                (
                    item["severity"],
                    item["variant_id"],
                    item["array_name"] or "—",
                    item["field"],
                    item["declared"],
                    item["observed_current"],
                )
                for item in conflicts
                if isinstance(item, Mapping)
            ],
            ("Severity", "Variant", "Array", "Field", "Declared", "Current"),
        )
    )
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        lines.extend(["", f"## `{variant['variant_id']}`", ""])
        lines.extend(
            _markdown_table(
                [
                    (
                        item["name"],
                        item["dtype"],
                        item["shape_template"],
                        item["required"],
                        item["access_pattern"],
                    )
                    for item in variant["arrays"]
                    if isinstance(item, Mapping)
                ],
                ("Array", "dtype", "Shape", "Required", "Expected access"),
            )
        )
    lines.extend(["", "## Dated Physical Observations", ""])
    lines.extend(
        _markdown_table(
            [
                (
                    item["observation_id"],
                    item["run_path"],
                    item["completion_status"],
                    len(item["arrays"]),
                    sum(bool(array["sharded"]) for array in item["arrays"]),
                )
                for item in observations
                if isinstance(item, Mapping)
            ],
            ("Observation", "Run", "Completion", "Arrays", "Sharded arrays"),
        )
    )
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        lines.extend(["", f"### `{observation['observation_id']}`", ""])
        lines.extend(
            _markdown_table(
                [
                    (
                        item["path"],
                        item["dtype"],
                        item["shape"],
                        item["inner_chunk_shape"] or item["outer_shape"],
                        item["outer_shape"] if item["sharded"] else "—",
                    )
                    for item in observation["arrays"]
                    if isinstance(item, Mapping)
                ],
                ("Array", "dtype", "Shape", "Inner chunk", "Outer shard"),
            )
        )
    lines.extend(["", "## Downstream Detection Lineage", ""])
    lines.extend(
        _markdown_table(
            [
                (
                    item["array_name"],
                    item["occurrence_count"],
                    item["dtypes"],
                    item["declaring_stages"],
                )
                for item in downstream
                if isinstance(item, Mapping)
            ],
            ("Leaf", "Occurrences", "Observed dtypes", "Declared stages"),
        )
    )
    lines.extend(
        [
            "",
            "## Contract Checklist",
            "",
            "Execution order and exit gates are maintained in `docs/canonical_detection_storage_implementation_checklist.md`.",
            "",
            "- [x] Use exact `float32` for first-generation canonical detection bounding boxes and centers; treat current `float64` as an explicit transition representation.",
            "- [x] Defer `float16` and quantized integer detection geometry until canonical storage specs are complete; require a new schema version and behavioral benchmarks before adoption.",
            "- [x] Classify `detection_artifact_runs` separately from canonical `detect_runs` as immutable, selector-ineligible quarantined evidence with run-local identity.",
            "- [ ] Add a StageSpec for immutable `detect_quality_runs` snapshots; do not conflate it with nested historical reports.",
            "- [ ] Decide whether dense refined root arrays remain the editable authority or become a compatibility projection of `instances`.",
            "- [ ] Add every clipped/publication lineage column to a versioned snapshot schema.",
            "- [ ] Lock exact dtype, axis names, null/fill semantics, and requiredness before assigning storage plans.",
            "- [ ] Benchmark canonical detections, quality snapshots, editable refined detections, and published refined snapshots in the first implementation wave.",
            "- [ ] Revisit unbound-artifact storage and benchmarks only if a supported future consumer or canonical binding path is approved.",
            "",
            "The machine-readable JSON retains all affiliated schema and writer "
            "evidence, including dynamic writer sites and physical chunk/shard shapes.",
            "",
        ]
    )
    return "\n".join(lines)
