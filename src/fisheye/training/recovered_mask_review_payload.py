"""Build versioned recovery payloads through existing mask/editor contracts."""

from __future__ import annotations

import numpy as np

from fisheye.shared.detect_reason_codec import decode_reason_bytes, write_reason_columns
from fisheye.shared.refined_subject_mask_mutation import (
    stamp_refined_subject_mask_editable_draft,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.recovered_training_review_contract import (
    COORDINATE_SYSTEM,
    REVIEW_SCHEMA,
    NATIVE_REVIEW_SCHEMA,
    NATIVE_COORDINATE_SYSTEM,
    initial_contract_digest,
)
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.training.mask_tail_keypoints import (
    ORIGIN_CODES,
    SCHEMA_NAME,
    derive_tail_seed,
    recipe_for_schema,
)
from fisheye.training.recover_merged_training_recording import _sha256_array
from fisheye.tune.refined_subject_mask_review import prepare_refined_subject_run
from fisheye.tune.keypoint_failure_review import (
    _DEFAULT_CONFIDENCE_THRESHOLD,
    _resolve_review_geometry_defaults,
)
from fisheye.utils.extend_keypoint_skeleton import _schema_to_attr_payload


def run_paths(version, *, native=False):
    if (
        not version
        or version.startswith(".")
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in version
        )
    ):
        raise ValueError("Version must be a safe non-hidden path component")
    if native:
        return {
            "crop": f"crop_runs/mask_tail_full_roi_{version}",
            "mask": f"subject_mask_runs/mask_tail_snapshot_{version}",
            "mask_edit": f"refined_subject_masks_runs/mask_tail_edit_{version}",
            "seed": f"keypoints_runs/head_tail11_fins_seed_{version}",
            "pose_edit": f"refined_keypoints_runs/head_tail11_fins_edit_{version}",
        }
    return {
        "crop": f"crop_runs/recovered_full_roi_{version}",
        "mask": f"subject_mask_runs/recovered_masks_{version}",
        "mask_edit": f"refined_subject_masks_runs/recovered_masks_edit_{version}",
        "seed": f"keypoints_runs/head_tail11_fins_seed_{version}",
        "pose_edit": f"refined_keypoints_runs/head_tail11_fins_edit_{version}",
    }


def array_hashes(group):
    hashes = {
        name: _sha256_array(np.asarray(array[:])) for name, array in group.arrays()
    }
    for name, child in group.groups():
        hashes.update(
            {f"{name}/{key}": value for key, value in array_hashes(child).items()}
        )
    return hashes


def _array(group, name, data):
    values = np.asarray(data)
    # One row per physical image/mask chunk: edits and shuffled training don't
    # decode unrelated rows. A single writer owns each target run here.
    chunks = (
        (1, *values.shape[1:])
        if values.ndim >= 3
        else (min(256, len(values)), *values.shape[1:])
    )
    group.create_array(name, data=values, chunks=chunks)


def build_review_payload(
    root, arrays, labels, binding, *, version, pose_schema=SCHEMA_NAME, native=False
):
    recipe = recipe_for_schema(pose_schema)
    if native:
        recipe = {
            **recipe,
            "existing_keypoint_policy": "preserve_head_snout_fins_by_name_v1",
            "tail_policy": "derive_all_11_stations_from_mask",
        }
    _, schema = _schema_to_attr_payload(pose_schema)
    point_count = len(schema["keypoint_labels"])
    paths = run_paths(version, native=native)
    n = len(arrays["roi_images"])
    lineage = {
        "frame_indices": arrays["frame_indices"],
        "source_crop_row_ids": np.arange(n, dtype=np.int64),
    }
    for name in (
        "source_merged_row",
        "source_pose_local_row",
        "source_training_crop_row_ids",
    ):
        if name in arrays:
            lineage[name] = arrays[name]
    # Preserve historical ROI/detection identifiers as lineage, without
    # relabeling the sampled row key as an acquisition frame index.
    for name in ("source_roi_idx", "source_refined_row_ids", "source_detect_row_index"):
        if name in arrays:
            lineage[name] = arrays[name]
    common = {
        "schema_id": NATIVE_REVIEW_SCHEMA if native else REVIEW_SCHEMA,
        "schema_version": 1,
        "stage_selector_eligible": False,
        "coordinate_system": NATIVE_COORDINATE_SYSTEM if native else COORDINATE_SYSTEM,
        "frame_index_domain": (
            "source_crop_frame_index" if native else "legacy_training_sample_row"
        ),
        "sensor_pixel_origin_available": False,
        "row_count": n,
        "source_bindings": binding,
        "source_crop_run": paths["crop"].split("/")[1],
    }
    provenance = build_writer_run_provenance(
        command=(
            "fisheye.training.native_mask_tail_review"
            if native
            else "fisheye.training.recover_merged_subject_masks"
        ),
        params={"recipe": recipe, "version": version, "pose_schema": pose_schema},
        input_run_ids=(
            {
                "native_masks": binding["mask_run"],
                "native_keypoints": binding["keypoint_run"],
            }
            if native
            else {"merged_masks": binding["mask_run"]}
        ),
    )

    def create(path, values, extra=None):
        family, name = path.split("/")
        parent = require_runs_parent(root, family)
        run = parent.create_group(name)
        mark_run_started(run, run_name=name, stage=family.removesuffix("_runs"))
        run.attrs.update({**common, **(extra or {})})
        for key, value in values.items():
            _array(run, key, value)
        return run

    crop = create(
        paths["crop"],
        {
            **lineage,
            "roi_images": arrays["roi_images"],
            "source_bbox_norm_coords": arrays["source_bbox_norm_coords"],
        },
        {
            "crop_storage_mode": "materialized",
            "roi_size": list(arrays["roi_images"].shape[1:]),
            "pixel_operation": "identity_copy",
        },
    )
    if "source_roi_coordinates_full" in arrays:
        _array(
            crop, "source_roi_coordinates_full", arrays["source_roi_coordinates_full"]
        )
    mask = create(
        paths["mask"],
        {
            **lineage,
            "masks_roi": arrays["masks_roi"],
            "available_channels": np.ones(len(labels), dtype=bool),
            "target_valid_channels": arrays["target_valid_channels"],
            "detection_source": arrays["detection_source"],
        },
        {
            "mask_labels": list(labels),
            "method": (
                "refined_dense_mask_snapshot_v1"
                if native or "refined_mask_snapshot" in binding
                else "recovered_merged_mask_pixels_v1"
            ),
            "output_semantics": "multilabel",
            "label_schema_id": binding["mask_run_attrs"]["label_schema_id"],
        },
    )
    for name in ("label_origin_codes", "supervision_mode_codes"):
        if name in arrays:
            _array(mask, name, arrays[name])

    derived = derive_tail_seed(
        arrays["masks_roi"],
        labels,
        arrays["head_keypoints_roi"],
        schema_name=pose_schema,
    )
    origins = ORIGIN_CODES
    if native:
        # Native source labels are mapped by name. Tail stations, including the
        # tip, are one new mask-derived arc-length sequence. Historical points
        # remain available in source_keypoints_roi and its bound label list.
        origins = {**ORIGIN_CODES, "existing_keypoint": 4}
        source_labels = binding["source_keypoint_labels"]
        source_points = arrays["source_keypoints_roi"]
        preserve = set(schema["keypoint_labels"][:3] + schema["keypoint_labels"][14:])
        for target_index, label in enumerate(schema["keypoint_labels"]):
            if label in preserve and label in source_labels:
                values = source_points[:, source_labels.index(label)]
                finite = np.isfinite(values).all(axis=1)
                derived["keypoints_roi"][finite, target_index] = values[finite]
                derived["keypoint_origin"][finite, target_index] = 4

    attrs = {
        "pose_schema": schema,
        "skeleton_id": schema["skeleton_id"],
        "keypoint_labels": schema["keypoint_labels"],
        "kpt_shape": [point_count, 2],
        "keypoint_origin_codes": origins,
        "derivation_recipe": recipe,
        "source_subject_mask_run": paths["mask"].split("/")[1],
        "source_mask_sha256": _sha256_array(arrays["masks_roi"]),
        "source_seed_run": paths["seed"].split("/")[1],
        "training_row_policy": "all_points_finite_inside_crop_and_usable",
        "mask_edit_policy": "new_derivation_version_required_after_mask_corrections",
    }
    min_angle, min_area, max_area = _resolve_review_geometry_defaults(attrs)
    attrs["recovered_review_qc"] = {
        "min_triangle_angle": min_angle,
        "min_triangle_area": min_area,
        "max_triangle_area": max_area,
        "confidence_threshold": _DEFAULT_CONFIDENCE_THRESHOLD,
    }
    reasons = np.array(
        [
            (
                "needs_manual_fins"
                if ok
                else f"tail_derivation_failed:{reason}|needs_manual_fins"
            )
            for ok, reason in zip(
                derived["tail_valid"], derived["tail_failure_reason"], strict=True
            )
        ],
        dtype=object,
    )
    snout_reasons = None
    if "snout_valid" in derived:
        snout_reasons = decode_reason_bytes(derived["snout_failure_reason_bytes"])
        for i in np.flatnonzero(~derived["snout_valid"]):
            reasons[i] += f"|snout_derivation_failed:{snout_reasons[i]}"
    if native:
        reasons = np.array(
            [
                (
                    "needs_manual_review"
                    if ok
                    else f"tail_derivation_failed:{reason}|needs_manual_review"
                )
                for ok, reason in zip(
                    derived["tail_valid"], derived["tail_failure_reason"], strict=True
                )
            ],
            dtype=object,
        )
    values = {
        **lineage,
        **{k: v for k, v in derived.items() if k != "tail_failure_reason"},
        "keypoint_manual_edit": np.zeros((n, point_count), dtype=bool),
        "keypoint_confidences": np.where(
            np.isfinite(derived["keypoints_roi"]).all(axis=2), 1.0, np.nan
        ).astype(np.float32),
        "failure_indices": np.arange(n, dtype=np.int32),
    }
    values.update(
        {
            name: np.zeros(n, dtype=bool)
            for name in (
                "refined_success",
                "usable_keypoints",
                "geometry_valid",
                "confidence_valid",
                "edit_applied",
            )
        }
    )
    if native:
        values["source_keypoints_roi"] = arrays["source_keypoints_roi"]
    seed = create(
        paths["seed"],
        values,
        {**attrs, "recovery_payload_mutability": "immutable_seed"},
    )
    pose_edit = create(
        paths["pose_edit"],
        values,
        {**attrs, "recovery_payload_mutability": "editable_annotations"},
    )
    for group in (seed, pose_edit):
        write_reason_columns(group, reasons, chunk_size=min(n, 256), overwrite=False)
    # Complete only the local raw source before using the maintained dense
    # mask editor's seed constructor. It preserves the original mask pixels.
    family, name = paths["mask"].split("/")
    mark_run_complete(
        mask, parent_group=root[family], run_name=name, run_provenance=provenance
    )
    _, refined = prepare_refined_subject_run(
        root,
        subject_run=name,
        refined_run=paths["mask_edit"].split("/")[1],
        components=labels,
    )
    mask_edit = refined.group
    mask_edit.attrs.update(
        {**common, "recovery_payload_mutability": "editable_annotations"}
    )
    for name, values in lineage.items():
        if name not in mask_edit:
            _array(mask_edit, name, values)
    stamp_refined_subject_mask_editable_draft(mask_edit)
    np.testing.assert_array_equal(mask_edit["masks_roi"][:], arrays["masks_roi"])
    for path, group in zip(
        paths.values(), (crop, mask, mask_edit, seed, pose_edit), strict=True
    ):
        group.attrs["initial_array_sha256"] = array_hashes(group)
        group.attrs["initial_contract_sha256"] = initial_contract_digest(group)
        family, name = path.split("/")
        mark_run_complete(
            group, parent_group=root[family], run_name=name, run_provenance=provenance
        )
    failures = [
        {
            "roi_idx": i,
            **{
                name: int(arrays[name][i])
                for name in (
                    "source_merged_row",
                    "source_pose_local_row",
                    "source_training_crop_row_ids",
                )
                if name in arrays
            },
            "source_frame_idx": int(arrays["frame_indices"][i]),
            "reason": (
                str(reason)
                if not derived["tail_valid"][i]
                else f"snout_derivation_failed:{snout_reasons[i]}"
            ),
        }
        for i, reason in enumerate(derived["tail_failure_reason"])
        if not derived["tail_valid"][i]
        or (
            snout_reasons is not None
            and not derived["snout_valid"][i]
            and not (native and derived["keypoint_origin"][i, 18] == 4)
        )
    ]
    return {
        "paths": paths,
        "row_count": n,
        "pose_schema": pose_schema,
        "keypoint_count": point_count,
        **(
            {"snout_valid_count": int(derived["snout_valid"].sum())}
            if snout_reasons is not None
            else {}
        ),
        "tail_valid_count": int(derived["tail_valid"].sum()),
        "training_eligible_count": 0,
        "failures": failures,
    }
