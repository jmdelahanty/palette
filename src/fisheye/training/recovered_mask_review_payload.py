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
    recipe_with_visible_endpoint,
    registered_recipe,
)
from fisheye.training.mask_tail_border_acceptance import (
    ATTR as TAIL_ACCEPTANCE_ATTR,
    expected_row_identity,
    validate_acceptance_record,
)
from fisheye.training.recover_merged_training_recording import _sha256_array
from fisheye.tune.refined_subject_mask_review import prepare_refined_subject_run
from fisheye.tune.keypoint_failure_review import (
    _DEFAULT_CONFIDENCE_THRESHOLD,
    _resolve_review_geometry_defaults,
)
from fisheye.utils.extend_keypoint_skeleton import _schema_to_attr_payload


def run_paths(version, *, native=False, reference_crop=False):
    """Archive-relative run paths of one payload version.

    ``reference_crop`` (tail-successor format v2) omits the crop run: those
    successors bind the existing crop run through ``source_crop_run``.
    """
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
        paths = {
            "crop": f"crop_runs/mask_tail_full_roi_{version}",
            "mask": f"subject_mask_runs/mask_tail_snapshot_{version}",
            "mask_edit": f"refined_subject_masks_runs/mask_tail_edit_{version}",
            "seed": f"keypoints_runs/head_tail11_fins_seed_{version}",
            "pose_edit": f"refined_keypoints_runs/head_tail11_fins_edit_{version}",
        }
    else:
        paths = {
            "crop": f"crop_runs/recovered_full_roi_{version}",
            "mask": f"subject_mask_runs/recovered_masks_{version}",
            "mask_edit": f"refined_subject_masks_runs/recovered_masks_edit_{version}",
            "seed": f"keypoints_runs/head_tail11_fins_seed_{version}",
            "pose_edit": f"refined_keypoints_runs/head_tail11_fins_edit_{version}",
        }
    if reference_crop:
        del paths["crop"]
    return paths


def payload_crop_run(result):
    """Crop run name supplying a built/published payload's pixels."""
    paths = result["paths"]
    if "crop" in paths:
        return paths["crop"].split("/")[1]
    return str(result["source_crop_run"])


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
    root, arrays, labels, binding, *, version, pose_schema=SCHEMA_NAME, native=False,
    derivation_method="legacy",
    row_method_codes=None,
    reference_crop_run=None,
):
    """Build one payload version in a private local root.

    With ``reference_crop_run`` (tail-successor format v2) no crop run is
    written; every run records that existing crop run as ``source_crop_run``
    and ``arrays`` need not carry ``roi_images``/``source_bbox_norm_coords``.
    """
    reference_crop = reference_crop_run is not None
    n = len(arrays["masks_roi"])
    if not reference_crop and len(arrays["roi_images"]) != n:
        raise ValueError("Crop and mask row axes disagree")
    source_proof = binding.get("mask_apply_refresh") or {}
    acceptance_records = source_proof.get(
        "tail_crop_border_acceptances", {}
    )
    if not isinstance(acceptance_records, dict):
        raise ValueError("Invalid tail crop-border acceptance binding")
    accepted_rows = None
    if acceptance_records:
        accepted_rows = np.zeros(n, dtype=bool)
        body_idx = labels.index("subject_body")
        for key, record in acceptance_records.items():
            if not isinstance(key, str):
                raise ValueError("Accepted tail ROI key must be a canonical string")
            row = int(key)
            if row < 0 or row >= len(accepted_rows) or not isinstance(record, dict):
                raise ValueError("Invalid accepted tail ROI")
            actual_identity = expected_row_identity(arrays, row)
            if not validate_acceptance_record(
                key, record, body=arrays["masks_roi"][row, body_idx],
                source_crop_run=str(source_proof.get("source_crop_run") or "").split("/")[-1],
                row_identity=actual_identity,
            ):
                raise ValueError("Accepted tail ROI source binding is stale")
            accepted_rows[row] = True
    recipe = registered_recipe(
        pose_schema, method=derivation_method,
        visible_endpoint=accepted_rows is not None,
    )
    if native:
        recipe = {
            **recipe,
            "existing_keypoint_policy": "preserve_head_snout_fins_by_name_v1",
            "tail_policy": "derive_all_11_stations_from_mask",
        }
    _, schema = _schema_to_attr_payload(pose_schema)
    point_count = len(schema["keypoint_labels"])
    paths = run_paths(version, native=native, reference_crop=reference_crop)
    crop_name = (
        str(reference_crop_run) if reference_crop else paths["crop"].split("/")[1]
    )
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
    if acceptance_records and "instance_key" in arrays:
        lineage["instance_key"] = arrays["instance_key"]
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
        "source_crop_run": crop_name,
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

    crop = None if reference_crop else create(
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
    if crop is not None and "source_roi_coordinates_full" in arrays:
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

    if derivation_method == "legacy":
        if row_method_codes is not None:
            raise ValueError("Legacy recipe cannot accept per-row method codes")
        derived = derive_tail_seed(
            arrays["masks_roi"], labels, arrays["head_keypoints_roi"],
            schema_name=pose_schema,
            **({"accepted_crop_border_rows": accepted_rows} if accepted_rows is not None else {}),
        )
    else:
        codes = (
            np.ones(n, dtype=np.uint8)
            if row_method_codes is None else np.asarray(row_method_codes)
        )
        if (
            codes.shape != (n,)
            or codes.dtype != np.dtype("uint8")
            or not np.isin(codes, [0, 1]).all()
        ):
            raise ValueError("Invalid per-row tail derivation methods")
        if np.any(codes == 0):
            derived = derive_tail_seed(
                arrays["masks_roi"], labels, arrays["head_keypoints_roi"],
                schema_name=pose_schema,
                **({"accepted_crop_border_rows": accepted_rows} if accepted_rows is not None else {}),
            )
        else:
            derived = None
        if np.any(codes == 1):
            selected = codes == 1
            head_anchored = derive_tail_seed(
                arrays["masks_roi"][selected], labels,
                arrays["head_keypoints_roi"][selected],
                schema_name=pose_schema, method=derivation_method,
                **({"accepted_crop_border_rows": accepted_rows[selected]} if accepted_rows is not None else {}),
            )
            if derived is None:
                derived = head_anchored
            else:
                for name, values in head_anchored.items():
                    derived[name][selected] = values
        derived["tail_derivation_method_code"] = codes.copy()
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
        **({"tail_derivation_method_codes": {"legacy": 0, derivation_method: 1}}
           if derivation_method != "legacy" else {}),
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
    if accepted_rows is not None:
        for row in np.flatnonzero(derived["tail_tip_truncated"]):
            reasons[row] += "|tail_tip_is_visible_crop_endpoint"
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
    if acceptance_records:
        mask_edit.attrs[TAIL_ACCEPTANCE_ATTR] = {
            key: {
                **record,
                "accepted_source_crop_run": record.get("accepted_source_crop_run", record["source_crop_run"]),
                "source_crop_run": crop_name,
            }
            for key, record in acceptance_records.items()
        }
    for name, values in lineage.items():
        if name not in mask_edit:
            _array(mask_edit, name, values)
    stamp_refined_subject_mask_editable_draft(mask_edit)
    np.testing.assert_array_equal(mask_edit["masks_roi"][:], arrays["masks_roi"])
    groups = {
        "crop": crop, "mask": mask, "mask_edit": mask_edit,
        "seed": seed, "pose_edit": pose_edit,
    }
    for key, path in paths.items():
        group = groups[key]
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
        "source_crop_run": crop_name,
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
