"""Publish a mask-derived review successor without replacing saved annotations.

The existing crop-only contracts are retained: each successor seals its mask
pixels and derivation seed, and has its own editable annotation surface.  The
old mask task, seeds, annotations, and selectors are never retargeted here.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import zarr

from fisheye.shared.keypoint_motion_authority import (
    keypoint_source_crop_run_from_attributes,
)
from fisheye.shared.detect_reason_codec import (
    decode_reason_bytes,
    read_reason_labels,
    write_reason_columns,
)
from fisheye.shared.recovered_training_review_contract import (
    NATIVE_REVIEW_SCHEMA,
    initial_contract_digest,
)
from fisheye.shared.refined_subject_mask_mutation import (
    resolve_mutable_refined_subject_mask_run,
)
from fisheye.shared.run_provenance import build_writer_run_provenance, sha256_payload
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import is_run_complete
from fisheye.training.mask_tail_keypoints import (
    LEGACY_SCHEMA_NAME,
    SCHEMA_NAME,
    recipe_for_schema,
    recipe_with_visible_endpoint,
)
from fisheye.training.mask_tail_border_acceptance import bound_acceptances
from fisheye.training.recover_merged_subject_masks import (
    publish_review_payload,
    review_tasks,
    validate_initial_payload,
)
from fisheye.training.recover_merged_training_recording import _sha256_array
from fisheye.training.recovered_mask_review_payload import (
    array_hashes,
    build_review_payload,
    run_paths,
)
from fisheye.tune import keypoint_review_backend as editor
from fisheye.tune.recovered_keypoint_review import is_recovered_roi_review
from fisheye.utils.extend_keypoint_skeleton import _schema_to_attr_payload

REFRESH_SCHEMA = "palette.training.mask_apply_tail_successor.v1"
REFRESH_POLICY = "new_mask_seed_and_review_version_preserve_recorded_manual_points_v1"
_IDENTITY_ARRAYS = (
    "frame_indices",
    "source_crop_row_ids",
    "source_merged_row",
    "source_pose_local_row",
    "source_training_crop_row_ids",
    "source_roi_idx",
    "source_refined_row_ids",
    "source_detect_row_index",
)


def _safe_name(value):
    value = str(value)
    if not value or value.startswith(".") or "/" in value or "\\" in value:
        raise ValueError("Expected a non-hidden run name")
    return value


def _capture(root, *, mask_name, pose_name, revision):
    mask = resolve_mutable_refined_subject_mask_run(root, mask_name)
    pose = root[f"refined_keypoints_runs/{pose_name}"]
    crop_name = keypoint_source_crop_run_from_attributes(pose.attrs)
    crop = root[f"crop_runs/{crop_name}"]
    if not is_recovered_roi_review(root, pose, crop):
        raise ValueError("Tail refresh requires a supported crop-only training review")
    if (
        mask.attrs.get("schema_id") != pose.attrs.get("schema_id")
        or keypoint_source_crop_run_from_attributes(mask.attrs) != crop_name
        or mask.attrs.get("source_bindings") != pose.attrs.get("source_bindings")
        or initial_contract_digest(mask) != mask.attrs.get("initial_contract_sha256")
        or not is_run_complete(mask, legacy_default=False)
        or not is_run_complete(pose, legacy_default=False)
        or int(mask.attrs.get("edit_revision", 0)) != int(revision)
    ):
        raise ValueError("Mask and keypoint review source identity/revision mismatch")
    n, k, _ = pose["keypoints_roi"].shape
    schema_name = SCHEMA_NAME if k == 19 else LEGACY_SCHEMA_NAME
    _, schema = _schema_to_attr_payload(schema_name)
    if list(pose.attrs["keypoint_labels"]) != schema["keypoint_labels"]:
        raise ValueError("Unsupported tail landmark schema")
    recipe = dict(recipe_for_schema(schema_name))
    if pose.attrs["schema_id"] == NATIVE_REVIEW_SCHEMA:
        recipe.update(
            existing_keypoint_policy="preserve_head_snout_fins_by_name_v1",
            tail_policy="derive_all_11_stations_from_mask",
        )
    existing_recipe = pose.attrs.get("derivation_recipe")
    variant_recipe = recipe_with_visible_endpoint(schema_name)
    if pose.attrs["schema_id"] == NATIVE_REVIEW_SCHEMA:
        variant_recipe.update(
            existing_keypoint_policy="preserve_head_snout_fins_by_name_v1",
            tail_policy="derive_all_11_stations_from_mask",
        )
    if (
        existing_recipe not in (recipe, variant_recipe)
        or pose.attrs.get("mask_edit_policy")
        != "new_derivation_version_required_after_mask_corrections"
    ):
        raise ValueError("Tail refresh cannot change an existing derivation recipe")
    labels = tuple(mask.attrs["mask_labels"])
    accepted = bound_acceptances(mask, mask_labels=labels)
    identities = {}
    for name in (*_IDENTITY_ARRAYS, *(("instance_key",) if accepted else ())):
        present = [name in g for g in (mask, pose, crop)]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"Incomplete row identity: {name}")
        values = np.asarray(pose[name][:])
        if not all(np.array_equal(g[name][:], values) for g in (mask, crop)):
            raise ValueError(f"Mask/keypoint row identity mismatch: {name}")
        identities[name] = values
    masks = np.asarray(mask["masks_roi"][:])
    points = np.asarray(pose["keypoints_roi"][:])
    manual = np.asarray(pose["keypoint_manual_edit"][:], dtype=bool)
    origins = np.asarray(pose["keypoint_origin"][:])
    original_reasons = read_reason_labels(pose)
    if (
        masks.dtype != np.uint8
        or masks.ndim != 4
        or masks.shape[0] != n
        or masks.shape[1] != len(labels)
        or np.any(masks > 1)
        or np.any(~np.isin(origins, list(pose.attrs["keypoint_origin_codes"].values())))
        or not np.array_equal(manual, origins == 3)
        or original_reasons is None
        or len(original_reasons) != n
    ):
        raise ValueError("Invalid dense mask or manual-origin provenance")
    source_mask_name = _safe_name(mask.attrs.get("source_subject_mask_run") or "")
    original = root[f"subject_mask_runs/{source_mask_name}"]
    if keypoint_source_crop_run_from_attributes(original.attrs) != crop_name:
        raise ValueError("Mask source crop mismatch")
    binding = dict(pose.attrs["source_bindings"])
    # Retain the original supplier declaration without recursive generations.
    binding.pop("mask_apply_refresh", None)
    proof = {
        "schema_id": REFRESH_SCHEMA,
        "policy": REFRESH_POLICY,
        "source_mask_run": str(mask.path),
        "source_pose_run": str(pose.path),
        "source_crop_run": str(crop.path),
        "source_mask_edit_revision": int(revision),
        "source_pose_edit_revision": int(pose.attrs.get("edit_revision", 0)),
        "source_mask_contract_sha256": initial_contract_digest(mask),
        "source_pose_contract_sha256": initial_contract_digest(pose),
        "source_mask_sha256": _sha256_array(masks),
        "source_points_sha256": _sha256_array(points),
        "source_manual_sha256": _sha256_array(manual),
        "source_origins_sha256": _sha256_array(origins),
        "source_reasons_sha256": sha256_payload([str(v) for v in original_reasons]),
        "row_identity_sha256": {
            name: _sha256_array(value) for name, value in identities.items()
        },
        "recovered_review_qc": dict(pose.attrs["recovered_review_qc"]),
    }
    if accepted:
        proof["tail_crop_border_acceptances"] = accepted
    arrays = {
        **identities,
        "masks_roi": masks,
        "head_keypoints_roi": points[:, :3].copy(),
        "roi_images": np.asarray(crop["roi_images"][:]),
        "source_bbox_norm_coords": np.asarray(crop["source_bbox_norm_coords"][:]),
        "detection_source": np.asarray(original["detection_source"][:]),
        "target_valid_channels": np.asarray(original["target_valid_channels"][:]),
    }
    for name in ("source_roi_coordinates_full",):
        if name in crop:
            arrays[name] = np.asarray(crop[name][:])
    for name in ("label_origin_codes", "supervision_mode_codes"):
        if name in original:
            arrays[name] = np.asarray(original[name][:])
    if pose.attrs["schema_id"] == NATIVE_REVIEW_SCHEMA:
        arrays["source_keypoints_roi"] = np.asarray(pose["source_keypoints_roi"][:])
    return (
        arrays,
        labels,
        binding,
        proof,
        points,
        manual,
        origins,
        schema_name,
        original_reasons,
    )


def _carry_labels(local, result, *, points, manual, origins, proof, original_reasons):
    paths = result["paths"]
    root = zarr.open_group(str(local), mode="a", use_consolidated=False)
    seed, target = (root[paths[name]] for name in ("seed", "pose_edit"))
    seed_points = np.asarray(seed["keypoints_roi"][:])
    seed_origins = np.asarray(seed["keypoint_origin"][:])
    # The current head is the orientation supplier. Native existing landmarks
    # also remain existing landmarks; only automatic mask-derived points renew.
    retained = np.zeros_like(manual)
    retained[:, :3] = True
    retained |= origins == 4
    retained[:, 3:14] = False
    seed_points[retained] = points[retained]
    seed_origins[retained] = origins[retained]
    for group in (seed, target):
        group["keypoints_roi"][:] = seed_points
        group["keypoint_origin"][:] = seed_origins
        group["keypoint_manual_edit"][:] = retained & manual
        group["keypoint_confidences"][:] = np.where(
            np.isfinite(seed_points).all(axis=2), 1.0, np.nan
        ).astype(np.float32)
        group.attrs["recovered_review_qc"] = proof["recovered_review_qc"]
        group.attrs["initial_array_sha256"] = array_hashes(group)
        group.attrs["initial_contract_sha256"] = initial_contract_digest(group)
    expected = seed_points.copy()
    expected[manual] = points[manual]  # Includes deliberate manual clears/NaNs.
    final_origins = seed_origins.copy()
    final_origins[manual] = origins[manual]
    session = editor.resolve_review_session(
        str(local),
        refined_run=paths["pose_edit"].split("/")[1],
        crop_run=paths["crop"].split("/")[1],
        include_all=True,
    )
    height, width = session.roi_images.shape[1:3]
    finite_inside = (
        np.isfinite(expected).all(axis=(1, 2))
        & (expected >= 0).all(axis=(1, 2))
        & (expected[:, :, 0] < width).all(axis=1)
        & (expected[:, :, 1] < height).all(axis=1)
    )
    for row in np.flatnonzero(finite_inside):
        editor.save_roi_correction(session, position=int(row), points=expected[row])
    target = session.refined
    target["keypoints_roi"][:] = expected
    target["keypoint_origin"][:] = final_origins
    target["keypoint_manual_edit"][:] = manual
    target["edit_applied"][:] = manual.any(axis=1)
    target["keypoint_confidences"][:] = np.where(
        np.isfinite(expected).all(axis=2), 1.0, np.nan
    ).astype(np.float32)
    reasons = read_reason_labels(target)
    for row in range(len(expected)):
        # Retain operator annotations, but replace statuses derived from old
        # geometry/masks. Running the editor's QC is not itself a manual edit.
        tags = [tag for tag in str(reasons[row]).split("|") if tag and tag != "clean"]
        if np.isfinite(expected[row, 14:18]).all():
            tags = [tag for tag in tags if tag != "needs_manual_fins"]
        for tag in str(original_reasons[row]).split("|"):
            if (
                tag
                and tag not in {"clean", "geometry_issue", "needs_manual_fins"}
                and not tag.startswith(
                    ("tail_derivation_failed:", "snout_derivation_failed:")
                )
                and tag not in tags
            ):
                tags.append(tag)
        if not manual[row].any():
            tags = [tag for tag in tags if tag != "manual_correction"]
        reasons[row] = "|".join(tags) or "clean"
    write_reason_columns(
        target, reasons, chunk_size=target["reason_bytes"].chunks[0], overwrite=True
    )
    eligible = np.asarray(target["training_eligible"][:], dtype=bool)
    eligible &= np.asarray(target["tail_valid"][:], dtype=bool) & finite_inside
    target["training_eligible"][:] = eligible
    target["usable_keypoints"][:] = eligible
    # The immutable seed and editable initial snapshot each bind their own data.
    for path in paths.values():
        group = root[path]
        group.attrs["initial_array_sha256"] = array_hashes(group)
        group.attrs["initial_contract_sha256"] = initial_contract_digest(group)
    result["manual_point_count"] = int(manual.sum())
    result["training_eligible_count"] = int(eligible.sum())
    return result


def _reuse_completed(archive, paths, binding):
    """Accept untouched immutable children and legal edits in an existing draft."""
    if not all((archive / path).exists() for path in paths.values()):
        return False
    for name, path in paths.items():
        group = zarr.open_group(str(archive / path), mode="r", use_consolidated=False)
        if (
            group.attrs.get("source_bindings") != binding
            or initial_contract_digest(group)
            != group.attrs.get("initial_contract_sha256")
            or not is_run_complete(group, legacy_default=False)
        ):
            raise ValueError("Conflicting existing tail refresh version")
        if (
            name not in {"pose_edit", "mask_edit"}
            and not validate_initial_payload(archive / path)["valid"]
        ):
            raise ValueError("Changed immutable tail refresh source")
    return True


def validate_completed_tail_version(*, archive, version, source_bindings):
    """Validate the sealed snapshot recorded by a completed browser effect.

    The original editable pose may subsequently advance. Reusing this historical
    snapshot never rebases it or claims to include those later annotations.
    """
    proof = source_bindings.get("mask_apply_refresh") or {}
    if (
        proof.get("schema_id") != REFRESH_SCHEMA
        or proof.get("policy") != REFRESH_POLICY
        or version != "mask_apply_" + sha256_payload(proof)[:24]
    ):
        raise ValueError("Invalid completed tail-version source proof")
    archive = Path(archive).resolve()
    paths = run_paths(
        version,
        native=source_bindings.get("source_kind")
        == "native_reviewed_training_masks_v1",
    )
    with archive_metadata_publication_lock(archive):
        if not _reuse_completed(archive, paths, source_bindings):
            raise ValueError(
                "Completed tail successor has missing publication children"
            )
    return paths


def regenerate_training_tail_version(
    *,
    archive,
    refined_mask_run,
    refined_keypoint_run,
    apply_id,
    expected_mask_revision,
    scratch_root=Path("/tmp"),
):
    """Create/reuse one unselected successor from applied masks and saved labels.

    The caller owns the refined mask write lock. This function serializes with
    canonical keypoint Apply/publication using the archive lock. Browser-only or
    unapplied keypoint edits are not inputs; the browser adapter must check its
    checkpoint store before invoking this publisher.
    """
    archive = Path(archive).resolve()
    mask_name, pose_name = _safe_name(refined_mask_run), _safe_name(
        refined_keypoint_run
    )
    if not str(apply_id).strip():
        raise ValueError("A durable mask Apply ID is required")
    with archive_metadata_publication_lock(archive):
        root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
        captured = _capture(
            root,
            mask_name=mask_name,
            pose_name=pose_name,
            revision=expected_mask_revision,
        )
        (
            arrays,
            labels,
            original_binding,
            proof,
            points,
            manual,
            origins,
            schema_name,
            original_reasons,
        ) = captured
        proof = {**proof, "apply_id": str(apply_id)}
        # One durable Apply cannot silently change its source after a partial
        # publication or a later failed secondary effect.
        for _, previous in root["crop_runs"].groups():
            prior = (previous.attrs.get("source_bindings") or {}).get(
                "mask_apply_refresh"
            )
            if (
                isinstance(prior, dict)
                and prior.get("apply_id") == str(apply_id)
                and prior.get("source_mask_run") == proof["source_mask_run"]
                and prior != proof
            ):
                raise ValueError(
                    "Source changed since this Apply's tail version was published"
                )
        binding = {**original_binding, "mask_apply_refresh": proof}
        version = "mask_apply_" + sha256_payload(proof)[:24]
        native = (
            root[f"refined_keypoints_runs/{pose_name}"].attrs["schema_id"]
            == NATIVE_REVIEW_SCHEMA
        )
        paths = run_paths(version, native=native)
        reused = _reuse_completed(archive, paths, binding)
        publications = []
        if not reused:
            for key in ("crop", "mask"):
                source_path = (
                    proof["source_crop_run"]
                    if key == "crop"
                    else f"subject_mask_runs/{root[f'refined_subject_masks_runs/{mask_name}'].attrs['source_subject_mask_run']}"
                )
                if not validate_initial_payload(archive / source_path)["valid"]:
                    raise ValueError("Immutable source crop/mask payload is invalid")

            def source_check(_current=None):
                current = zarr.open_group(
                    str(archive), mode="r", use_consolidated=False
                )
                current_proof = _capture(
                    current,
                    mask_name=mask_name,
                    pose_name=pose_name,
                    revision=expected_mask_revision,
                )[3]
                if {**current_proof, "apply_id": str(apply_id)} != proof:
                    raise ValueError(
                        "Mask or keypoint source changed during tail refresh"
                    )

            with tempfile.TemporaryDirectory(
                prefix="palette-mask-apply-tail-", dir=scratch_root
            ) as temp:
                local = Path(temp) / "review.zarr"
                target_root = zarr.open_group(
                    str(local), mode="w", use_consolidated=False
                )
                target_root.attrs.update(dict(root.attrs))
                result = build_review_payload(
                    target_root,
                    arrays,
                    labels,
                    binding,
                    version=version,
                    native=native,
                    pose_schema=schema_name,
                )
                _carry_labels(
                    local,
                    result,
                    points=points,
                    manual=manual,
                    origins=origins,
                    proof=proof,
                    original_reasons=original_reasons,
                )
                provenance = build_writer_run_provenance(
                    command="fisheye.training.mask_tail_apply_refresh",
                    params={
                        "policy": REFRESH_POLICY,
                        "version": version,
                        "source": proof,
                    },
                    input_run_ids={"masks": mask_name, "keypoints": pose_name},
                )
                for path in paths.values():
                    target_root[path].attrs["run_provenance"] = provenance
                source_check()
                publications = publish_review_payload(
                    archive, local, paths, binding, source_check, resume=True
                )
        consolidate_metadata_capture_expected_warnings(archive)
        published = zarr.open_group(str(archive), mode="r", use_consolidated=True)
        pose = published[paths["pose_edit"]]
        seed = published[paths["seed"]]
        tail_ok = np.asarray(seed["tail_valid"][:], dtype=bool)
        reasons = read_reason_labels(seed)
        failures = [
            {
                "roi_idx": int(row),
                "source_frame_idx": int(pose["frame_indices"][row]),
                "reason": next(
                    (
                        tag.removeprefix("tail_derivation_failed:")
                        for tag in str(reasons[row]).split("|")
                        if tag.startswith("tail_derivation_failed:")
                    ),
                    "tail_derivation_failed",
                ),
            }
            for row in np.flatnonzero(~tail_ok)
        ]
        if "snout_valid" in seed:
            snout_reasons = decode_reason_bytes(seed["snout_failure_reason_bytes"][:])
            for row in np.flatnonzero(
                tail_ok & ~np.asarray(seed["snout_valid"][:], dtype=bool)
            ):
                failures.append(
                    {
                        "roi_idx": int(row),
                        "source_frame_idx": int(pose["frame_indices"][row]),
                        "reason": "snout_derivation_failed:" + str(snout_reasons[row]),
                    }
                )
        result = {
            "status": "reused" if reused else "generated",
            "schema_id": REFRESH_SCHEMA,
            "version": version,
            "paths": paths,
            "source_bindings": binding,
            "source_mask_edit_revision": int(expected_mask_revision),
            "source_pose_run": proof["source_pose_run"],
            "row_count": len(tail_ok),
            "keypoint_count": int(pose["keypoints_roi"].shape[1]),
            "tail_valid_count": int(tail_ok.sum()),
            "failures": failures,
            "manual_point_count": int(manual.sum()),
            "training_eligible_count": int(
                np.count_nonzero(pose["training_eligible"][:])
            ),
            "publications": publications,
        }
        if "tail_tip_truncated" in seed:
            result["visible_endpoint_rows"] = [
                int(row) for row in np.flatnonzero(seed["tail_tip_truncated"][:])
            ]
        result["tasks"] = review_tasks(
            archive, binding["recording_id"], result, version
        )
        if native:
            for task in result["tasks"]:
                task["dataset_id"] = (
                    f"{binding['recording_id']}:native_mask_tail:{version}"
                )
        return result
