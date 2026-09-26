"""Publish a mask-derived review successor without replacing saved annotations.

The existing crop-only contracts are retained: each successor seals its mask
pixels and derivation seed, and has its own editable annotation surface.  The
old mask task, seeds, annotations, and selectors are never retargeted here.

Successor formats (the policy id is part of every version digest):

- ``v1`` (``REFRESH_POLICY``): five runs, including an identity copy of the
  source crop run, written chunk-only. Historical; still readable, and still
  used to resume an Apply whose v1 publication was already started.
- ``v2`` (``REFRESH_POLICY_V2``): four runs that reference the existing crop
  run through ``source_crop_run`` (its contract digest is bound in the
  proof), laid out with the ``training_review_run_v1`` shard profile.
- ``v3`` (``REFRESH_POLICY_V3``, the default): v2's layout, and each row's
  tail method is selected from the current masks: the legacy method first,
  and the head-anchored method only where legacy fails at the snout join
  (``FALLBACK_FAILURE_REASONS``) and head-anchored yields a valid tail. Rows
  legacy derives keep exactly the legacy output. Per-row method codes and
  the selection policy are recorded in the proof.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
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
    HEAD_ANCHORED_FALLBACK_SUCCESSOR_POLICY,
    REFERENCED_CROP_SUCCESSOR_POLICIES,
    REFERENCED_CROP_SUCCESSOR_POLICY,
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
from fisheye.shared.zarr.training_review_run_storage import (
    shard_training_review_run,
)
from fisheye.shared.zarr_run_completion import is_run_complete, require_runs_parent
from fisheye.training.mask_tail_keypoints import (
    LEGACY_SCHEMA_NAME,
    SCHEMA_NAME,
    recipe_for_schema,
    recipe_with_visible_endpoint,
    registered_recipe,
    derive_tail_seed,
)
from fisheye.analysis.subject_shape_runs import HEAD_ANCHORED_CENTERLINE_METHOD
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
    payload_crop_run,
    run_paths,
)
from fisheye.tune import keypoint_review_backend as editor
from fisheye.tune.recovered_keypoint_review import is_recovered_roi_review
from fisheye.utils.extend_keypoint_skeleton import _schema_to_attr_payload

REFRESH_SCHEMA = "palette.training.mask_apply_tail_successor.v1"
# v2 proofs add ``source_crop_contract_sha256``; the grammar version says so.
REFRESH_SCHEMA_V2 = "palette.training.mask_apply_tail_successor.v2"
REFRESH_POLICY = "new_mask_seed_and_review_version_preserve_recorded_manual_points_v1"
REFRESH_POLICY_V2 = REFERENCED_CROP_SUCCESSOR_POLICY
# v3 proofs add ``tail_method_selection``.
REFRESH_SCHEMA_V3 = "palette.training.mask_apply_tail_successor.v3"
REFRESH_POLICY_V3 = HEAD_ANCHORED_FALLBACK_SUCCESSOR_POLICY
SUCCESSOR_FORMAT_POLICIES = {"v1": REFRESH_POLICY, "v2": REFRESH_POLICY_V2, "v3": REFRESH_POLICY_V3}
SUCCESSOR_FORMAT_SCHEMAS = {
    REFRESH_POLICY: REFRESH_SCHEMA,
    REFRESH_POLICY_V2: REFRESH_SCHEMA_V2,
    REFRESH_POLICY_V3: REFRESH_SCHEMA_V3,
}
DEFAULT_SUCCESSOR_FORMAT = "v3"
_REFERENCED_CROP_FORMATS = frozenset({"v2", "v3"})
TAIL_METHOD_SELECTION_POLICY = "legacy_first_head_anchored_fallback_v1"
# Legacy failures at the snout join, which the head-anchored route addresses.
FALLBACK_FAILURE_REASONS = ("snout_extension_too_long", "snout_extension_no_mask_path")
_EDITABLE_RUN_KEYS = frozenset({"pose_edit", "mask_edit"})
# The first child each format publishes: v1 its crop copy, v2 its mask snapshot.
# A partial publication is therefore always visible in one of these families.
_PROOF_FAMILIES = ("crop_runs", "subject_mask_runs")
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


def _source_run_paths(root, *, mask_name, pose_name):
    """Archive-relative run groups whose contents `_capture` reads."""

    mask = root[f"refined_subject_masks_runs/{mask_name}"]
    pose = root[f"refined_keypoints_runs/{pose_name}"]
    paths = [
        f"refined_subject_masks_runs/{mask_name}",
        f"refined_keypoints_runs/{pose_name}",
        f"crop_runs/{_safe_name(keypoint_source_crop_run_from_attributes(pose.attrs))}",
    ]
    if mask.attrs.get("source_subject_mask_run"):
        paths.append(f"subject_mask_runs/{_safe_name(mask.attrs['source_subject_mask_run'])}")
    seed_name = pose.attrs.get("source_seed_run")
    if seed_name:
        paths.append(f"keypoints_runs/{_safe_name(seed_name)}")
        seed = root.get(f"keypoints_runs/{seed_name}")
        seed_mask = seed.attrs.get("source_subject_mask_run") if seed is not None else None
        if seed_mask:
            paths.append(f"subject_mask_runs/{_safe_name(seed_mask)}")
    return tuple(sorted(set(paths)))


def _source_file_fingerprint(archive, run_paths):
    """Stat identity of every file in the source runs.

    Zarr stores replace files atomically (new inode) on every write, so any
    write to a source run changes this fingerprint without re-reading and
    re-hashing the array contents.
    """

    entries = []
    for run_path in run_paths:
        for directory, _dirs, files in os.walk(archive / run_path):
            for name in files:
                path = Path(directory) / name
                stat = path.stat()
                entries.append(
                    (
                        str(path.relative_to(archive)),
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
    return tuple(sorted(entries))


def _policy_format(policy):
    for name, value in SUCCESSOR_FORMAT_POLICIES.items():
        if value == policy:
            return name
    raise ValueError(f"Unknown tail successor policy: {policy!r}")


def _prior_apply_proofs(root, *, apply_id, source_mask_run):
    """Refresh proofs already published (possibly partially) for this Apply."""
    proofs = []
    for family in _PROOF_FAMILIES:
        parent = root.get(family)
        if parent is None:
            continue
        for _, previous in parent.groups():
            prior = (previous.attrs.get("source_bindings") or {}).get(
                "mask_apply_refresh"
            )
            if (
                isinstance(prior, dict)
                and prior.get("apply_id") == str(apply_id)
                and prior.get("source_mask_run") == source_mask_run
                and prior not in proofs
            ):
                proofs.append(prior)
    return proofs


def _capture(
    root, *, mask_name, pose_name, revision, upgrade_target_rows=None,
    successor_format=DEFAULT_SUCCESSOR_FORMAT,
):
    reference_crop = successor_format in _REFERENCED_CROP_FORMATS
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
    upgraded_recipe = registered_recipe(
        schema_name, method=HEAD_ANCHORED_CENTERLINE_METHOD,
        visible_endpoint=bool("crop_border_policy" in (existing_recipe or {})),
    ) if schema_name == SCHEMA_NAME else None
    if upgraded_recipe is not None and pose.attrs["schema_id"] == NATIVE_REVIEW_SCHEMA:
        upgraded_recipe.update(
            existing_keypoint_policy="preserve_head_snout_fins_by_name_v1",
            tail_policy="derive_all_11_stations_from_mask",
        )
    source_is_legacy = existing_recipe in (recipe, variant_recipe)
    source_is_upgraded = upgraded_recipe is not None and existing_recipe == upgraded_recipe
    if (
        not (source_is_legacy or source_is_upgraded)
        or (upgrade_target_rows is not None and not source_is_legacy)
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
    preserved_pose = None
    preserved_seed_reasons = None
    row_method_codes = None
    if source_is_upgraded:
        if "tail_derivation_method_code" not in pose:
            raise ValueError("Upgraded source lacks per-row method identity")
        raw_method_codes = np.asarray(pose["tail_derivation_method_code"][:])
        if (
            raw_method_codes.shape != (n,)
            or raw_method_codes.dtype != np.dtype("uint8")
            or not np.isin(raw_method_codes, [0, 1]).all()
        ):
            raise ValueError("Invalid upgraded per-row method identity")
        row_method_codes = raw_method_codes
        expected_codes = {"legacy": 0, HEAD_ANCHORED_CENTERLINE_METHOD: 1}
        seed_name = _safe_name(pose.attrs.get("source_seed_run") or "")
        seed = root[f"keypoints_runs/{seed_name}"]
        code_hash = _sha256_array(row_method_codes)
        if (
            pose.attrs.get("tail_derivation_method_codes") != expected_codes
            or seed.attrs.get("tail_derivation_method_codes") != expected_codes
            or "tail_derivation_method_code" not in seed
            or not np.array_equal(seed["tail_derivation_method_code"][:], row_method_codes)
            or (pose.attrs.get("initial_array_sha256") or {}).get("tail_derivation_method_code") != code_hash
            or (seed.attrs.get("initial_array_sha256") or {}).get("tail_derivation_method_code") != code_hash
            or initial_contract_digest(seed) != seed.attrs.get("initial_contract_sha256")
            or array_hashes(seed) != seed.attrs.get("initial_array_sha256")
            or not is_run_complete(seed, legacy_default=False)
        ):
            raise ValueError("Upgraded per-row method identity or seed is stale")
    selected_rows = None
    preserved_seed = None
    if upgrade_target_rows is not None:
        requested_rows = list(upgrade_target_rows)
        if any(
            isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer))
            for row in requested_rows
        ):
            raise ValueError("Upgrade target rows must be integer row indices")
        selected_rows = sorted({int(row) for row in requested_rows})
        if not selected_rows or any(row < 0 or row >= n for row in selected_rows):
            raise ValueError("Upgrade requires valid explicit target rows")
        if any(
            "tail_derivation_failed:snout_extension_too_long" not in str(original_reasons[row]).split("|")
            for row in selected_rows
        ):
            raise ValueError("Upgrade target is not a recorded long-snout failure")
        seed_name = _safe_name(pose.attrs.get("source_seed_run") or "")
        seed = root[f"keypoints_runs/{seed_name}"]
        if (
            seed.attrs.get("source_bindings") != pose.attrs.get("source_bindings")
            or initial_contract_digest(seed) != seed.attrs.get("initial_contract_sha256")
            or array_hashes(seed) != seed.attrs.get("initial_array_sha256")
            or not is_run_complete(seed, legacy_default=False)
        ):
            raise ValueError("Upgrade source seed identity is invalid")
        seed_mask_name = _safe_name(seed.attrs.get("source_subject_mask_run") or "")
        seed_mask = root[f"subject_mask_runs/{seed_mask_name}"]
        unchanged = np.ones(n, dtype=bool)
        unchanged[selected_rows] = False
        old_masks = np.asarray(seed_mask["masks_roi"][:])
        changed_non_target = np.flatnonzero(
            unchanged & np.any(masks != old_masks, axis=(1, 2, 3))
        ).tolist()
        preserved_seed = {
            name: np.asarray(array[:]) for name, array in seed.arrays()
            if array.shape and array.shape[0] == n
        }
        preserved_pose = {
            name: np.asarray(array[:]) for name, array in pose.arrays()
            if array.shape and array.shape[0] == n
        }
        preserved_seed_reasons = read_reason_labels(seed)
        accepted_rows = np.zeros(n, dtype=bool)
        for key in accepted:
            accepted_rows[int(key)] = True
        legacy_now = derive_tail_seed(
            masks, labels, points[:, :3], schema_name=schema_name,
            accepted_crop_border_rows=accepted_rows if accepted else None,
        )
        row_method_codes = np.zeros(n, dtype=np.uint8)
        for row in selected_rows:
            if str(legacy_now["tail_failure_reason"][row]) == "snout_extension_too_long":
                row_method_codes[row] = 1
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
        "schema_id": SUCCESSOR_FORMAT_SCHEMAS[SUCCESSOR_FORMAT_POLICIES[successor_format]],
        "policy": SUCCESSOR_FORMAT_POLICIES[successor_format],
        "source_mask_run": str(mask.path),
        "source_pose_run": str(pose.path),
        "source_crop_run": str(crop.path),
        # v2 references these pixels instead of copying them; bind the crop's
        # identity (its contract digest covers its initial array hashes).
        **(
            {"source_crop_contract_sha256": initial_contract_digest(crop)}
            if reference_crop
            else {}
        ),
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
    if selected_rows is not None:
        proof["geometry_upgrade"] = {
            "method": HEAD_ANCHORED_CENTERLINE_METHOD,
            "target_rows": selected_rows,
            "source_recipe": existing_recipe,
            "source_seed_run": str(seed.path),
            "source_seed_contract_sha256": initial_contract_digest(seed),
            "source_seed_array_sha256": dict(seed.attrs["initial_array_sha256"]),
            "source_pose_array_sha256": array_hashes(pose),
            "non_target_policy": "copy_source_seed_exact_v1",
            "non_target_mask_changed_rows": {
                str(row): {
                    "source_mask_sha256": _sha256_array(old_masks[row]),
                    "current_mask_sha256": _sha256_array(masks[row]),
                }
                for row in changed_non_target
            },
            "legacy_rechecked_rows": selected_rows,
            "head_anchored_rows": [row for row in selected_rows if row_method_codes[row] == 1],
            "legacy_recheck_reasons": [
                str(legacy_now["tail_failure_reason"][row]) for row in selected_rows
            ],
        }
    if row_method_codes is not None:
        proof["source_row_method_codes_sha256"] = _sha256_array(row_method_codes)
    fallback = (
        successor_format == "v3"
        and upgrade_target_rows is None
        and upgraded_recipe is not None
    )
    if fallback:
        row_method_codes = _fallback_method_codes(
            masks, labels, points, schema_name=schema_name, accepted=accepted
        )
        proof["tail_method_selection"] = {
            "policy": TAIL_METHOD_SELECTION_POLICY,
            "fallback_failure_reasons": list(FALLBACK_FAILURE_REASONS),
            "row_method_codes_sha256": _sha256_array(row_method_codes),
            "head_anchored_rows": [int(row) for row in np.flatnonzero(row_method_codes)],
        }
    if accepted:
        proof["tail_crop_border_acceptances"] = accepted
    arrays = {
        **identities,
        "masks_roi": masks,
        "head_keypoints_roi": points[:, :3].copy(),
        "detection_source": np.asarray(original["detection_source"][:]),
        "target_valid_channels": np.asarray(original["target_valid_channels"][:]),
    }
    if not reference_crop:
        # v1 copies the crop run's pixels and boxes into its own crop run.
        arrays["roi_images"] = np.asarray(crop["roi_images"][:])
        arrays["source_bbox_norm_coords"] = np.asarray(
            crop["source_bbox_norm_coords"][:]
        )
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
        HEAD_ANCHORED_CENTERLINE_METHOD
        if (selected_rows is not None or source_is_upgraded or fallback)
        else "legacy",
        preserved_seed,
        preserved_pose,
        preserved_seed_reasons,
        selected_rows,
        row_method_codes,
    )


def _fallback_method_codes(masks, labels, points, *, schema_name, accepted):
    """Per-row tail method under legacy-first, head-anchored fallback.

    ``1`` only where legacy fails with a ``FALLBACK_FAILURE_REASONS`` reason
    and the head-anchored method derives a valid tail; otherwise ``0``.
    """

    n = len(masks)
    accepted_rows = np.zeros(n, dtype=bool)
    for key in accepted:
        accepted_rows[int(key)] = True
    border = {"accepted_crop_border_rows": accepted_rows} if accepted else {}
    legacy = derive_tail_seed(masks, labels, points[:, :3], schema_name=schema_name, **border)
    reasons = [str(value) for value in legacy["tail_failure_reason"]]
    valid = np.asarray(legacy["tail_valid"], dtype=bool)
    candidates = np.flatnonzero(
        ~valid & np.asarray([reason in FALLBACK_FAILURE_REASONS for reason in reasons])
    )
    codes = np.zeros(n, dtype=np.uint8)
    if candidates.size:
        retry = derive_tail_seed(
            masks[candidates], labels, points[candidates][:, :3],
            schema_name=schema_name, method=HEAD_ANCHORED_CENTERLINE_METHOD,
            **({"accepted_crop_border_rows": accepted_rows[candidates]} if accepted else {}),
        )
        codes[candidates[np.asarray(retry["tail_valid"], dtype=bool)]] = 1
    return codes


def _carry_labels(
    local, result, *, points, manual, origins, proof, original_reasons,
    preserved_seed=None, preserved_pose=None, preserved_seed_reasons=None,
    selected_rows=None, archive=None,
):
    paths = result["paths"]
    root = zarr.open_group(str(local), mode="a", use_consolidated=False)
    if "crop" not in paths:
        # v2: expose the referenced immutable crop to the editor session in
        # the private scratch root only. It is read (image shape) and never
        # written, published, or retained.
        require_runs_parent(root, "crop_runs")
        crop_link = Path(local) / "crop_runs" / payload_crop_run(result)
        crop_link.symlink_to(Path(archive) / proof["source_crop_run"])
        try:
            return _carry_labels_into(
                root, local, result, points=points, manual=manual,
                origins=origins, proof=proof, original_reasons=original_reasons,
                preserved_seed=preserved_seed, preserved_pose=preserved_pose,
                preserved_seed_reasons=preserved_seed_reasons,
                selected_rows=selected_rows,
            )
        finally:
            crop_link.unlink()  # Remove the link itself before the parent.
            shutil.rmtree(crop_link.parent)
    return _carry_labels_into(
        root, local, result, points=points, manual=manual, origins=origins,
        proof=proof, original_reasons=original_reasons,
        preserved_seed=preserved_seed, preserved_pose=preserved_pose,
        preserved_seed_reasons=preserved_seed_reasons, selected_rows=selected_rows,
    )


def _carry_labels_into(
    root, local, result, *, points, manual, origins, proof, original_reasons,
    preserved_seed=None, preserved_pose=None, preserved_seed_reasons=None,
    selected_rows=None,
):
    paths = result["paths"]
    seed, target = (root[paths[name]] for name in ("seed", "pose_edit"))
    if preserved_seed is not None:
        keep = np.ones(len(points), dtype=bool)
        keep[selected_rows] = False
        for group in (seed, target):
            for name, source in preserved_seed.items():
                if name != "reason_bytes" and name in group and group[name].shape == source.shape:
                    values = np.asarray(group[name][:])
                    values[keep] = source[keep]
                    group[name][:] = values
            method_codes = np.asarray(group["tail_derivation_method_code"][:])
            method_codes[keep] = 0
            group["tail_derivation_method_code"][:] = method_codes
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
        crop_run=payload_crop_run(result),
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
    if preserved_seed is not None:
        # The upgrade is row-scoped. Preserve both the immutable numerical
        # seed and the independently edited annotation surface on other rows.
        for group, source in ((seed, preserved_seed), (target, preserved_pose)):
            for name, old_values in source.items():
                if name == "reason_bytes" or name not in group or group[name].shape != old_values.shape:
                    continue
                values = np.asarray(group[name][:])
                values[keep] = old_values[keep]
                group[name][:] = values
            reason_values = read_reason_labels(group)
            old_reasons = preserved_seed_reasons if group is seed else original_reasons
            reason_values[keep] = old_reasons[keep]
            write_reason_columns(
                group, reason_values,
                chunk_size=group["reason_bytes"].chunks[0], overwrite=True,
            )
        # New method identity is the only intentional added row field.
        assert np.array_equal(seed["tail_derivation_method_code"][:][keep], np.zeros(int(keep.sum()), dtype=np.uint8))
        assert np.array_equal(target["tail_derivation_method_code"][:][keep], np.zeros(int(keep.sum()), dtype=np.uint8))
        eligible = np.asarray(target["training_eligible"][:], dtype=bool)
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
    proof = binding.get("mask_apply_refresh") or {}
    if "crop" not in paths:
        # v2: the referenced crop run is an immutable input of this version.
        crop_path = archive / str(proof.get("source_crop_run") or "")
        crop = zarr.open_group(str(crop_path), mode="r", use_consolidated=False)
        if (
            not validate_initial_payload(crop_path)["valid"]
            or initial_contract_digest(crop)
            != proof.get("source_crop_contract_sha256")
        ):
            raise ValueError("Changed immutable tail refresh source crop")
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
        proof.get("policy") not in SUCCESSOR_FORMAT_SCHEMAS
        or proof.get("schema_id") != SUCCESSOR_FORMAT_SCHEMAS[proof.get("policy")]
        or version != "mask_apply_" + sha256_payload(proof)[:24]
    ):
        raise ValueError("Invalid completed tail-version source proof")
    archive = Path(archive).resolve()
    paths = run_paths(
        version,
        native=source_bindings.get("source_kind")
        == "native_reviewed_training_masks_v1",
        reference_crop=proof["policy"] in REFERENCED_CROP_SUCCESSOR_POLICIES,
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
    upgrade_target_rows=None,
    successor_format=None,
):
    """Create/reuse one unselected successor from applied masks and saved labels.

    ``successor_format`` defaults to v2, except that an Apply whose earlier
    (possibly partial) publication used v1 resumes in v1 so the same source
    keeps the same version. Passing a format explicitly is for compatibility
    tests and reproduction only.

    The caller owns the refined mask write lock. This function serializes with
    canonical keypoint Apply/publication using the archive lock. Browser-only or
    unapplied keypoint edits are not inputs; the browser adapter must check its
    checkpoint store before invoking this publisher.
    """
    archive = Path(archive).resolve()
    if upgrade_target_rows is not None:
        upgrade_target_rows = tuple(upgrade_target_rows)
    mask_name, pose_name = _safe_name(refined_mask_run), _safe_name(
        refined_keypoint_run
    )
    if not str(apply_id).strip():
        raise ValueError("A durable mask Apply ID is required")
    with archive_metadata_publication_lock(archive):
        root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
        prior_proofs = _prior_apply_proofs(
            root,
            apply_id=apply_id,
            source_mask_run=f"refined_subject_masks_runs/{mask_name}",
        )
        if successor_format is None:
            prior_formats = {_policy_format(p.get("policy")) for p in prior_proofs}
            successor_format = (
                prior_formats.pop()
                if len(prior_formats) == 1
                else DEFAULT_SUCCESSOR_FORMAT
            )
        if successor_format not in SUCCESSOR_FORMAT_POLICIES:
            raise ValueError(f"Unknown tail successor format: {successor_format!r}")
        reference_crop = successor_format in _REFERENCED_CROP_FORMATS
        source_runs = _source_run_paths(
            root, mask_name=mask_name, pose_name=pose_name
        )
        source_fingerprint = _source_file_fingerprint(archive, source_runs)
        captured = _capture(
            root,
            mask_name=mask_name,
            pose_name=pose_name,
            revision=expected_mask_revision,
            upgrade_target_rows=upgrade_target_rows,
            successor_format=successor_format,
        )
        if _source_file_fingerprint(archive, source_runs) != source_fingerprint:
            raise ValueError("Mask or keypoint source changed during tail refresh")
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
            derivation_method,
            preserved_seed,
            preserved_pose,
            preserved_seed_reasons,
            selected_rows,
            row_method_codes,
        ) = captured
        proof = {**proof, "apply_id": str(apply_id)}
        # One durable Apply cannot silently change its source after a partial
        # publication or a later failed secondary effect.
        if any(prior != proof for prior in prior_proofs):
            raise ValueError(
                "Source changed since this Apply's tail version was published"
            )
        binding = {**original_binding, "mask_apply_refresh": proof}
        version = "mask_apply_" + sha256_payload(proof)[:24]
        native = (
            root[f"refined_keypoints_runs/{pose_name}"].attrs["schema_id"]
            == NATIVE_REVIEW_SCHEMA
        )
        paths = run_paths(version, native=native, reference_crop=reference_crop)
        crop_name = proof["source_crop_run"].split("/")[-1]
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
                # Called before each child publication: the stat fingerprint
                # detects any write to the captured source runs cheaply.
                if _source_file_fingerprint(archive, source_runs) != source_fingerprint:
                    raise ValueError(
                        "Mask or keypoint source changed during tail refresh"
                    )

            def full_source_check():
                current = zarr.open_group(
                    str(archive), mode="r", use_consolidated=False
                )
                current_proof = _capture(
                    current,
                    mask_name=mask_name,
                    pose_name=pose_name,
                    revision=expected_mask_revision,
                    upgrade_target_rows=upgrade_target_rows,
                    successor_format=successor_format,
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
                    derivation_method=derivation_method,
                    row_method_codes=row_method_codes,
                    reference_crop_run=crop_name if reference_crop else None,
                )
                _carry_labels(
                    local,
                    result,
                    points=points,
                    manual=manual,
                    origins=origins,
                    proof=proof,
                    original_reasons=original_reasons,
                    preserved_seed=preserved_seed,
                    preserved_pose=preserved_pose,
                    preserved_seed_reasons=preserved_seed_reasons,
                    selected_rows=selected_rows,
                    archive=archive,
                )
                if reference_crop:
                    # Single writer: each complete local run is rewritten into
                    # its planned shards before anything is published.
                    for key, path in paths.items():
                        shard_training_review_run(
                            local / path, mutable=key in _EDITABLE_RUN_KEYS
                        )
                provenance = build_writer_run_provenance(
                    command="fisheye.training.mask_tail_apply_refresh",
                    params={
                        "policy": proof["policy"],
                        "version": version,
                        "source": proof,
                    },
                    input_run_ids={"masks": mask_name, "keypoints": pose_name},
                )
                for path in paths.values():
                    target_root[path].attrs["run_provenance"] = provenance
                source_check()
                full_source_check()
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
            "schema_id": proof["schema_id"],
            "version": version,
            "paths": paths,
            "source_crop_run": (
                crop_name if reference_crop else paths["crop"].split("/")[1]
            ),
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
        task_result = result
        if selected_rows is not None:
            result["upgrade_target_rows"] = selected_rows
            task_result = {
                **result,
                "failures": [
                    failure for failure in failures
                    if failure["roi_idx"] in selected_rows
                ],
            }
        result["tasks"] = review_tasks(
            archive, binding["recording_id"], task_result, version
        )
        if selected_rows is not None:
            for task in result["tasks"]:
                if task["workflow_kind"] == "keypoints":
                    task["scope"]["target_roi_indices"] = selected_rows
        if native:
            for task in result["tasks"]:
                task["dataset_id"] = (
                    f"{binding['recording_id']}:native_mask_tail:{version}"
                )
        return result


def upgrade_training_tail_geometry_version(
    *, archive, refined_mask_run, refined_keypoint_run, upgrade_id,
    expected_mask_revision, target_rows, scratch_root=Path("/tmp"),
):
    """Publish an explicit, selector-ineligible v4 successor for failed rows."""
    return regenerate_training_tail_version(
        archive=archive,
        refined_mask_run=refined_mask_run,
        refined_keypoint_run=refined_keypoint_run,
        apply_id=upgrade_id,
        expected_mask_revision=expected_mask_revision,
        scratch_root=scratch_root,
        upgrade_target_rows=target_rows,
    )
