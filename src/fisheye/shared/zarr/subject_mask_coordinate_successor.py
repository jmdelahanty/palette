"""Publish raw/refined subject-mask coordinate successors without inference."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import zarr

from fisheye.shared import __version__ as FISHEYE_SHARED_VERSION
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.model_input_transform import model_input_transform_from_attrs
from fisheye.shared.refined_subject_mask_coordinate_publication import (
    REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
    prepare_refined_subject_mask_coordinate_context,
    publish_refined_subject_mask_coordinate_surfaces,
)
from fisheye.shared.subject_mask_coordinate_publication import (
    SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
    prepare_subject_mask_coordinate_context,
    publish_subject_mask_coordinate_surfaces,
)
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.coordinate_successor_authority import (
    REFINED_SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
    SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
    build_coordinate_successor_authority,
    load_coordinate_successor_authority,
    stamp_coordinate_successor_authority,
)
from fisheye.shared.zarr.coordinate_successor_files import (
    copy_metadata_and_link_payload,
    metadata_tree_sha256,
    validate_payload_file_equivalence,
)
from fisheye.shared.zarr.sealed_geometry_crop_profile import (
    bind_persisted_padded_placement_record,
    bind_persisted_run_attribute_record,
    bind_sealed_geometry_crop_successor_source,
    stamp_persisted_padded_placement_provenance,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_sha256,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_FAMILY,
    SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE,
    SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR,
    validate_subject_mask_bundle_manifest,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    build_subject_mask_coordinate_successor_manifest,
    subject_mask_core_metadata_declaration_maps,
    validate_persisted_subject_mask_core_publication,
    validate_subject_mask_core_run_manifest,
)
from fisheye.shared.zarr.subject_mask_coordinate_validation_receipt import (
    RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
    REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
    SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
    build_subject_mask_coordinate_validation_receipt,
    stamp_subject_mask_coordinate_validation_receipt,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)


SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID = (
    "palette.subject_mask_coordinate_successor.production_publication"
)
SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION = 2
SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_POLICY = (
    "sealed_bundle_members_hardlink_payload_canonical_v2_metadata_pair_v2"
)
SELECTOR_ACTIVATION_DEFERRED = "deferred_separate_reviewed_change"

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTORS = ("latest", "latest_complete", "latest_pending", "authoritative_run")
_RAW_FAMILY = "subject_mask_runs"
_REFINED_FAMILY = "refined_subject_masks_runs"
_HISTORICAL_SEMANTIC_NORMALIZATION_ATTR = (
    "coordinate_successor_historical_semantic_normalization"
)
_HISTORICAL_REFINED_SEMANTIC_NORMALIZATION_ATTR = (
    "coordinate_successor_historical_refined_semantic_normalization"
)
_RAW_COORDINATE_VALIDATION_RECORD_NAMES = (
    "context",
    "derivation",
    "padded_crop_lineage",
    "row_identity",
    "surface_inventory",
    "temporal_authority",
)
_REFINED_COORDINATE_VALIDATION_RECORD_NAMES = (
    "component_qc_inventory",
    "context",
    "measurement_authority",
    "refinement_authority",
    "row_identity",
    "scientific_manifest",
    "source_authority",
    "surface_inventory",
    "temporal_authority",
)
_COORDINATE_VALIDATOR_IDENTITY = {
    "package": "fisheye.shared",
    "version": FISHEYE_SHARED_VERSION,
}


def _require_run_id(value: object, *, label: str) -> str:
    result = str(value or "").strip()
    if not _RUN_ID.fullmatch(result):
        raise ValueError(f"{label} must be one safe nonempty run ID.")
    return result


def _selector_snapshot(root: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in (_RAW_FAMILY, _REFINED_FAMILY, SUBJECT_MASK_BUNDLE_FAMILY):
        parent = root[family]
        result[family] = {
            name: {"present": name in parent.attrs, "value": parent.attrs.get(name)}
            for name in _SELECTORS
        }
    authority = root.attrs.get("subject_mask_authority")
    result["root_subject_mask_authority_digest"] = (
        canonical_json_sha256(authority) if isinstance(authority, Mapping) else None
    )
    return result


def _manifest_member(
    *, role: str, family: str, run_id: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    logical = manifest.get("payload", {}).get("logical_content", {})
    return {
        "role": role,
        "family": family,
        "run_id": run_id,
        "run_path": f"{family}/{run_id}",
        "manifest_schema_id": manifest.get("schema_id"),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_payload_digest": manifest.get("payload_digest"),
        "manifest_document_digest": canonical_json_sha256(manifest),
        "logical_content_digest": logical.get("digest"),
    }


def _coordinate_validation_source_bindings(
    manifest: Mapping[str, Any],
    *,
    source_run_path: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):  # guarded by manifest validation
        raise ValueError("Subject-mask source manifest payload is unavailable.")
    logical = payload.get("logical_content")
    source = payload.get("source")
    if not isinstance(logical, Mapping) or not isinstance(source, Mapping):
        raise ValueError("Subject-mask source manifest bindings are unavailable.")
    validation = source.get("validation_receipt")
    if validation is None:
        return None
    if not isinstance(validation, Mapping):
        raise ValueError("Subject-mask source-validation binding is invalid.")
    return (
        {
            "run_path": source_run_path,
            "core_manifest_payload_digest": manifest["payload_digest"],
            "core_manifest_document_digest": canonical_json_sha256(manifest),
            "logical_content_digest": logical["digest"],
        },
        {
            "schema_id": validation["schema_id"],
            "schema_version": validation["schema_version"],
            "payload_digest": validation["payload_digest"],
            "document_sha256": validation["document_sha256"],
            "semantic_unit_count": validation["semantic_unit_count"],
        },
    )


def _stamp_coordinate_validation_receipt(
    run: Any,
    *,
    kind: str,
    successor_run_path: str,
    source_manifest: Mapping[str, Any],
    source_run_path: str,
    bundle_authority_kind: str,
    bundle_manifest: Mapping[str, Any],
    coordinate_records: Mapping[str, Mapping[str, str]],
    coordinate_record_names: tuple[str, ...],
    payload_equivalence: Mapping[str, Any],
) -> Any | None:
    source_bindings = _coordinate_validation_source_bindings(
        source_manifest,
        source_run_path=source_run_path,
    )
    if source_bindings is None:
        return None
    source, source_validation = source_bindings
    receipt = build_subject_mask_coordinate_validation_receipt(
        kind=kind,
        successor_run_path=successor_run_path,
        source=source,
        source_validation=source_validation,
        bundle_authority={
            "kind": bundle_authority_kind,
            "document_digest": canonical_json_sha256(bundle_manifest),
        },
        coordinate_records=coordinate_records,
        coordinate_record_names=coordinate_record_names,
        payload_equivalence={
            "schema_id": payload_equivalence["schema_id"],
            "schema_version": payload_equivalence["schema_version"],
            "receipt_digest": payload_equivalence["receipt_digest"],
            "inventory_digest": payload_equivalence["inventory_digest"],
            "payload_file_count": payload_equivalence["payload_file_count"],
        },
        validator_identity=_COORDINATE_VALIDATOR_IDENTITY,
    )
    stamp_subject_mask_coordinate_validation_receipt(
        run,
        receipt,
        expected_kind=kind,
        expected_successor_run_path=successor_run_path,
        expected_coordinate_record_names=coordinate_record_names,
    )
    return bind_persisted_run_attribute_record(
        run,
        attr_name=SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
    )


def _bundle_authority(
    archive: Path,
    root: Any,
    *,
    bundle_run_id: str,
    raw_run_id: str,
    raw_manifest: Mapping[str, Any],
    refined_run_id: str,
    refined_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_path = f"{SUBJECT_MASK_BUNDLE_FAMILY}/{bundle_run_id}"
    validate_direct_consolidated_subtree(archive, subtree_path=bundle_path)
    bundle = root[bundle_path]
    if (
        bundle.attrs.get("status") != RUN_STATUS_COMPLETE
        or bundle.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or bundle.attrs.get("stage_selector_eligible") is not False
        or bundle.attrs.get(SUBJECT_MASK_BUNDLE_SELECTOR_ELIGIBLE_ATTR) is not False
    ):
        raise ValueError("Subject-mask source bundle is not complete and inactive.")
    manifest = bundle.attrs.get(SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise ValueError("Subject-mask source bundle manifest is absent.")
    errors = validate_subject_mask_bundle_manifest(manifest)
    if errors:
        raise ValueError("Subject-mask source bundle is invalid: " + "; ".join(errors))
    if manifest.get("schema_version") != 3:
        raise ValueError("Coordinate successor requires the reviewed bundle-v3 source.")
    payload = manifest["payload"]
    if payload.get("bundle_id") != bundle_run_id:
        raise ValueError("Subject-mask bundle ID differs from its exact path.")
    publication = payload.get("publication")
    if (
        not isinstance(publication, Mapping)
        or publication.get("activation_state") != "deferred"
    ):
        raise ValueError("Subject-mask source bundle is not activation-deferred.")
    members = payload.get("members")
    if not isinstance(members, Mapping):
        raise ValueError("Subject-mask source bundle members are absent.")
    expected_raw = _manifest_member(
        role="raw",
        family=_RAW_FAMILY,
        run_id=raw_run_id,
        manifest=raw_manifest,
    )
    expected_refined = _manifest_member(
        role="refined",
        family=_REFINED_FAMILY,
        run_id=refined_run_id,
        manifest=refined_manifest,
    )
    if members.get("raw") != expected_raw or members.get("refined") != expected_refined:
        raise ValueError(
            "Subject-mask bundle does not bind the exact raw/refined source pair."
        )
    return copy.deepcopy(dict(manifest))


def _source_manifest(run: Any, *, run_id: str, kind: str) -> dict[str, Any]:
    manifest = run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Subject-mask source {run_id!r} lacks a run manifest.")
    errors = validate_subject_mask_core_run_manifest(manifest)
    if errors:
        raise ValueError(
            f"Subject-mask source {run_id!r} is invalid: " + "; ".join(errors)
        )
    payload = manifest["payload"]
    if payload.get("run_id") != run_id or payload.get("kind") != kind:
        raise ValueError("Subject-mask source manifest binds another run or kind.")
    if (
        run.attrs.get("status") != RUN_STATUS_COMPLETE
        or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or run.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Subject-mask source is not complete and selector-ineligible.")
    return copy.deepcopy(dict(manifest))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_artifact(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Subject-mask model artifact not found: {resolved}")
    if _SHA256.fullmatch(str(expected_sha256 or "")) is None:
        raise ValueError("Subject-mask source lacks one exact model SHA-256.")
    actual = _sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(
            "Subject-mask model bytes differ from source inference authority."
        )
    stat = resolved.stat()
    return {
        "role": "subject_mask_unet_checkpoint",
        "path": str(resolved),
        "fingerprint_scheme": "content_v1",
        "sha256": actual,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "source": "direct_scientific_commit_rehash",
    }


def _labels(manifest: Mapping[str, Any]) -> list[str]:
    labels = manifest["payload"]["logical_schema"]["components"]["labels"]
    if not isinstance(labels, list) or not labels:
        raise ValueError("Subject-mask core lacks ordered component labels.")
    return [str(item) for item in labels]


def _raw_semantic_normalization(
    run: Any,
    *,
    manifest: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Derive modern run attrs from exact historical schema and array evidence."""

    logical_schema = manifest["payload"]["logical_schema"]
    expected_schema_semantics = {
        "output_semantics": "multilabel",
        "overlap_policy": "independent_sigmoid",
        "probability_semantics": "sigmoid_multilabel_logits",
    }
    for name, expected in expected_schema_semantics.items():
        if logical_schema.get(name) != expected:
            raise ValueError(
                f"Historical subject-mask logical schema {name!r} does not "
                f"support canonical successor semantics {expected!r}."
            )

    schema_threshold = logical_schema.get("threshold")
    if type(schema_threshold) is not float or schema_threshold != threshold:
        raise ValueError(
            "Historical subject-mask logical-schema threshold differs from its "
            "exact inference authority."
        )
    if "mask_probs_roi" not in run:
        raise ValueError("Historical subject-mask run lacks mask_probs_roi.")
    probability_array = run["mask_probs_roi"]
    probability_dtype = np.dtype(probability_array.dtype)
    if probability_dtype == np.dtype("uint8"):
        probability_dtype_name = "uint8"
        probability_encoding = "linear_uint8_0_255"
    elif probability_dtype == np.dtype("float16"):
        probability_dtype_name = "float16"
        probability_encoding = "unit_float"
    else:
        raise ValueError(
            "Historical subject-mask probabilities must use uint8 or float16 "
            "storage before semantic normalization."
        )
    if logical_schema.get("probability_encoding") != probability_encoding:
        raise ValueError(
            "Historical subject-mask logical-schema probability encoding differs "
            "from the exact persisted mask_probs_roi dtype."
        )

    labels = _labels(manifest)
    shape = tuple(int(value) for value in probability_array.shape)
    if len(shape) != 4 or shape[1] != len(labels):
        raise ValueError(
            "Historical subject-mask probability channels differ from the exact "
            "ordered component registry."
        )
    masks_materialized = "masks_roi" in run
    if masks_materialized:
        binary = run["masks_roi"]
        if tuple(int(value) for value in binary.shape) != shape:
            raise ValueError(
                "Historical subject-mask binary cache shape differs from "
                "mask_probs_roi."
            )
        if np.dtype(binary.dtype) != np.dtype("uint8"):
            raise ValueError(
                "Historical subject-mask binary cache must use uint8 storage."
            )

    resolved_attrs = {
        **expected_schema_semantics,
        "probabilities_dtype": probability_dtype_name,
        "probabilities_encoding": probability_encoding,
        "masks_roi_materialized": masks_materialized,
        "binary_masks_materialized": masks_materialized,
        "binary_masks_source": (
            f"threshold(mask_probs_roi, threshold={threshold})"
            if masks_materialized
            else "not_materialized"
        ),
        "bbox_xyxy_convention": "pixel_edge_half_open",
        "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
    }
    return json_attr_safe(
        {
            "schema_id": (
                "palette.subject_mask_coordinate_successor."
                "historical_semantic_normalization"
            ),
            "schema_version": 1,
            "policy": "logical_schema_and_persisted_arrays_exact_evidence_v1",
            "evidence": {
                "logical_schema_id": logical_schema.get("schema_id"),
                "logical_schema_version": logical_schema.get("schema_version"),
                "logical_schema_semantics": {
                    **expected_schema_semantics,
                    "probability_encoding": probability_encoding,
                    "threshold": threshold,
                },
                "mask_probs_roi": {
                    "dtype": probability_dtype_name,
                    "shape": list(shape),
                },
                "masks_roi_physically_present": masks_materialized,
                "ordered_mask_labels": labels,
            },
            "resolved_run_attrs": resolved_attrs,
        }
    )


def _refined_semantic_normalization(
    evidence: Any,
    *,
    source_manifest: Mapping[str, Any],
    evidence_run_path: str,
) -> dict[str, Any]:
    """Bind omitted refined semantics to the core's exact validated source draft."""

    payload = source_manifest.get("payload")
    source = payload.get("source") if isinstance(payload, Mapping) else None
    if not isinstance(source, Mapping) or source.get("run_path") != evidence_run_path:
        raise ValueError(
            "Refined core manifest does not bind the supplied worker-draft evidence."
        )
    validation = source.get("validation_receipt")
    if not isinstance(validation, Mapping):
        raise ValueError(
            "Refined core manifest lacks its worker-draft validation receipt."
        )
    if (
        evidence.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or evidence.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            "Refined worker-draft cache evidence is not complete and selector-ineligible."
        )
    resolved_attrs = {
        "derived_mask_caches_stale": False,
        "metrics_stale": False,
        "contours_stale": False,
        "masks_roi_materialized": True,
        "mask_storage_authority": "masks_roi",
        "editable_mask_surface": "masks_roi",
        "mask_bitpacked_materialized": False,
        "mask_rle_materialized": False,
        "bbox_xyxy_convention": "pixel_edge_half_open",
        "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
    }
    for name, expected in resolved_attrs.items():
        actual = evidence.attrs.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Refined worker-draft semantic evidence {name!r} differs from "
                f"the required exact value {expected!r}."
            )
    return json_attr_safe(
        {
            "schema_id": (
                "palette.subject_mask_coordinate_successor."
                "historical_refined_semantic_normalization"
            ),
            "schema_version": 2,
            "policy": (
                "core_manifest_bound_validated_worker_draft_refined_semantics_v2"
            ),
            "evidence": {
                "source_refined_manifest_payload_digest": source_manifest[
                    "payload_digest"
                ],
                "source_refined_manifest_document_digest": canonical_json_sha256(
                    source_manifest
                ),
                "worker_draft_run_path": evidence_run_path,
                "worker_draft_validation_receipt": copy.deepcopy(dict(validation)),
                "worker_draft_refined_semantics": resolved_attrs,
            },
            "resolved_run_attrs": resolved_attrs,
        }
    )


def _raw_inference_inputs(
    root: Any, raw_manifest: Mapping[str, Any], model_path: Path
) -> dict[str, Any]:
    payload = raw_manifest["payload"]
    source_path = str(payload["source"]["run_path"])
    producer = root[source_path]
    transform_attrs = producer.attrs.get("model_input_transform")
    transform = model_input_transform_from_attrs(dict(transform_attrs))
    prior_artifact = producer.attrs.get("subject_mask_model_artifact")
    if not isinstance(prior_artifact, Mapping):
        provenance = producer.attrs.get("run_provenance")
        artifacts = (
            provenance.get("input_artifacts")
            if isinstance(provenance, Mapping)
            else None
        )
        matches = [
            item
            for item in artifacts or ()
            if isinstance(item, Mapping)
            and item.get("role") == "subject_mask_unet_checkpoint"
        ]
        if len(matches) != 1:
            raise ValueError(
                "Raw subject-mask producer lacks one exact model artifact authority."
            )
        prior_artifact = matches[0]
    artifact = _model_artifact(
        model_path, expected_sha256=str(prior_artifact.get("sha256") or "")
    )
    threshold = producer.attrs.get("mask_probability_threshold")
    if type(threshold) is not float:
        threshold = payload["logical_schema"].get("threshold")
    if type(threshold) is not float:
        raise ValueError(
            "Raw subject-mask source lacks an exact probability threshold."
        )
    provenance = producer.attrs.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Raw subject-mask producer lacks exact stage provenance.")
    return {
        "producer_run_path": source_path,
        "model_input_transform": transform,
        "model_artifact": artifact,
        "mask_probability_threshold": threshold,
        "provenance": copy.deepcopy(dict(provenance)),
    }


def inspect_subject_mask_coordinate_successor_source(
    *,
    analysis_zarr: Path,
    source_raw_run_id: str,
    source_refined_run_id: str,
    source_bundle_run_id: str,
    refined_evidence_run_path: str,
    raw_successor_run_id: str,
    refined_successor_run_id: str,
    subject_mask_model_path: Path,
) -> dict[str, Any]:
    """Validate the exact immutable pair and its retained refinement evidence."""

    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    raw_id = _require_run_id(source_raw_run_id, label="source_raw_run_id")
    refined_id = _require_run_id(source_refined_run_id, label="source_refined_run_id")
    bundle_id = _require_run_id(source_bundle_run_id, label="source_bundle_run_id")
    raw_target_id = _require_run_id(raw_successor_run_id, label="raw_successor_run_id")
    refined_target_id = _require_run_id(
        refined_successor_run_id, label="refined_successor_run_id"
    )
    if raw_id == raw_target_id or refined_id == refined_target_id:
        raise ValueError("Subject-mask successors cannot replace their source runs.")
    evidence_parts = str(refined_evidence_run_path).strip("/").split("/")
    if len(evidence_parts) != 2 or not evidence_parts[0].endswith("_runs"):
        raise ValueError(
            "refined_evidence_run_path must name one exact runs-family child."
        )
    paths = {
        "raw": archive / _RAW_FAMILY / raw_id,
        "refined": archive / _REFINED_FAMILY / refined_id,
        "evidence": archive / refined_evidence_run_path,
        "raw_target": archive / _RAW_FAMILY / raw_target_id,
        "refined_target": archive / _REFINED_FAMILY / refined_target_id,
    }
    for name in ("raw", "refined", "evidence"):
        if not paths[name].is_dir():
            raise FileNotFoundError(
                f"Subject-mask {name} source not found: {paths[name]}"
            )
    for name in ("raw_target", "refined_target"):
        if paths[name].exists():
            raise FileExistsError(
                f"Immutable subject-mask target exists: {paths[name]}"
            )

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    raw_manifest = _source_manifest(
        root[f"{_RAW_FAMILY}/{raw_id}"], run_id=raw_id, kind="raw_probability_uint8"
    )
    refined_manifest = _source_manifest(
        root[f"{_REFINED_FAMILY}/{refined_id}"],
        run_id=refined_id,
        kind="refined_dense_core",
    )
    for family, run_id in (
        (_RAW_FAMILY, raw_id),
        (_REFINED_FAMILY, refined_id),
    ):
        source_errors = validate_persisted_subject_mask_core_publication(
            archive,
            family=family,
            run_id=run_id,
        )
        if source_errors:
            raise ValueError(
                f"Subject-mask source {family}/{run_id} is invalid: "
                + "; ".join(source_errors)
            )
    bundle_manifest = _bundle_authority(
        archive,
        root,
        bundle_run_id=bundle_id,
        raw_run_id=raw_id,
        raw_manifest=raw_manifest,
        refined_run_id=refined_id,
        refined_manifest=refined_manifest,
    )
    evidence = root[str(refined_evidence_run_path).strip("/")]
    if "components" not in evidence:
        raise ValueError("Refined evidence run lacks component provenance groups.")
    refined_semantic_normalization = _refined_semantic_normalization(
        evidence,
        source_manifest=refined_manifest,
        evidence_run_path=str(refined_evidence_run_path).strip("/"),
    )
    inference = _raw_inference_inputs(root, raw_manifest, subject_mask_model_path)
    raw_semantic_normalization = _raw_semantic_normalization(
        root[f"{_RAW_FAMILY}/{raw_id}"],
        manifest=raw_manifest,
        threshold=inference["mask_probability_threshold"],
    )
    crop_reference = raw_manifest["payload"]["coordinate_dependencies"]["document"][
        "crop"
    ]
    sealed_crop = bind_sealed_geometry_crop_successor_source(
        analysis_zarr=archive,
        root=root,
        crop_reference=crop_reference,
        source_manifest=raw_manifest,
        source_arrays={
            "source_crop_row_ids": root[f"{_RAW_FAMILY}/{raw_id}/source_crop_row_ids"],
            "instance_key": root[f"{_RAW_FAMILY}/{raw_id}/instance_key"],
            "source_acquisition_frame_index": root[
                f"{_RAW_FAMILY}/{raw_id}/source_acquisition_frame_index"
            ],
            "source_crop_xywh": root[f"{_RAW_FAMILY}/{raw_id}/source_crop_xywh"],
        },
        source_run_path=f"{_RAW_FAMILY}/{raw_id}",
        model_input_transform=inference["model_input_transform"],
    )
    return json_attr_safe(
        {
            "schema_id": "palette.subject_mask_coordinate_successor.source_inspection",
            "schema_version": 1,
            "status": "ready",
            "analysis_zarr": str(archive),
            "source_raw_run_id": raw_id,
            "source_refined_run_id": refined_id,
            "source_bundle_run_id": bundle_id,
            "refined_evidence_run_path": str(refined_evidence_run_path).strip("/"),
            "raw_successor_run_id": raw_target_id,
            "refined_successor_run_id": refined_target_id,
            "source_raw_manifest_digest": canonical_json_sha256(raw_manifest),
            "source_refined_manifest_digest": canonical_json_sha256(refined_manifest),
            "source_bundle_manifest_digest": canonical_json_sha256(bundle_manifest),
            "source_raw_metadata_tree_sha256": metadata_tree_sha256(paths["raw"]),
            "source_refined_metadata_tree_sha256": metadata_tree_sha256(
                paths["refined"]
            ),
            "source_evidence_metadata_tree_sha256": metadata_tree_sha256(
                paths["evidence"]
            ),
            "producer_run_path": inference["producer_run_path"],
            "raw_semantic_normalization": raw_semantic_normalization,
            "refined_semantic_normalization": (
                refined_semantic_normalization
            ),
            "historical_crop_adapter": sealed_crop.as_record(),
            "model_artifact": inference["model_artifact"],
            "selectors_before": _selector_snapshot(root),
            "selector_eligible": False,
            "selector_activation": SELECTOR_ACTIVATION_DEFERRED,
            "registry_updated": False,
        }
    )


def _clear_successor_attrs(run: Any, *, source_run_path: str, owner_attr: str) -> str:
    attrs = dict(run.attrs)
    for name in (
        SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
        "coordinate_successor_authority",
        "coordinate_successor_authority_sha256",
        SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
        f"{SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE}_sha256",
        _HISTORICAL_SEMANTIC_NORMALIZATION_ATTR,
        f"{_HISTORICAL_SEMANTIC_NORMALIZATION_ATTR}_sha256",
        _HISTORICAL_REFINED_SEMANTIC_NORMALIZATION_ATTR,
        f"{_HISTORICAL_REFINED_SEMANTIC_NORMALIZATION_ATTR}_sha256",
        RUN_COMPLETED_AT_ATTR,
        "palette_run_failed_at_utc",
        "palette_run_error",
    ):
        attrs.pop(name, None)
    owner = uuid4().hex
    attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "production_candidate": True,
            "production_selector_activation": SELECTOR_ACTIVATION_DEFERRED,
            "coordinate_contract": "coordinate_successor_preparing",
            "coordinate_successor_policy": SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_POLICY,
            "coordinate_successor_source_run_path": source_run_path,
            ATOMIC_PUBLICATION_OWNER_ATTR: owner,
            owner_attr: owner,
        }
    )
    run.attrs.put(attrs)
    return owner


def _record_pointers(values: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, value in values.items():
        if isinstance(value, Mapping):
            record_ref = value.get("record_ref")
            record_sha256 = value.get("record_sha256")
        else:
            record_ref = getattr(value, "record_ref", None)
            record_sha256 = getattr(value, "record_sha256", None)
        if not isinstance(record_ref, str) or not isinstance(record_sha256, str):
            raise ValueError(
                f"Subject-mask coordinate record {name!r} lacks one exact pointer."
            )
        result[name] = {
            "record_ref": record_ref,
            "record_sha256": record_sha256,
        }
    return result


def _rebase_component_provenance(
    run: Any,
    *,
    labels: list[str],
    raw_successor_path: str,
) -> None:
    for label in labels:
        provenance = run[f"components/{label}/provenance"]
        attrs = dict(provenance.attrs)
        surface = str(attrs.get("source_surface_path") or "")
        surface_name = surface.rsplit("/", 1)[-1]
        if surface_name not in {"mask_probs_roi", "masks_roi"}:
            raise ValueError(
                f"Refined component {label!r} has an invalid source surface."
            )
        rebased = f"{raw_successor_path}/{surface_name}"
        attrs["source_surface_path"] = rebased
        if "source_probability_path" in attrs:
            attrs["source_probability_path"] = rebased
        provenance.attrs.put(attrs)


def _refined_dependencies(
    source_manifest: Mapping[str, Any],
    raw_successor_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    dependencies = copy.deepcopy(source_manifest["payload"]["coordinate_dependencies"])
    document = dependencies["document"]
    document["raw_core"] = {
        "run_id": raw_successor_manifest["payload"]["run_id"],
        "manifest_payload_digest": raw_successor_manifest["payload_digest"],
        "coordinate_catalog_digest": raw_successor_manifest["payload"][
            "coordinate_contract"
        ]["digest"],
    }
    dependencies["digest"] = canonical_json_sha256(document)
    return dependencies


def _publish_refined_with_sealed_crop(
    root: Any,
    *,
    refined_run_path: str,
    refined_owner: str,
    raw_run_path: str,
    mask_labels: list[str],
    sealed_crop: Any,
) -> Any:
    """Prepare and publish refined surfaces through the shared crop resolver."""

    stamp_persisted_padded_placement_provenance(
        root[refined_run_path],
        sealed_crop,
    )
    prepare_refined_subject_mask_coordinate_context(
        root,
        refined_run_path,
        expected_publication_owner=refined_owner,
        source_subject_mask_path=raw_run_path,
        mask_labels=mask_labels,
        assignment_keypoint_surfaces=None,
        source_selector_eligible=False,
    )
    return publish_refined_subject_mask_coordinate_surfaces(
        root,
        refined_run_path,
        expected_publication_owner=refined_owner,
    )


def publish_subject_mask_coordinate_successors(
    *,
    analysis_zarr: Path,
    source_raw_run_id: str,
    source_refined_run_id: str,
    source_bundle_run_id: str,
    refined_evidence_run_path: str,
    raw_successor_run_id: str,
    refined_successor_run_id: str,
    subject_mask_model_path: Path,
) -> dict[str, Any]:
    """Publish one ordered raw/refined selector-ineligible successor pair."""

    initial = inspect_subject_mask_coordinate_successor_source(
        analysis_zarr=analysis_zarr,
        source_raw_run_id=source_raw_run_id,
        source_refined_run_id=source_refined_run_id,
        source_bundle_run_id=source_bundle_run_id,
        refined_evidence_run_path=refined_evidence_run_path,
        raw_successor_run_id=raw_successor_run_id,
        refined_successor_run_id=refined_successor_run_id,
        subject_mask_model_path=subject_mask_model_path,
    )
    archive = Path(initial["analysis_zarr"])
    raw_id = str(initial["source_raw_run_id"])
    refined_id = str(initial["source_refined_run_id"])
    bundle_id = str(initial["source_bundle_run_id"])
    raw_target_id = str(initial["raw_successor_run_id"])
    refined_target_id = str(initial["refined_successor_run_id"])
    evidence_path = str(initial["refined_evidence_run_path"])
    raw_source_fs = archive / _RAW_FAMILY / raw_id
    refined_source_fs = archive / _REFINED_FAMILY / refined_id
    evidence_fs = archive / evidence_path
    raw_target_fs = archive / _RAW_FAMILY / raw_target_id
    refined_target_fs = archive / _REFINED_FAMILY / refined_target_id
    receipts: dict[str, Any] = {}

    with archive_metadata_publication_lock(archive):
        checked = inspect_subject_mask_coordinate_successor_source(
            analysis_zarr=archive,
            source_raw_run_id=raw_id,
            source_refined_run_id=refined_id,
            source_bundle_run_id=bundle_id,
            refined_evidence_run_path=evidence_path,
            raw_successor_run_id=raw_target_id,
            refined_successor_run_id=refined_target_id,
            subject_mask_model_path=subject_mask_model_path,
        )
        if checked["selectors_before"] != initial["selectors_before"]:
            raise RuntimeError("Subject-mask selectors changed after planning.")
        root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
        raw_source = root[f"{_RAW_FAMILY}/{raw_id}"]
        refined_source = root[f"{_REFINED_FAMILY}/{refined_id}"]
        raw_manifest = _source_manifest(
            raw_source, run_id=raw_id, kind="raw_probability_uint8"
        )
        refined_manifest = _source_manifest(
            refined_source, run_id=refined_id, kind="refined_dense_core"
        )
        bundle_manifest = _bundle_authority(
            archive,
            root,
            bundle_run_id=bundle_id,
            raw_run_id=raw_id,
            raw_manifest=raw_manifest,
            refined_run_id=refined_id,
            refined_manifest=refined_manifest,
        )
        inference = _raw_inference_inputs(root, raw_manifest, subject_mask_model_path)
        raw_semantic_normalization = _raw_semantic_normalization(
            raw_source,
            manifest=raw_manifest,
            threshold=inference["mask_probability_threshold"],
        )
        if raw_semantic_normalization != checked["raw_semantic_normalization"]:
            raise RuntimeError(
                "Subject-mask semantic normalization changed between dry-run and apply."
            )
        evidence = root[evidence_path]
        refined_semantic_normalization = _refined_semantic_normalization(
            evidence,
            source_manifest=refined_manifest,
            evidence_run_path=evidence_path,
        )
        if (
            refined_semantic_normalization
            != checked["refined_semantic_normalization"]
        ):
            raise RuntimeError(
                "Refined semantic normalization changed between dry-run and apply."
            )
        crop_reference = raw_manifest["payload"]["coordinate_dependencies"]["document"][
            "crop"
        ]
        sealed_crop = bind_sealed_geometry_crop_successor_source(
            analysis_zarr=archive,
            root=root,
            crop_reference=crop_reference,
            source_manifest=raw_manifest,
            source_arrays={
                "source_crop_row_ids": raw_source["source_crop_row_ids"],
                "instance_key": raw_source["instance_key"],
                "source_acquisition_frame_index": raw_source[
                    "source_acquisition_frame_index"
                ],
                "source_crop_xywh": raw_source["source_crop_xywh"],
            },
            source_run_path=f"{_RAW_FAMILY}/{raw_id}",
            model_input_transform=inference["model_input_transform"],
        )
        try:
            receipts["raw_copy"] = copy_metadata_and_link_payload(
                raw_source_fs, raw_target_fs
            )
            receipts["raw_payload_equivalence"] = validate_payload_file_equivalence(
                raw_source_fs,
                raw_target_fs,
                source_label=f"{_RAW_FAMILY}/{raw_id}",
                target_label=f"{_RAW_FAMILY}/{raw_target_id}",
            )
            raw_run = root[f"{_RAW_FAMILY}/{raw_target_id}"]
            raw_owner = _clear_successor_attrs(
                raw_run,
                source_run_path=f"{_RAW_FAMILY}/{raw_id}",
                owner_attr=SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
            )
            raw_run.attrs.update(
                {
                    "model_input_transform": inference[
                        "model_input_transform"
                    ].to_attrs(),
                    "mask_probability_threshold": inference[
                        "mask_probability_threshold"
                    ],
                    "source_checkpoint": inference["model_artifact"]["path"],
                    "subject_mask_model_artifact": inference["model_artifact"],
                    "mask_labels": _labels(raw_manifest),
                    "provenance": inference["provenance"],
                    "coordinate_successor_historical_crop_adapter": sealed_crop.as_record(),
                    "coordinate_successor_historical_crop_adapter_sha256": canonical_json_sha256(
                        sealed_crop.as_record()
                    ),
                    **raw_semantic_normalization["resolved_run_attrs"],
                    _HISTORICAL_SEMANTIC_NORMALIZATION_ATTR: raw_semantic_normalization,
                    f"{_HISTORICAL_SEMANTIC_NORMALIZATION_ATTR}_sha256": canonical_json_sha256(
                        raw_semantic_normalization
                    ),
                }
            )
            stamp_persisted_padded_placement_provenance(raw_run, sealed_crop)
            mark_run_started(raw_run, run_name=raw_target_id, stage="subject_masks")
            crop_path = raw_manifest["payload"]["coordinate_dependencies"]["document"][
                "crop"
            ]["run_path"]
            prepare_subject_mask_coordinate_context(
                root,
                f"{_RAW_FAMILY}/{raw_target_id}",
                expected_publication_owner=raw_owner,
                crop_path=crop_path,
                mask_labels=_labels(raw_manifest),
                model_input_transform=inference["model_input_transform"],
                model_artifact=inference["model_artifact"],
                mask_probability_threshold=inference["mask_probability_threshold"],
                _resolved_crop_source=sealed_crop.source,
            )
            raw_surfaces = publish_subject_mask_coordinate_surfaces(
                root,
                f"{_RAW_FAMILY}/{raw_target_id}",
                expected_publication_owner=raw_owner,
                _resolved_crop_source=sealed_crop.source,
            )
            # The subject-mask publisher's bound context is the pre-stamping
            # context snapshot. Reopen the exact subgroup after publication so
            # an older Zarr metadata cache cannot restore pre-publication attrs
            # on a later update.
            root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
            raw_run = root[f"{_RAW_FAMILY}/{raw_target_id}"]
            raw_run.attrs["coordinate_successor_padded_crop_lineage"] = {
                "source_crop_adapter": sealed_crop.as_record(),
                "placement_ownership": bind_persisted_padded_placement_record(raw_run),
            }
            raw_run.attrs["coordinate_successor_padded_crop_lineage_sha256"] = (
                canonical_json_sha256(
                    raw_run.attrs["coordinate_successor_padded_crop_lineage"]
                )
            )
            padded_lineage_record = bind_persisted_run_attribute_record(
                raw_run, attr_name="coordinate_successor_padded_crop_lineage"
            )
            semantic_normalization_record = bind_persisted_run_attribute_record(
                raw_run, attr_name=_HISTORICAL_SEMANTIC_NORMALIZATION_ATTR
            )
            raw_coordinate_records = _record_pointers(
                {
                    "context": raw_surfaces.context.context_record,
                    "row_identity": raw_surfaces.context.row_identity,
                    "temporal_authority": raw_surfaces.context.temporal_authority,
                    "surface_inventory": raw_surfaces.inventory,
                    "derivation": raw_surfaces.derivation,
                    "padded_crop_lineage": padded_lineage_record,
                }
            )
            raw_validation_record = _stamp_coordinate_validation_receipt(
                raw_run,
                kind=RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
                successor_run_path=f"{_RAW_FAMILY}/{raw_target_id}",
                source_manifest=raw_manifest,
                source_run_path=f"{_RAW_FAMILY}/{raw_id}",
                bundle_authority_kind="inactive_subject_mask_bundle_v3",
                bundle_manifest=bundle_manifest,
                coordinate_records=raw_coordinate_records,
                coordinate_record_names=_RAW_COORDINATE_VALIDATION_RECORD_NAMES,
                payload_equivalence=receipts["raw_payload_equivalence"],
            )
            receipts["raw_coordinate_validation_receipt"] = (
                None
                if raw_validation_record is None
                else {
                    "record_ref": raw_validation_record["record_ref"],
                    "record_sha256": raw_validation_record["record_sha256"],
                }
            )
            raw_authority_records = dict(raw_coordinate_records)
            if raw_validation_record is not None:
                raw_authority_records["coordinate_validation_receipt"] = {
                    "record_ref": raw_validation_record["record_ref"],
                    "record_sha256": raw_validation_record["record_sha256"],
                }
            raw_authority = build_coordinate_successor_authority(
                kind=SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
                source_family=_RAW_FAMILY,
                source_run_path=f"{_RAW_FAMILY}/{raw_id}",
                source_manifest=raw_manifest,
                source_authority_kind="inactive_subject_mask_bundle_v3",
                source_authority=bundle_manifest,
                successor_family=_RAW_FAMILY,
                successor_run_path=f"{_RAW_FAMILY}/{raw_target_id}",
                payload_equivalence={
                    "policy": "same_filesystem_hardlink_payload_exact_logical_digest_v2",
                    "source_logical_content_digest": raw_manifest["payload"][
                        "logical_content"
                    ]["digest"],
                    **receipts["raw_copy"],
                    "payload_file_equivalence": receipts[
                        "raw_payload_equivalence"
                    ],
                    "padded_crop_lineage": padded_lineage_record,
                    "historical_semantic_normalization": semantic_normalization_record,
                },
                coordinate_records=raw_authority_records,
            )
            stamp_coordinate_successor_authority(raw_run, raw_authority)
            raw_run.attrs["status"] = RUN_STATUS_COMPLETE
            mark_run_complete(raw_run, run_name=raw_target_id)
            consolidate_metadata_capture_expected_warnings(archive)
            direct, consolidated = subject_mask_core_metadata_declaration_maps(
                archive,
                family=_RAW_FAMILY,
                run_id=raw_target_id,
                manifest=raw_manifest,
            )
            raw_successor_manifest = build_subject_mask_coordinate_successor_manifest(
                raw_manifest,
                run_id=raw_target_id,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
            )
            raw_run.attrs[SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE] = (
                raw_successor_manifest
            )
            consolidate_metadata_capture_expected_warnings(archive)
            raw_errors = validate_persisted_subject_mask_core_publication(
                archive, family=_RAW_FAMILY, run_id=raw_target_id
            )
            if raw_errors:
                raise RuntimeError(
                    "Raw mask successor validation failed: " + "; ".join(raw_errors)
                )
            # The publisher just performed the complete value-and-derived-
            # metric scan while this transaction holds the archive publication
            # lock. The manifest validation above and authority reload below
            # verify its persisted metadata. A complete-reader reload here
            # would repeat the same multi-gigabyte scientific scan without an
            # intervening mutation boundary.

            receipts["refined_copy"] = copy_metadata_and_link_payload(
                refined_source_fs, refined_target_fs
            )
            receipts["refined_payload_equivalence"] = (
                validate_payload_file_equivalence(
                    refined_source_fs,
                    refined_target_fs,
                    source_label=f"{_REFINED_FAMILY}/{refined_id}",
                    target_label=f"{_REFINED_FAMILY}/{refined_target_id}",
                )
            )
            refined_run = root[f"{_REFINED_FAMILY}/{refined_target_id}"]
            refined_owner = _clear_successor_attrs(
                refined_run,
                source_run_path=f"{_REFINED_FAMILY}/{refined_id}",
                owner_attr=REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
            )
            if "components" in refined_run:
                raise ValueError(
                    "Refined immutable core unexpectedly already has components."
                )
            receipts["component_evidence_copy"] = copy_metadata_and_link_payload(
                evidence_fs / "components", refined_target_fs / "components"
            )
            receipts["component_evidence_payload_equivalence"] = (
                validate_payload_file_equivalence(
                    evidence_fs / "components",
                    refined_target_fs / "components",
                    source_label=f"{evidence_path}/components",
                    target_label=f"{_REFINED_FAMILY}/{refined_target_id}/components",
                )
            )
            draft = root[evidence_path]
            required_attrs = (
                "provenance",
                "method",
                "refinement_semantics",
                "finalization_semantics",
                "label_schema_id",
                "mask_labels",
            )
            missing = [name for name in required_attrs if name not in draft.attrs]
            if missing:
                raise ValueError(
                    f"Refined evidence is missing required attrs {missing!r}."
                )
            refined_run.attrs.update(
                {
                    **{
                        name: copy.deepcopy(draft.attrs[name])
                        for name in required_attrs
                    },
                    **refined_semantic_normalization["resolved_run_attrs"],
                    _HISTORICAL_REFINED_SEMANTIC_NORMALIZATION_ATTR: (
                        refined_semantic_normalization
                    ),
                    f"{_HISTORICAL_REFINED_SEMANTIC_NORMALIZATION_ATTR}_sha256": (
                        canonical_json_sha256(refined_semantic_normalization)
                    ),
                }
            )
            refined_run.attrs["source_subject_mask_run"] = raw_target_id
            refined_run.attrs.pop("assignment_keypoint_coordinate_contract", None)
            _rebase_component_provenance(
                refined_run,
                labels=_labels(refined_manifest),
                raw_successor_path=f"{_RAW_FAMILY}/{raw_target_id}",
            )
            mark_run_started(
                refined_run, run_name=refined_target_id, stage="refined_subject_masks"
            )
            # Refined context preparation reloads the selected raw coordinate
            # metadata (without scanning its probability raster). Keep the
            # shared sealed-crop resolver for that reload and the publisher's
            # second context load. Refreshing the archive
            # root also prevents pre-lifecycle metadata cached by the earlier
            # copy from restoring source run attrs.
            root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
            refined_surfaces = _publish_refined_with_sealed_crop(
                root,
                refined_run_path=f"{_REFINED_FAMILY}/{refined_target_id}",
                refined_owner=refined_owner,
                raw_run_path=f"{_RAW_FAMILY}/{raw_target_id}",
                mask_labels=_labels(refined_manifest),
                sealed_crop=sealed_crop,
            )
            root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
            refined_run = root[f"{_REFINED_FAMILY}/{refined_target_id}"]
            refined_semantic_record = bind_persisted_run_attribute_record(
                refined_run,
                attr_name=_HISTORICAL_REFINED_SEMANTIC_NORMALIZATION_ATTR,
            )
            refined_coordinate_records = _record_pointers(
                {
                    "context": refined_surfaces.context.context_record,
                    "row_identity": refined_surfaces.context.row_identity,
                    "temporal_authority": refined_surfaces.context.temporal_authority,
                    "source_authority": refined_surfaces.context.source_authority,
                    "refinement_authority": (
                        refined_surfaces.context.refinement_authority
                    ),
                    "surface_inventory": refined_surfaces.inventory,
                    "component_qc_inventory": (
                        refined_surfaces.component_qc_inventory
                    ),
                    "measurement_authority": refined_surfaces.measurement_authority,
                    "scientific_manifest": refined_surfaces.scientific_manifest,
                }
            )
            refined_validation_record = _stamp_coordinate_validation_receipt(
                refined_run,
                kind=REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
                successor_run_path=f"{_REFINED_FAMILY}/{refined_target_id}",
                source_manifest=refined_manifest,
                source_run_path=f"{_REFINED_FAMILY}/{refined_id}",
                bundle_authority_kind="inactive_subject_mask_bundle_v3",
                bundle_manifest=bundle_manifest,
                coordinate_records=refined_coordinate_records,
                coordinate_record_names=_REFINED_COORDINATE_VALIDATION_RECORD_NAMES,
                payload_equivalence=receipts["refined_payload_equivalence"],
            )
            receipts["refined_coordinate_validation_receipt"] = (
                None
                if refined_validation_record is None
                else {
                    "record_ref": refined_validation_record["record_ref"],
                    "record_sha256": refined_validation_record["record_sha256"],
                }
            )
            refined_authority_records = dict(refined_coordinate_records)
            if refined_validation_record is not None:
                refined_authority_records["coordinate_validation_receipt"] = {
                    "record_ref": refined_validation_record["record_ref"],
                    "record_sha256": refined_validation_record["record_sha256"],
                }
            refined_authority = build_coordinate_successor_authority(
                kind=REFINED_SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
                source_family=_REFINED_FAMILY,
                source_run_path=f"{_REFINED_FAMILY}/{refined_id}",
                source_manifest=refined_manifest,
                source_authority_kind="inactive_subject_mask_bundle_v3_plus_raw_successor",
                source_authority={
                    "bundle_manifest": bundle_manifest,
                    "raw_successor_authority": raw_authority,
                },
                successor_family=_REFINED_FAMILY,
                successor_run_path=f"{_REFINED_FAMILY}/{refined_target_id}",
                payload_equivalence={
                    "policy": "same_filesystem_hardlink_payload_exact_logical_digest_v2",
                    "source_logical_content_digest": refined_manifest["payload"][
                        "logical_content"
                    ]["digest"],
                    **receipts["refined_copy"],
                    "payload_file_equivalence": receipts[
                        "refined_payload_equivalence"
                    ],
                    "component_evidence_source_run_path": evidence_path,
                    "component_evidence_payload_equivalence": receipts[
                        "component_evidence_payload_equivalence"
                    ],
                    "historical_refined_semantic_normalization": (
                        refined_semantic_record
                    ),
                },
                coordinate_records=refined_authority_records,
            )
            stamp_coordinate_successor_authority(refined_run, refined_authority)
            refined_run.attrs["status"] = RUN_STATUS_COMPLETE
            mark_run_complete(refined_run, run_name=refined_target_id)
            consolidate_metadata_capture_expected_warnings(archive)
            direct, consolidated = subject_mask_core_metadata_declaration_maps(
                archive,
                family=_REFINED_FAMILY,
                run_id=refined_target_id,
                manifest=refined_manifest,
            )
            refined_successor_manifest = (
                build_subject_mask_coordinate_successor_manifest(
                    refined_manifest,
                    run_id=refined_target_id,
                    direct_metadata_declarations=direct,
                    consolidated_metadata_declarations=consolidated,
                    coordinate_dependencies=_refined_dependencies(
                        refined_manifest, raw_successor_manifest
                    ),
                )
            )
            refined_run.attrs[SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE] = (
                refined_successor_manifest
            )
            consolidate_metadata_capture_expected_warnings(archive)

            final_root = open_zarr_root(archive, mode="r")
            if _selector_snapshot(final_root) != initial["selectors_before"]:
                raise RuntimeError("Subject-mask selectors changed during publication.")
            refined_errors = validate_persisted_subject_mask_core_publication(
                archive, family=_REFINED_FAMILY, run_id=refined_target_id
            )
            if refined_errors:
                raise RuntimeError(
                    "Refined mask successor validation failed: "
                    + "; ".join(refined_errors)
                )
            # Refined publication likewise completed its one full dense-mask
            # and derived-metric scan under this same lock. Persisted core
            # declarations and both successor authorities are verified below
            # without redundantly rescanning raw and refined rasters.
            load_coordinate_successor_authority(
                final_root[f"{_RAW_FAMILY}/{raw_target_id}"],
                expected_kind=SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
                expected_successor_run_path=f"{_RAW_FAMILY}/{raw_target_id}",
            )
            load_coordinate_successor_authority(
                final_root[f"{_REFINED_FAMILY}/{refined_target_id}"],
                expected_kind=REFINED_SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
                expected_successor_run_path=f"{_REFINED_FAMILY}/{refined_target_id}",
            )
            for path, expected in (
                (raw_source_fs, initial["source_raw_metadata_tree_sha256"]),
                (refined_source_fs, initial["source_refined_metadata_tree_sha256"]),
                (evidence_fs, initial["source_evidence_metadata_tree_sha256"]),
            ):
                if metadata_tree_sha256(path) != expected:
                    raise RuntimeError(f"Immutable source metadata changed at {path}.")
        except BaseException as exc:
            for family, run_id, path in (
                (_REFINED_FAMILY, refined_target_id, refined_target_fs),
                (_RAW_FAMILY, raw_target_id, raw_target_fs),
            ):
                if not path.exists():
                    continue
                try:
                    failed_root = zarr.open_group(
                        str(archive), mode="a", use_consolidated=False
                    )
                    failed = failed_root[f"{family}/{run_id}"]
                    failed.attrs["status"] = "failed"
                    failed.attrs["stage_selector_eligible"] = False
                    mark_run_failed(failed, run_name=run_id, error=str(exc))
                except BaseException:
                    pass
            try:
                consolidate_metadata_capture_expected_warnings(archive)
            except BaseException:
                pass
            raise

    return json_attr_safe(
        {
            "schema_id": SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "published_at_utc": utc_now(),
            "analysis_zarr": str(archive),
            "source_raw_run_path": f"{_RAW_FAMILY}/{raw_id}",
            "source_refined_run_path": f"{_REFINED_FAMILY}/{refined_id}",
            "raw_successor_run_path": f"{_RAW_FAMILY}/{raw_target_id}",
            "refined_successor_run_path": f"{_REFINED_FAMILY}/{refined_target_id}",
            "historical_crop_adapter": initial["historical_crop_adapter"],
            "copy": receipts,
            "coordinate_contract": "canonical_v2",
            "selector_eligible": False,
            "selectors_before": initial["selectors_before"],
            "selectors_after": initial["selectors_before"],
            "selector_activation": SELECTOR_ACTIVATION_DEFERRED,
            "registry_updated": False,
            "scientific_validation": (
                "one_full_raw_and_refined_surface_scan_each_under_single_locked_"
                "publication_transaction_v1"
            ),
        }
    )


__all__ = [
    "SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_POLICY",
    "SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID",
    "SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION",
    "inspect_subject_mask_coordinate_successor_source",
    "publish_subject_mask_coordinate_successors",
]
