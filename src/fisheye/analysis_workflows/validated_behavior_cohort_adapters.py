"""Source adapters for the generic validated-behavior cohort contracts.

Only this module understands the historical composable-chaser task or the
current recording-behavior bundle.  The normalized cohort and bundle-set
envelopes remain protocol-independent; another behavior family can implement
the same adapter outputs without adding another publication engine.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.analysis_workflows.exact_chaser_projection_receipt import (
    validate_exact_chaser_projection_receipt,
)
from fisheye.analysis_workflows.validated_behavior_cohort import (
    CAPABILITY_STATES,
    MEMBERSHIP_REASON_CODES,
    ValidatedBehaviorCohortError,
    build_capability_contract,
    build_validated_behavior_bundle_set,
    build_validated_behavior_cohort_membership,
    policy_envelope,
    validate_validated_behavior_bundle_set,
    validate_validated_behavior_cohort_membership,
)
from fisheye.analysis_workflows.validated_recording_behavior_bundle import (
    BUNDLE_METHOD_ID,
    BUNDLE_SCHEMA_ID,
    BUNDLE_SCHEMA_VERSION,
    BUNDLE_STATUS,
    CAPABILITY_KEYS,
    REASON_CODES_BY_STATE,
    read_validated_recording_behavior_bundle,
)
from fisheye.cohorts.registry import (
    MANIFEST_SCHEMA_ID as FROZEN_COHORT_SCHEMA_ID,
    MANIFEST_SCHEMA_VERSION as FROZEN_COHORT_SCHEMA_VERSION,
    validate_frozen_cohort,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


HISTORICAL_TASK_ADAPTER_ID = "composable_chaser_task_v5"
HISTORICAL_TASK_PROFILE = "historical_composable_chaser_task_v5_import_v1"
FROZEN_COHORT_ADAPTER_ID = "frozen_cohort_manifest_v2"
FROZEN_COHORT_PROFILE = "frozen_cohort_manifest_v2_import_v1"
RECORDING_BUNDLE_ADAPTER_ID = "validated_recording_behavior_bundle_v1"
RECORDING_BUNDLE_CAPABILITY_PROFILE = "validated_recording_behavior_bundle_v1"
INVALID_DISPOSITIONS_SCHEMA_ID = (
    "palette.analysis.validated_behavior_invalid_member_dispositions"
)
INVALID_DISPOSITIONS_SCHEMA_VERSION = 1

EXACT_CHASER_ADMISSION_ROLE = "exact_chaser_projection"
RECORDING_ANALYSIS_UNIT_POLICY_ID = "recording_scoped_distinct_animal_v1"
MISSING_BATCH_POLICY_ID = "missing_acquisition_batch_not_inferred_v1"

HISTORICAL_CONTROLLER_PROXY_TEMPORAL_POLICY = {
    "policy_id": "historical_controller_input_provenance_proxy_v1",
    "temporal_alignment_requirement": "input_provenance_proxy_allowed",
    "temporal_alignment_class": "controller_input_provenance_proxy",
    "physical_presentation_verified": False,
    "presentation_timestamp_available": False,
    "camera_presentation_clock_transform_available": False,
    "camera_exposure_reference": "unknown",
    "scientific_use_class": "exploratory_proxy",
}

_NONADMITTED_CAPABILITY_REASONS = {
    "invalid": "blocked_by_invalid_membership",
    "unavailable": "blocked_by_unavailable_membership",
    "excluded": "member_not_admitted",
}


def _fail(message: str) -> None:
    raise ValidatedBehaviorCohortError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def sha256_file(path: str | Path) -> str:
    """Return the byte identity used by adapter source bindings."""

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: str | Path, *, field: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{field} does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorCohortError(
            f"Cannot read strict JSON object from {source}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"{field} must contain one JSON object.")
    return source, value


def _load_composable_chaser_task_v5(path: str | Path) -> dict[str, Any]:
    # Import lazily: the historical execution module also imports plotting
    # dependencies that generic membership/bundle consumers do not need.
    from fisheye.utils.materialize_composable_chaser_successor_cohort import (
        TASK_SCHEMA_ID,
        load_cohort_task,
    )

    task = load_cohort_task(path)
    if task.get("schema_id") != TASK_SCHEMA_ID or task.get("schema_version") != 5:
        _fail("Historical adapter accepts only composable-chaser task schema v5.")
    return task


def _canonical_under(value: object, *, root: Path, field: str) -> str:
    raw = _text(value, field=field)
    path = Path(raw)
    if not path.is_absolute() or str(path.resolve(strict=False)) != raw:
        _fail(f"{field} must be one canonical absolute path.")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValidatedBehaviorCohortError(
            f"{field} escapes its declared root."
        ) from exc
    return raw


def _source_subject_ids(member: Mapping[str, Any]) -> tuple[list[str], str]:
    """Extract capture-time UUID-like values without promoting their authority."""

    values = member.get("subject_values")
    candidates: set[str] = set()
    if isinstance(values, Mapping):
        for key, raw in values.items():
            if "uuid" not in str(key).casefold():
                continue
            sequence = raw if isinstance(raw, list) else [raw]
            for item in sequence:
                if type(item) is str and item.strip():
                    candidates.add(item.strip())
    if candidates:
        return sorted(candidates), "capture_time_non_authoritative"
    return [], "unavailable_in_bound_source_membership"


def recording_scoped_analysis_unit_policy(
    *,
    distinct_animal_count: int,
    decision_timestamp_utc: str,
    decision_evidence_path: str | Path,
    decision_evidence_file_sha256: str,
    capture_subject_uuid_reuse_count: int,
) -> dict[str, Any]:
    """Build the explicit temporary recording-by-animal analysis-unit policy."""

    evidence = Path(decision_evidence_path).expanduser().resolve()
    record = {
        "policy_id": RECORDING_ANALYSIS_UNIT_POLICY_ID,
        "analysis_unit_kind": "recording",
        "member_id_field": "recording_id",
        "scientific_scope": "temporary_recording_by_distinct_animal_unit",
        "source_subject_identity_use": "retained_as_non_authoritative_evidence_only",
        "distinct_animal_count": int(distinct_animal_count),
        "capture_subject_uuid_reuse_count": int(capture_subject_uuid_reuse_count),
        "subject_identity_status": (
            "capture_uuid_reuse_incident_recording_scoped_workaround"
            if int(capture_subject_uuid_reuse_count) > 0
            else "recording_scoped_distinct_animal_decision"
        ),
        "decision_timestamp_utc": _text(
            decision_timestamp_utc, field="decision_timestamp_utc"
        ),
        "decision_evidence": {
            "path": str(evidence),
            "file_sha256": _digest(
                decision_evidence_file_sha256,
                field="decision_evidence_file_sha256",
            ),
        },
    }
    return policy_envelope(record)


def missing_acquisition_batch_policy() -> dict[str, Any]:
    """Return the explicit no-inference policy for absent batch identity."""

    return policy_envelope(
        {
            "policy_id": MISSING_BATCH_POLICY_ID,
            "missing_identity_status": "missing_historical_not_inferred",
            "authoritative_identity_status": "authoritative",
            "inference_allowed": False,
            "confirmatory_batch_clustered_analysis_allowed": False,
        }
    )


def historical_controller_proxy_temporal_policy() -> dict[str, Any]:
    """Return the exact caveat retained by historical chaser task v5."""

    return policy_envelope(HISTORICAL_CONTROLLER_PROXY_TEMPORAL_POLICY)


def _normalize_dispositions(
    value: Mapping[str, Mapping[str, Any]], *, recording_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if set(value) != recording_ids:
        missing = sorted(recording_ids - set(value))
        extra = sorted(set(value) - recording_ids)
        _fail(
            "Dispositions must name every and only source member; "
            f"missing={missing!r}, extra={extra!r}."
        )
    result: dict[str, dict[str, Any]] = {}
    for recording_id, raw in value.items():
        record = _mapping(raw, field=f"dispositions.{recording_id}")
        if set(record) != {
            "membership_state",
            "reason_code",
            "disposition_evidence",
            "admission_receipts",
        }:
            _fail(f"Disposition for {recording_id!r} has an inexact field set.")
        state = record.get("membership_state")
        if state not in MEMBERSHIP_REASON_CODES:
            _fail(f"Disposition state for {recording_id!r} is invalid.")
        if record.get("reason_code") not in MEMBERSHIP_REASON_CODES[state]:
            _fail(f"Disposition reason for {recording_id!r} is invalid.")
        receipts = record.get("admission_receipts")
        if not isinstance(receipts, list) or (state == "admitted") != bool(receipts):
            _fail(f"Disposition receipts for {recording_id!r} disagree with its state.")
        result[str(recording_id)] = _plain(record)
    return result


def read_invalid_member_dispositions(
    path: str | Path,
    *,
    expected_source_membership_record_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read one self-digested explicit non-admission decision document."""

    source, raw = _read_object(path, field="invalid-member dispositions")
    persisted = _digest(raw.get("record_sha256"), field="invalid dispositions digest")
    body = {key: value for key, value in raw.items() if key != "record_sha256"}
    if canonical_json_sha256(body) != persisted:
        _fail("Invalid-member disposition document digest is stale.")
    if set(body) != {
        "schema_id",
        "schema_version",
        "source_membership_record_sha256",
        "entry_count",
        "entries",
    }:
        _fail("Invalid-member disposition document field set is inexact.")
    if (
        body.get("schema_id") != INVALID_DISPOSITIONS_SCHEMA_ID
        or body.get("schema_version") != INVALID_DISPOSITIONS_SCHEMA_VERSION
        or body.get("source_membership_record_sha256")
        != _digest(
            expected_source_membership_record_sha256,
            field="expected source-membership digest",
        )
    ):
        _fail("Invalid-member dispositions bind another schema or source roster.")
    entries = body.get("entries")
    if not isinstance(entries, list) or body.get("entry_count") != len(entries):
        _fail("Invalid-member disposition count is stale.")
    result: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, field=f"invalid dispositions entry {index}")
        if set(entry) != {"recording_id", "reason_code", "detail"}:
            _fail(f"Invalid-member disposition entry {index} is inexact.")
        recording_id = _text(
            entry.get("recording_id"),
            field=f"invalid dispositions entry {index} recording_id",
        )
        if recording_id in result:
            _fail("Invalid-member dispositions contain a duplicate recording.")
        reason = entry.get("reason_code")
        if reason not in MEMBERSHIP_REASON_CODES["invalid"]:
            _fail(f"Invalid-member disposition reason for {recording_id!r} is invalid.")
        detail = _text(
            entry.get("detail"), field=f"invalid dispositions entry {index} detail"
        )
        result[recording_id] = {
            "membership_state": "invalid",
            "reason_code": reason,
            "disposition_evidence": {
                "evidence_type": "explicit_invalid_member_dispositions_v1",
                "detail": detail,
                "path": str(source),
                "file_sha256": sha256_file(source),
                "record_sha256": persisted,
            },
            "admission_receipts": [],
        }
    return result, {
        "path": str(source),
        "file_sha256": sha256_file(source),
        "record_sha256": persisted,
    }


def plan_composable_chaser_task_v5_dispositions(
    source_task_path: str | Path,
    *,
    receipt_generation: str,
    receipt_filename: str,
    invalid_dispositions_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Plan exact admissions from one frozen receipt generation and invalid roster.

    This is source-adapter logic, not an export formula.  The explicit invalid
    document decides which parent members are not admitted.  Every remaining
    parent must have one exact, identity-matching projection receipt; a missing
    receipt fails instead of changing cohort membership.
    """

    task_path = Path(source_task_path).expanduser().resolve()
    task = _load_composable_chaser_task_v5(task_path)
    generation = _text(receipt_generation, field="receipt_generation")
    filename = _text(receipt_filename, field="receipt_filename")
    if (
        Path(generation).name != generation
        or Path(filename).name != filename
        or generation in {"latest", "current", "selected"}
        or filename in {"latest", "current", "selected"}
    ):
        _fail("Receipt generation and filename must be exact path components.")
    invalid, _ = read_invalid_member_dispositions(
        invalid_dispositions_path,
        expected_source_membership_record_sha256=task["task_sha256"],
    )
    parent_ids = {entry["recording_id"] for entry in task["entries"]}
    extras = sorted(set(invalid) - parent_ids)
    if extras:
        _fail(f"Invalid-member dispositions contain foreign recordings: {extras!r}.")
    dispositions: dict[str, dict[str, Any]] = {}
    for entry in task["entries"]:
        recording_id = entry["recording_id"]
        if recording_id in invalid:
            dispositions[recording_id] = invalid[recording_id]
            continue
        plot_output = (
            Path(
                _text(
                    entry.get("plot_output_dir"), field="source entry plot_output_dir"
                )
            )
            .expanduser()
            .resolve()
        )
        receipt_path = (
            plot_output / "source_validation_receipts" / generation / filename
        ).resolve()
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"Declared admitted member lacks its exact receipt: {receipt_path}"
            )
        _, raw_receipt = _read_object(receipt_path, field="projection receipt")
        receipt = validate_exact_chaser_projection_receipt(
            raw_receipt,
            expected_analysis_zarr=entry["analysis_zarr"],
            expected_recording_id=recording_id,
            validate_current_metadata=False,
            validate_child_receipts=False,
        )
        binding = {
            "role": EXACT_CHASER_ADMISSION_ROLE,
            "path": str(receipt_path),
            "file_sha256": sha256_file(receipt_path),
            "record_sha256": receipt["record_sha256"],
            "schema_id": receipt["schema_id"],
            "schema_version": receipt["schema_version"],
        }
        dispositions[recording_id] = {
            "membership_state": "admitted",
            "reason_code": None,
            "disposition_evidence": {
                "evidence_type": "exact_projection_receipt_admission_v1",
                "detail": None,
                "path": str(receipt_path),
                "file_sha256": binding["file_sha256"],
                "record_sha256": binding["record_sha256"],
            },
            "admission_receipts": [binding],
        }
    return dispositions


def _shallow_exact_chaser_receipt(
    binding: Mapping[str, Any],
    *,
    recording_id: str,
    analysis_zarr: str,
) -> dict[str, Any]:
    path = Path(_text(binding.get("path"), field="admission receipt path")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Admission receipt does not exist: {path}")
    observed_file_sha = sha256_file(path)
    if observed_file_sha != _digest(
        binding.get("file_sha256"), field="admission receipt file digest"
    ):
        _fail(f"Admission receipt file digest changed: {path}")
    _, raw = _read_object(path, field="admission receipt")
    receipt = validate_exact_chaser_projection_receipt(
        raw,
        expected_analysis_zarr=analysis_zarr,
        expected_recording_id=recording_id,
        validate_current_metadata=False,
        validate_child_receipts=False,
    )
    if binding.get("role") != EXACT_CHASER_ADMISSION_ROLE:
        _fail("Historical chaser admission requires the exact-chaser receipt role.")
    expected = {
        "role": EXACT_CHASER_ADMISSION_ROLE,
        "path": str(path),
        "file_sha256": observed_file_sha,
        "record_sha256": receipt["record_sha256"],
        "schema_id": receipt["schema_id"],
        "schema_version": receipt["schema_version"],
    }
    if _plain(binding) != _plain(expected):
        _fail(f"Admission receipt binding is stale or inexact: {path}")
    return expected


def _normalized_member(
    *,
    source_ordinal: int,
    dataset_id: object,
    recording_id: object,
    analysis_zarr: object,
    protocol_names: Sequence[object],
    protocol_hashes: Sequence[object],
    source_member: Mapping[str, Any],
    source_subject_ids: Sequence[str],
    source_subject_identity_status: str,
    disposition: Mapping[str, Any],
    analysis_root: Path,
    receipt_root: Path,
) -> dict[str, Any]:
    normalized_recording = _text(recording_id, field="recording_id")
    archive = _canonical_under(analysis_zarr, root=analysis_root, field="analysis_zarr")
    receipts = _plain(disposition["admission_receipts"])
    for receipt in receipts:
        receipt_path = _canonical_under(
            _mapping(receipt, field="admission receipt").get("path"),
            root=receipt_root,
            field="admission receipt path",
        )
        if Path(receipt_path).name.casefold().startswith("latest"):
            _fail("Admission receipt cannot be selected through a latest alias.")
        _shallow_exact_chaser_receipt(
            receipt,
            recording_id=normalized_recording,
            analysis_zarr=archive,
        )
    return {
        "source_ordinal": int(source_ordinal),
        "dataset_id": _text(dataset_id, field="dataset_id"),
        "recording_id": normalized_recording,
        "analysis_zarr": archive,
        "protocol_names": sorted(
            {_text(item, field="protocol name") for item in protocol_names}
        ),
        "protocol_hashes": sorted(
            {_digest(item, field="protocol hash") for item in protocol_hashes}
        ),
        "source_member_sha256": canonical_json_sha256(_plain(source_member)),
        "source_subject_ids": sorted(set(source_subject_ids)),
        "source_subject_identity_status": source_subject_identity_status,
        "acquisition_batch_id": None,
        "acquisition_batch_identity_status": "missing_historical_not_inferred",
        "analysis_unit_kind": "recording",
        "analysis_unit_id": normalized_recording,
        "membership_state": disposition["membership_state"],
        "reason_code": disposition["reason_code"],
        "disposition_evidence": _plain(disposition["disposition_evidence"]),
        "admission_receipts": receipts,
    }


def build_membership_from_composable_chaser_task_v5(
    source_task_path: str | Path,
    *,
    membership_id: str,
    dispositions_by_recording: Mapping[str, Mapping[str, Any]],
    analysis_zarr_root: str | Path,
    admission_receipt_root: str | Path,
    analysis_unit_policy: Mapping[str, Any],
    acquisition_batch_policy: Mapping[str, Any],
    temporal_alignment_policy: Mapping[str, Any],
    palette_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Import the historical schema-v5 task into the generic membership."""

    task_path = Path(source_task_path).expanduser().resolve()
    task = _load_composable_chaser_task_v5(task_path)
    entries = task["entries"]
    recording_ids = {
        _text(
            _mapping(item, field="source entry").get("recording_id"),
            field="recording_id",
        )
        for item in entries
    }
    if len(recording_ids) != len(entries):
        _fail("Source task contains duplicate recording IDs.")
    dispositions = _normalize_dispositions(
        dispositions_by_recording, recording_ids=recording_ids
    )
    analysis_root = Path(analysis_zarr_root).expanduser().resolve(strict=False)
    receipt_root = Path(admission_receipt_root).expanduser().resolve(strict=False)
    members = []
    for entry_raw in entries:
        entry = _mapping(entry_raw, field="source entry")
        recording_id = _text(entry.get("recording_id"), field="recording_id")
        members.append(
            _normalized_member(
                source_ordinal=int(entry.get("task_index")),
                dataset_id=entry.get("dataset_id"),
                recording_id=recording_id,
                analysis_zarr=entry.get("analysis_zarr"),
                protocol_names=[entry.get("protocol_name")],
                protocol_hashes=[entry.get("protocol_hash")],
                source_member=entry,
                source_subject_ids=[],
                source_subject_identity_status="unavailable_in_bound_source_membership",
                disposition=dispositions[recording_id],
                analysis_root=analysis_root,
                receipt_root=receipt_root,
            )
        )
    source_member_digests = [
        canonical_json_sha256(_plain(_mapping(item, field="source entry")))
        for item in entries
    ]
    source_binding = {
        "adapter_id": HISTORICAL_TASK_ADAPTER_ID,
        "schema_id": task["schema_id"],
        "schema_version": 5,
        "profile": HISTORICAL_TASK_PROFILE,
        "path": str(task_path),
        "file_sha256": sha256_file(task_path),
        "record_sha256": task["task_sha256"],
        "member_count": len(entries),
        "source_members_sha256": canonical_json_sha256(source_member_digests),
    }
    return build_validated_behavior_cohort_membership(
        membership_id=membership_id,
        source_membership=source_binding,
        members=members,
        analysis_zarr_root=analysis_root,
        admission_receipt_root=receipt_root,
        analysis_unit_policy=analysis_unit_policy,
        acquisition_batch_policy=acquisition_batch_policy,
        temporal_alignment_policy=temporal_alignment_policy,
        palette_commit=palette_commit,
        created_at_utc=created_at_utc,
    )


def build_membership_from_frozen_cohort_v2(
    source_manifest_path: str | Path,
    *,
    membership_id: str,
    dispositions_by_recording: Mapping[str, Mapping[str, Any]],
    analysis_zarr_root: str | Path,
    admission_receipt_root: str | Path,
    analysis_unit_policy: Mapping[str, Any],
    acquisition_batch_policy: Mapping[str, Any],
    temporal_alignment_policy: Mapping[str, Any],
    palette_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Import the durable frozen-cohort v2 interface into the same contract."""

    manifest_path, manifest = _read_object(source_manifest_path, field="frozen cohort")
    errors = validate_frozen_cohort(manifest, check_hash=True)
    if errors:
        _fail("Frozen cohort is invalid: " + "; ".join(errors))
    if (
        manifest.get("schema_id") != FROZEN_COHORT_SCHEMA_ID
        or manifest.get("schema_version") != FROZEN_COHORT_SCHEMA_VERSION
    ):
        _fail("Frozen-cohort importer accepts only schema version 2.")
    source_members = manifest["members"]
    recording_ids = {
        _text(
            _mapping(item, field="frozen member").get("recording_id"),
            field="recording_id",
        )
        for item in source_members
    }
    if len(recording_ids) != len(source_members):
        _fail("Frozen cohort contains duplicate recording IDs.")
    dispositions = _normalize_dispositions(
        dispositions_by_recording, recording_ids=recording_ids
    )
    analysis_root = Path(analysis_zarr_root).expanduser().resolve(strict=False)
    receipt_root = Path(admission_receipt_root).expanduser().resolve(strict=False)
    members = []
    for source_ordinal, raw in enumerate(source_members, start=1):
        member = _mapping(raw, field="frozen member")
        recording_id = _text(member.get("recording_id"), field="recording_id")
        subject_ids, subject_status = _source_subject_ids(member)
        names = member.get("protocol_names")
        hashes = member.get("protocol_hashes")
        if not isinstance(names, list) or not isinstance(hashes, list):
            _fail("Frozen member lacks explicit protocol names or hashes.")
        members.append(
            _normalized_member(
                source_ordinal=source_ordinal,
                dataset_id=member.get("dataset_id"),
                recording_id=recording_id,
                analysis_zarr=member.get("zarr_path"),
                protocol_names=names,
                protocol_hashes=hashes,
                source_member=member,
                source_subject_ids=subject_ids,
                source_subject_identity_status=subject_status,
                disposition=dispositions[recording_id],
                analysis_root=analysis_root,
                receipt_root=receipt_root,
            )
        )
    source_member_digests = [
        canonical_json_sha256(_plain(_mapping(item, field="frozen member")))
        for item in source_members
    ]
    source_binding = {
        "adapter_id": FROZEN_COHORT_ADAPTER_ID,
        "schema_id": FROZEN_COHORT_SCHEMA_ID,
        "schema_version": FROZEN_COHORT_SCHEMA_VERSION,
        "profile": FROZEN_COHORT_PROFILE,
        "path": str(manifest_path),
        "file_sha256": sha256_file(manifest_path),
        "record_sha256": manifest["manifest_sha256"],
        "member_count": len(source_members),
        "source_members_sha256": canonical_json_sha256(source_member_digests),
    }
    return build_validated_behavior_cohort_membership(
        membership_id=membership_id,
        source_membership=source_binding,
        members=members,
        analysis_zarr_root=analysis_root,
        admission_receipt_root=receipt_root,
        analysis_unit_policy=analysis_unit_policy,
        acquisition_batch_policy=acquisition_batch_policy,
        temporal_alignment_policy=temporal_alignment_policy,
        palette_commit=palette_commit,
        created_at_utc=created_at_utc,
    )


def validate_membership_current_sources(value: object) -> Mapping[str, Any]:
    """Revalidate the bound source roster, decision evidence, and receipts."""

    membership = validate_validated_behavior_cohort_membership(value)
    source = membership["source_membership"]
    source_path = Path(source["path"])
    if not source_path.is_file() or sha256_file(source_path) != source["file_sha256"]:
        _fail("Bound source-membership file is absent or changed.")
    adapter_id = source["adapter_id"]
    if adapter_id == HISTORICAL_TASK_ADAPTER_ID:
        document = _load_composable_chaser_task_v5(source_path)
        source_members = document["entries"]
        record_sha = document["task_sha256"]
        identities = [
            (
                int(item["task_index"]),
                item["dataset_id"],
                item["recording_id"],
                item["analysis_zarr"],
                [item["protocol_name"]],
                [item["protocol_hash"]],
            )
            for item in source_members
        ]
    elif adapter_id == FROZEN_COHORT_ADAPTER_ID:
        _, document = _read_object(source_path, field="frozen cohort")
        errors = validate_frozen_cohort(document, check_hash=True)
        if errors:
            _fail("Bound frozen cohort is invalid: " + "; ".join(errors))
        source_members = document["members"]
        record_sha = document["manifest_sha256"]
        identities = [
            (
                index,
                item["dataset_id"],
                item["recording_id"],
                item["zarr_path"],
                item["protocol_names"],
                item["protocol_hashes"],
            )
            for index, item in enumerate(source_members, start=1)
        ]
    else:
        _fail(f"No installed source adapter validates {adapter_id!r}.")
    source_digests = [canonical_json_sha256(_plain(item)) for item in source_members]
    if (
        record_sha != source["record_sha256"]
        or len(source_members) != source["member_count"]
        or canonical_json_sha256(source_digests) != source["source_members_sha256"]
    ):
        _fail("Bound source-membership identity or member roster changed.")
    by_source_ordinal = {
        int(member["source_ordinal"]): member for member in membership["members"]
    }
    if set(by_source_ordinal) != {item[0] for item in identities}:
        _fail("Normalized membership source ordinals differ from the parent roster.")
    for source_ordinal, dataset_id, recording_id, archive, names, hashes in identities:
        member = by_source_ordinal[source_ordinal]
        if (
            member["dataset_id"] != dataset_id
            or member["recording_id"] != recording_id
            or member["analysis_zarr"] != archive
            or list(member["protocol_names"]) != sorted(set(names))
            or list(member["protocol_hashes"]) != sorted(set(hashes))
            or member["source_member_sha256"] != source_digests[source_ordinal - 1]
        ):
            _fail("Normalized member differs from its exact source member.")
    unit_policy = membership["analysis_unit_policy"]["record"]
    required_unit_policy_fields = {
        "policy_id",
        "analysis_unit_kind",
        "member_id_field",
        "scientific_scope",
        "source_subject_identity_use",
        "distinct_animal_count",
        "capture_subject_uuid_reuse_count",
        "subject_identity_status",
        "decision_timestamp_utc",
        "decision_evidence",
    }
    reuse_count = unit_policy.get("capture_subject_uuid_reuse_count")
    expected_subject_status = (
        "capture_uuid_reuse_incident_recording_scoped_workaround"
        if type(reuse_count) is int and reuse_count > 0
        else "recording_scoped_distinct_animal_decision"
    )
    if (
        set(unit_policy) != required_unit_policy_fields
        or unit_policy.get("policy_id") != RECORDING_ANALYSIS_UNIT_POLICY_ID
        or unit_policy.get("analysis_unit_kind") != "recording"
        or unit_policy.get("member_id_field") != "recording_id"
        or unit_policy.get("scientific_scope")
        != "temporary_recording_by_distinct_animal_unit"
        or unit_policy.get("source_subject_identity_use")
        != "retained_as_non_authoritative_evidence_only"
        or type(reuse_count) is not int
        or reuse_count < 0
        or reuse_count > membership["member_count"]
        or unit_policy.get("subject_identity_status") != expected_subject_status
    ):
        _fail("Recording-scoped analysis-unit policy is inexact or invalid.")
    decision_timestamp = _text(
        unit_policy.get("decision_timestamp_utc"), field="decision timestamp"
    )
    try:
        parsed_decision_timestamp = datetime.fromisoformat(
            decision_timestamp.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValidatedBehaviorCohortError(
            "Analysis-unit decision timestamp is not ISO-8601."
        ) from exc
    if (
        parsed_decision_timestamp.tzinfo is None
        or parsed_decision_timestamp.utcoffset() is None
        or parsed_decision_timestamp.utcoffset().total_seconds() != 0
    ):
        _fail("Analysis-unit decision timestamp must be expressed in UTC.")
    evidence = _mapping(unit_policy.get("decision_evidence"), field="decision evidence")
    evidence_path = Path(
        _text(evidence.get("path"), field="decision evidence path")
    ).resolve()
    if not evidence_path.is_file() or sha256_file(evidence_path) != _digest(
        evidence.get("file_sha256"), field="decision evidence digest"
    ):
        _fail("Analysis-unit decision evidence is absent or changed.")
    if unit_policy.get("distinct_animal_count") != membership["member_count"]:
        _fail("Analysis-unit decision count differs from membership.")
    if (
        membership["acquisition_batch_policy"]["record"]
        != missing_acquisition_batch_policy()["record"]
    ):
        _fail("Historical membership may not infer an acquisition batch.")
    if (
        adapter_id == HISTORICAL_TASK_ADAPTER_ID
        and membership["temporal_alignment_policy"]["record"]
        != HISTORICAL_CONTROLLER_PROXY_TEMPORAL_POLICY
    ):
        _fail("Historical task v5 requires its explicit temporal-proxy caveat.")
    for member in membership["members"]:
        disposition_evidence = member["disposition_evidence"]
        evidence_path_raw = disposition_evidence["path"]
        if evidence_path_raw is not None:
            evidence_path = Path(evidence_path_raw)
            if (
                not evidence_path.is_file()
                or sha256_file(evidence_path) != disposition_evidence["file_sha256"]
            ):
                _fail("Member disposition evidence is absent or changed.")
        if (
            disposition_evidence["evidence_type"]
            == "explicit_invalid_member_dispositions_v1"
        ):
            invalid, _ = read_invalid_member_dispositions(
                evidence_path_raw,
                expected_source_membership_record_sha256=source["record_sha256"],
            )
            expected = invalid.get(member["recording_id"])
            if (
                expected is None
                or expected["reason_code"] != member["reason_code"]
                or expected["disposition_evidence"] != disposition_evidence
            ):
                _fail("Invalid-member disposition no longer matches its decision row.")
        for binding in member["admission_receipts"]:
            _shallow_exact_chaser_receipt(
                binding,
                recording_id=member["recording_id"],
                analysis_zarr=member["analysis_zarr"],
            )
    return membership


def validated_recording_behavior_capability_contract() -> dict[str, Any]:
    """Return the exact capability vocabulary for the current bundle adapter."""

    reasons: dict[str, set[str | None]] = {
        state: set(REASON_CODES_BY_STATE[state]) for state in CAPABILITY_STATES
    }
    reasons["unavailable"].update(
        {
            "blocked_by_invalid_membership",
            "blocked_by_unavailable_membership",
        }
    )
    reasons["inapplicable"].add("member_not_admitted")
    return build_capability_contract(
        profile_id=RECORDING_BUNDLE_CAPABILITY_PROFILE,
        keys=CAPABILITY_KEYS,
        reason_codes_by_state={key: tuple(value) for key, value in reasons.items()},
    )


def _normalize_bundle_capabilities(bundle: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in CAPABILITY_KEYS:
        raw = _mapping(bundle["capabilities"][key], field=f"capabilities.{key}")
        binding = None
        if raw["state"] == "complete":
            binding = {
                "scope": raw["binding_scope"],
                "key": raw["binding_key"],
            }
        result[key] = {
            "state": raw["state"],
            "reason_code": raw["reason_code"],
            "detail": raw["detail"],
            "binding": binding,
        }
    return result


def _complete_bundle_member(
    path: str | Path,
    *,
    membership_member: Mapping[str, Any],
    validate_current_sources: bool,
) -> dict[str, Any]:
    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Recording bundle does not exist: {bundle_path}")
    bundle = read_validated_recording_behavior_bundle(
        bundle_path,
        expected_analysis_zarr=membership_member["analysis_zarr"],
        expected_recording_id=membership_member["recording_id"],
        validate_current_sources=validate_current_sources,
    )
    admission = list(membership_member["admission_receipts"])
    if len(admission) != 1 or admission[0]["role"] != EXACT_CHASER_ADMISSION_ROLE:
        _fail("Recording-bundle adapter requires one exact-chaser admission receipt.")
    projection = bundle["projection_receipt"]
    if (
        projection["receipt_path"] != admission[0]["path"]
        or projection["receipt_sha256"] != admission[0]["record_sha256"]
        or projection["schema_id"] != admission[0]["schema_id"]
        or projection["schema_version"] != admission[0]["schema_version"]
    ):
        _fail("Recording bundle binds another admission receipt generation.")
    binding_inventory = {
        "source_bindings": _plain(bundle["source_bindings"]),
        "scientific_child_bindings": _plain(bundle["scientific_child_bindings"]),
    }
    return {
        "recording_id": membership_member["recording_id"],
        "bundle_state": "complete",
        "reason_code": None,
        "bundle": {
            "adapter_id": RECORDING_BUNDLE_ADAPTER_ID,
            "path": str(bundle_path),
            "file_sha256": sha256_file(bundle_path),
            "record_sha256": bundle["record_sha256"],
            "schema_id": bundle["schema_id"],
            "schema_version": bundle["schema_version"],
            "method_id": bundle["method_id"],
            "status": bundle["status"],
            "receipt_bindings": [
                {
                    "role": EXACT_CHASER_ADMISSION_ROLE,
                    "path": projection["receipt_path"],
                    "file_sha256": admission[0]["file_sha256"],
                    "record_sha256": projection["receipt_sha256"],
                    "schema_id": projection["schema_id"],
                    "schema_version": projection["schema_version"],
                }
            ],
            "binding_inventory_sha256": canonical_json_sha256(binding_inventory),
        },
        "capabilities": _normalize_bundle_capabilities(bundle),
    }


def _nonadmitted_bundle_member(member: Mapping[str, Any]) -> dict[str, Any]:
    membership_state = member["membership_state"]
    if membership_state == "admitted":
        _fail("Admitted members require a complete bundle record.")
    if membership_state == "invalid":
        capabilities = {
            key: {
                "state": "invalid" if key == "semantic_epochs" else "unavailable",
                "reason_code": "invalid_source"
                if key == "semantic_epochs"
                else _NONADMITTED_CAPABILITY_REASONS["invalid"],
                "detail": member["disposition_evidence"]["detail"],
                "binding": None,
            }
            for key in CAPABILITY_KEYS
        }
    else:
        state = "inapplicable" if membership_state == "excluded" else "unavailable"
        reason = _NONADMITTED_CAPABILITY_REASONS[membership_state]
        capabilities = {
            key: {
                "state": state,
                "reason_code": reason,
                "detail": member["disposition_evidence"]["detail"],
                "binding": None,
            }
            for key in CAPABILITY_KEYS
        }
    return {
        "recording_id": member["recording_id"],
        "bundle_state": membership_state,
        "reason_code": member["reason_code"],
        "bundle": None,
        "capabilities": capabilities,
    }


def build_bundle_set_from_validated_recording_behavior_bundles(
    *,
    bundle_set_id: str,
    membership: Mapping[str, Any],
    membership_path: str | Path,
    bundle_paths_by_recording: Mapping[str, str | Path],
    bundle_root: str | Path,
    palette_commit: str,
    created_at_utc: str,
    validate_current_sources: bool = True,
) -> dict[str, Any]:
    """Adapt v1 recording bundles into the generic closed bundle-set format."""

    validated_membership = validate_membership_current_sources(membership)
    membership_file = Path(membership_path).expanduser().resolve()
    if not membership_file.is_file():
        raise FileNotFoundError(
            f"Membership manifest does not exist: {membership_file}"
        )
    admitted = {
        member["recording_id"]
        for member in validated_membership["members"]
        if member["membership_state"] == "admitted"
    }
    if set(bundle_paths_by_recording) != admitted:
        _fail("Bundle paths must name every and only admitted membership record.")
    member_records = []
    for member in validated_membership["members"]:
        recording_id = member["recording_id"]
        if member["membership_state"] == "admitted":
            member_records.append(
                _complete_bundle_member(
                    bundle_paths_by_recording[recording_id],
                    membership_member=member,
                    validate_current_sources=validate_current_sources,
                )
            )
        else:
            member_records.append(_nonadmitted_bundle_member(member))
    contract = validated_recording_behavior_capability_contract()
    profile = {
        "adapter_id": RECORDING_BUNDLE_ADAPTER_ID,
        "bundle_schema_id": BUNDLE_SCHEMA_ID,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_method_id": BUNDLE_METHOD_ID,
        "bundle_status": BUNDLE_STATUS,
        "capability_contract_sha256": contract["record_sha256"],
    }
    return build_validated_behavior_bundle_set(
        bundle_set_id=bundle_set_id,
        membership=validated_membership,
        membership_path=membership_file,
        membership_file_sha256=sha256_file(membership_file),
        bundle_root=bundle_root,
        bundle_profile=profile,
        capability_contract=contract,
        members=member_records,
        palette_commit=palette_commit,
        created_at_utc=created_at_utc,
    )


def validate_recording_behavior_bundle_set_current_sources(
    value: object,
    *,
    membership: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Re-open the exact v1 bundle files selected by a normalized bundle set."""

    validated_membership = validate_membership_current_sources(membership)
    bundle_set = validate_validated_behavior_bundle_set(
        value, membership=validated_membership
    )
    membership_binding = bundle_set["membership"]
    membership_path = Path(membership_binding["path"])
    if (
        not membership_path.is_file()
        or sha256_file(membership_path) != membership_binding["file_sha256"]
    ):
        _fail("Bundle-set membership file is absent or changed.")
    _, persisted_membership_raw = _read_object(
        membership_path, field="bundle-set membership"
    )
    persisted_membership = validate_membership_current_sources(persisted_membership_raw)
    if _plain(persisted_membership) != _plain(validated_membership):
        _fail("Bundle set was validated against another membership file.")
    profile = bundle_set["bundle_profile"]
    expected_profile = {
        "adapter_id": RECORDING_BUNDLE_ADAPTER_ID,
        "bundle_schema_id": BUNDLE_SCHEMA_ID,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_method_id": BUNDLE_METHOD_ID,
        "bundle_status": BUNDLE_STATUS,
        "capability_contract_sha256": bundle_set["capability_contract"][
            "record_sha256"
        ],
    }
    if _plain(profile) != expected_profile:
        _fail("Bundle set is not the installed recording-bundle v1 profile.")
    members_by_id = {
        member["recording_id"]: member for member in validated_membership["members"]
    }
    for item in bundle_set["members"]:
        if item["bundle_state"] != "complete":
            continue
        rebuilt = _complete_bundle_member(
            item["bundle"]["path"],
            membership_member=members_by_id[item["recording_id"]],
            validate_current_sources=True,
        )
        if (
            rebuilt["bundle"] != item["bundle"]
            or rebuilt["capabilities"] != item["capabilities"]
        ):
            _fail("Current recording bundle differs from its bundle-set binding.")
    return bundle_set


__all__ = [
    "EXACT_CHASER_ADMISSION_ROLE",
    "FROZEN_COHORT_ADAPTER_ID",
    "FROZEN_COHORT_PROFILE",
    "HISTORICAL_TASK_ADAPTER_ID",
    "HISTORICAL_TASK_PROFILE",
    "HISTORICAL_CONTROLLER_PROXY_TEMPORAL_POLICY",
    "INVALID_DISPOSITIONS_SCHEMA_ID",
    "INVALID_DISPOSITIONS_SCHEMA_VERSION",
    "MISSING_BATCH_POLICY_ID",
    "RECORDING_ANALYSIS_UNIT_POLICY_ID",
    "RECORDING_BUNDLE_ADAPTER_ID",
    "build_bundle_set_from_validated_recording_behavior_bundles",
    "build_membership_from_composable_chaser_task_v5",
    "build_membership_from_frozen_cohort_v2",
    "historical_controller_proxy_temporal_policy",
    "missing_acquisition_batch_policy",
    "plan_composable_chaser_task_v5_dispositions",
    "read_invalid_member_dispositions",
    "recording_scoped_analysis_unit_policy",
    "sha256_file",
    "validate_membership_current_sources",
    "validate_recording_behavior_bundle_set_current_sources",
    "validated_recording_behavior_capability_contract",
]
