"""Exact compact-v7 eye-gaze source and reviewed convention receipt.

The compact eye-angle product proves its numerical and coordinate identities,
but the direction selected from an ellipse's directionless major axis remains
a biological assumption.  This boundary therefore requires both the exact
41-array logical payload and an explicit human-reviewed convention receipt.
It never resolves an eye run through ``latest`` and never substitutes nasal
eye angle, motion heading, or an unreviewed gaze convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import copy
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.gaze_convention_validation import (
    EXPECTED_GAZE_SIGN_CONVENTION,
    SCHEMA_ID as NUMERIC_VALIDATION_SCHEMA_ID,
    SCHEMA_VERSION as NUMERIC_VALIDATION_SCHEMA_VERSION,
)
from fisheye.analysis_workflows.eye_angle_candidate_execution import (
    eye_angle_logical_manifest_sha256,
)
from fisheye.shared.eye_angle_schema import validate_eye_angle_compact_run
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent


SCHEMA_ID = "palette.analysis.eye_gaze_source_handle"
SCHEMA_VERSION = 1
CONVENTION_RECEIPT_SCHEMA_ID = "palette.analysis.gaze_convention_review_receipt"
CONVENTION_RECEIPT_SCHEMA_VERSION = 1
RUN_PARENT = "analysis/eye_angle_runs"

_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_HANDLE_SEAL = object()


class EyeGazeSourceHandleError(ValueError):
    """Raised when exact eye gaze or its reviewed convention is invalid."""


def _fail(message: str) -> None:
    raise EyeGazeSourceHandleError(message)


def _digest(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field_name} must be one lowercase SHA-256 digest.")
    return value


def _text(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field_name} must be one non-empty exact string.")
    return value


def _run_name(value: object) -> str:
    name = _text(value, field_name="run_name")
    if _RUN_NAME_RE.fullmatch(name) is None or name in {
        "latest",
        "latest_complete",
        "selected",
        "current",
        "authoritative",
    }:
        _fail("run_name must be one exact immutable child, not a selector alias.")
    return name


def _strict_json(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be one JSON object.")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise EyeGazeSourceHandleError(f"{field_name} is not strict JSON: {exc}") from exc
    if type(decoded) is not dict:
        _fail(f"{field_name} must decode to one JSON object.")
    return decoded


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _decode_names(index_group: Any, expected_count: int) -> tuple[str, ...]:
    raw = np.asarray(index_group["name"][:], dtype=np.uint8)
    if raw.ndim != 2 or raw.shape[0] != expected_count:
        _fail("Eye channel-name index shape differs from its packed array.")
    names = tuple(decode_null_terminated_text(row) for row in raw)
    if any(not name for name in names) or len(set(names)) != len(names):
        _fail("Eye channel-name index is empty or duplicated.")
    return names


def _reviewed_at(value: object) -> str:
    text = _text(value, field_name="reviewed_at_utc")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EyeGazeSourceHandleError("reviewed_at_utc is not ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail("reviewed_at_utc must include a timezone.")
    return text


def build_gaze_convention_review_receipt(
    *,
    numeric_validation: Mapping[str, Any],
    source_eye_logical_sha256: str,
    reviewer: str,
    reviewed_at_utc: str,
    review_artifact_sha256: str,
) -> dict[str, Any]:
    """Seal numeric validation plus explicit biological direction review."""

    validation = _strict_json(numeric_validation, field_name="numeric_validation")
    source_digest = _digest(
        source_eye_logical_sha256, field_name="source_eye_logical_sha256"
    )
    reviewer_text = _text(reviewer, field_name="reviewer")
    reviewed_at = _reviewed_at(reviewed_at_utc)
    artifact_digest = _digest(
        review_artifact_sha256, field_name="review_artifact_sha256"
    )
    if (
        validation.get("schema_id") != NUMERIC_VALIDATION_SCHEMA_ID
        or validation.get("schema_version") != NUMERIC_VALIDATION_SCHEMA_VERSION
        or validation.get("status") != "pass"
    ):
        _fail("Numeric gaze-convention validation did not pass the exact schema.")
    checks = validation.get("checks")
    if not isinstance(checks, list) or not checks or any(
        not isinstance(check, Mapping) or check.get("passed") is not True
        for check in checks
    ):
        _fail("Every numeric gaze-convention check must explicitly pass.")
    assumption = validation.get("direction_assumption")
    if not isinstance(assumption, Mapping) or (
        assumption.get("name") != "ellipse_axis_direction_assumption"
        or assumption.get("review_required") is not True
    ):
        _fail("Numeric validation lacks the exact biological direction assumption.")
    comparison = validation.get("comparison_contract")
    if not isinstance(comparison, Mapping) or (
        comparison.get("coordinate_frame") != "fish_body_frame"
        or comparison.get("zero") != "fish_forward"
        or comparison.get("positive") != "anatomical_left"
        or comparison.get("eye_angle_fields")
        != ["left_gaze_signed_deg", "right_gaze_signed_deg"]
    ):
        _fail("Numeric validation has an incompatible gaze comparison contract.")
    run_path = _text(
        validation.get("eye_angle_run_path"), field_name="eye_angle_run_path"
    )
    review_png = validation.get("review_png")
    review_rows = validation.get("review_row_indices")
    if type(review_png) is not str or not review_png.strip():
        _fail("Biological review requires a rendered review_png.")
    if not isinstance(review_rows, list) or not review_rows or any(
        type(value) is not int or value < 0 for value in review_rows
    ):
        _fail("Biological review requires explicit non-empty review row identities.")
    body = {
        "schema_id": CONVENTION_RECEIPT_SCHEMA_ID,
        "schema_version": CONVENTION_RECEIPT_SCHEMA_VERSION,
        "source_eye_run_path": run_path,
        "source_eye_logical_sha256": source_digest,
        "numeric_validation": validation,
        "numeric_validation_sha256": canonical_json_sha256(validation),
        "biological_direction_review": {
            "status": "accepted",
            "reviewer": reviewer_text,
            "reviewed_at_utc": reviewed_at,
            "assumption": "ellipse_axis_direction_assumption",
            "review_artifact_sha256": artifact_digest,
            "review_row_indices": list(review_rows),
        },
        "accepted_comparison_contract": {
            "coordinate_frame": "fish_body_frame",
            "zero": "fish_forward",
            "positive": EXPECTED_GAZE_SIGN_CONVENTION,
            "fields": ["left_gaze_signed_deg", "right_gaze_signed_deg"],
        },
    }
    return {**body, "receipt_sha256": canonical_json_sha256(body)}


def validate_gaze_convention_review_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_run_path: str,
    expected_logical_sha256: str,
) -> dict[str, Any]:
    """Validate a convention receipt against one exact eye payload."""

    record = _strict_json(receipt, field_name="gaze convention receipt")
    supplied_digest = _digest(
        record.get("receipt_sha256"), field_name="receipt_sha256"
    )
    body = dict(record)
    body.pop("receipt_sha256")
    if canonical_json_sha256(body) != supplied_digest:
        _fail("Gaze convention receipt self-digest is stale.")
    if (
        body.get("schema_id") != CONVENTION_RECEIPT_SCHEMA_ID
        or body.get("schema_version") != CONVENTION_RECEIPT_SCHEMA_VERSION
        or body.get("source_eye_run_path") != expected_run_path
        or body.get("source_eye_logical_sha256") != expected_logical_sha256
    ):
        _fail("Gaze convention receipt does not bind the exact eye run payload.")
    numeric = _strict_json(body.get("numeric_validation"), field_name="numeric_validation")
    if canonical_json_sha256(numeric) != body.get("numeric_validation_sha256"):
        _fail("Gaze convention numeric-validation digest is stale.")
    # Re-run the closed semantic validation through the builder, then require
    # every derived field except the receipt digest to match exactly.
    review = body.get("biological_direction_review")
    if not isinstance(review, Mapping) or review.get("status") != "accepted":
        _fail("Biological direction review was not explicitly accepted.")
    rebuilt = build_gaze_convention_review_receipt(
        numeric_validation=numeric,
        source_eye_logical_sha256=expected_logical_sha256,
        reviewer=review.get("reviewer"),
        reviewed_at_utc=review.get("reviewed_at_utc"),
        review_artifact_sha256=review.get("review_artifact_sha256"),
    )
    if rebuilt != record:
        _fail("Gaze convention receipt contains altered or extra review fields.")
    return record


@dataclass(frozen=True, slots=True, init=False)
class EyeGazeSourceHandle:
    """Immutable verified snapshot of one exact compact-v7 eye run."""

    analysis_zarr_path: Path
    run_name: str
    run_path: str
    recording_id: str
    selector_eligible: bool
    n_frames: int
    channel_variant: str
    gaze_channel_names: tuple[str, str]
    vergence_channel_name: str
    logical_manifest_sha256: str
    convention_receipt: Mapping[str, Any] = field(repr=False)
    convention_receipt_sha256: str
    frame_acquisition_id: np.ndarray = field(repr=False, compare=False)
    gaze_signed_deg: np.ndarray = field(repr=False, compare=False)
    gaze_valid: np.ndarray = field(repr=False, compare=False)
    vergence_deg: np.ndarray = field(repr=False, compare=False)
    vergence_valid: np.ndarray = field(repr=False, compare=False)
    metadata_equivalence: Mapping[str, Any] = field(repr=False)
    verification_digest: str
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _seal: object | None = None, **values: Any) -> None:
        if _seal is not _HANDLE_SEAL:
            raise TypeError("Eye-gaze handles can only be minted by their strict loader.")
        for name, value in values.items():
            if name in {"convention_receipt", "metadata_equivalence"}:
                value = _freeze(copy.deepcopy(value))
            elif name in {
                "frame_acquisition_id",
                "gaze_signed_deg",
                "gaze_valid",
                "vergence_deg",
                "vergence_valid",
            }:
                value = _readonly(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _HANDLE_SEAL)

    def align_to_acquisition_frames(
        self, acquisition_frame_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return exact gaze/vergence arrays on a requested acquisition axis."""

        requested = np.asarray(acquisition_frame_ids)
        if requested.dtype != np.dtype(np.int64) or requested.ndim != 1:
            _fail("Requested acquisition-frame identities must be exact int64 vector.")
        if np.unique(requested).size != requested.size:
            _fail("Requested acquisition-frame identities are duplicated.")
        if np.any(requested < 0) or np.any(requested >= self.n_frames):
            _fail("Requested acquisition-frame identity is outside the eye frame axis.")
        indices = requested.astype(np.int64, copy=False)
        return (
            _readonly(self.gaze_signed_deg[indices]),
            _readonly(self.gaze_valid[indices]),
            _readonly(self.vergence_deg[indices]),
            _readonly(self.vergence_valid[indices]),
        )

    def assert_current(self) -> None:
        if self._seal is not _HANDLE_SEAL:
            _fail("Eye-gaze handle verification seal is absent.")
        refreshed = load_eye_gaze_source_handle(
            self.analysis_zarr_path,
            run_name=self.run_name,
            convention_receipt=self.convention_receipt,
            channel_variant=self.channel_variant,
        )
        if refreshed.verification_digest != self.verification_digest:
            _fail("Eye-gaze source changed after the handle was sealed.")


def _load_snapshot(
    archive: Path,
    *,
    run_name: str,
    convention_receipt: Mapping[str, Any],
    channel_variant: str,
    use_consolidated: bool,
) -> dict[str, Any]:
    root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
    try:
        parent = root[RUN_PARENT]
        run = parent[run_name]
    except KeyError as exc:
        raise EyeGazeSourceHandleError(
            f"Exact eye-angle run {RUN_PARENT}/{run_name} is absent."
        ) from exc
    if not is_run_complete_in_parent(parent, run, legacy_default=False):
        _fail("Exact eye-angle run is not complete.")
    selector_eligible = run.attrs.get("stage_selector_eligible")
    if type(selector_eligible) is not bool:
        _fail("Exact eye-angle run lacks an explicit selector-eligibility state.")
    issues = validate_eye_angle_compact_run(run)
    if issues:
        _fail(
            "Eye-angle run is not exact compact-v7: "
            + "; ".join(
                f"{issue.code}:{issue.path}:{issue.message}" for issue in issues
            )
        )
    logical_digest = eye_angle_logical_manifest_sha256(run)
    run_path = f"{RUN_PARENT}/{run_name}"
    receipt = validate_gaze_convention_review_receipt(
        convention_receipt,
        expected_run_path=run_path,
        expected_logical_sha256=logical_digest,
    )
    frame_angles = np.asarray(run["frame_angles"][:])
    frame_qa = np.asarray(run["frame_qa"][:])
    angle_names = _decode_names(run["angle_channel_index"], frame_angles.shape[1])
    qa_names = _decode_names(run["qa_channel_index"], frame_qa.shape[1])
    suffix = "" if channel_variant == "raw" else "_smoothed"
    required_angles = (
        f"left_gaze_signed_deg{suffix}",
        f"right_gaze_signed_deg{suffix}",
        f"vergence_eye_angle_deg{suffix}",
    )
    missing = [name for name in required_angles if name not in angle_names]
    if missing or "valid_frame" not in qa_names:
        _fail(f"Eye frame tables lack required semantic channels: {missing!r}.")
    gaze = np.column_stack(
        [frame_angles[:, angle_names.index(name)] for name in required_angles[:2]]
    ).astype(np.float64, copy=False)
    vergence = np.asarray(
        frame_angles[:, angle_names.index(required_angles[2])], dtype=np.float64
    )
    frame_valid = np.asarray(
        frame_qa[:, qa_names.index("valid_frame")], dtype=bool
    )
    gaze_valid = frame_valid[:, None] & np.isfinite(gaze)
    vergence_valid = frame_valid & np.isfinite(vergence)
    recording_id = _text(root.attrs.get("recording_id"), field_name="recording_id")
    verification = canonical_json_sha256(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "recording_id": recording_id,
            "run_path": run_path,
            "selector_eligible": selector_eligible,
            "channel_variant": channel_variant,
            "gaze_channel_names": list(required_angles[:2]),
            "vergence_channel_name": required_angles[2],
            "logical_manifest_sha256": logical_digest,
            "convention_receipt_sha256": receipt["receipt_sha256"],
        }
    )
    return {
        "recording_id": recording_id,
        "selector_eligible": selector_eligible,
        "n_frames": int(frame_angles.shape[0]),
        "channel_variant": channel_variant,
        "gaze_channel_names": tuple(required_angles[:2]),
        "vergence_channel_name": required_angles[2],
        "logical_manifest_sha256": logical_digest,
        "convention_receipt": receipt,
        "convention_receipt_sha256": str(receipt["receipt_sha256"]),
        "frame_acquisition_id": np.arange(frame_angles.shape[0], dtype=np.int64),
        "gaze_signed_deg": gaze,
        "gaze_valid": gaze_valid,
        "vergence_deg": vergence,
        "vergence_valid": vergence_valid,
        "verification_digest": verification,
    }


def load_eye_gaze_source_handle(
    analysis_zarr_path: str | Path,
    *,
    run_name: str,
    convention_receipt: Mapping[str, Any],
    channel_variant: str = "smoothed",
) -> EyeGazeSourceHandle:
    """Load one explicit complete compact-v7 run with reviewed convention."""

    name = _run_name(run_name)
    if type(channel_variant) is not str or channel_variant not in {"raw", "smoothed"}:
        _fail("channel_variant must be exact 'raw' or 'smoothed'.")
    archive = Path(analysis_zarr_path).expanduser().resolve()
    direct = _load_snapshot(
        archive,
        run_name=name,
        convention_receipt=convention_receipt,
        channel_variant=channel_variant,
        use_consolidated=False,
    )
    consolidated = _load_snapshot(
        archive,
        run_name=name,
        convention_receipt=convention_receipt,
        channel_variant=channel_variant,
        use_consolidated=True,
    )
    if direct["verification_digest"] != consolidated["verification_digest"]:
        _fail("Eye-gaze direct metadata differs from consolidated metadata.")
    equivalence = validate_direct_consolidated_subtree(
        archive, subtree_path=f"{RUN_PARENT}/{name}"
    ).to_json()
    return EyeGazeSourceHandle(
        analysis_zarr_path=archive,
        run_name=name,
        run_path=f"{RUN_PARENT}/{name}",
        metadata_equivalence=equivalence,
        **consolidated,
        _seal=_HANDLE_SEAL,
    )


def require_eye_gaze_source_handle(value: object) -> EyeGazeSourceHandle:
    if type(value) is not EyeGazeSourceHandle:
        raise TypeError("A strict loader-minted EyeGazeSourceHandle is required.")
    value.assert_current()
    return value


__all__ = [
    "CONVENTION_RECEIPT_SCHEMA_ID",
    "CONVENTION_RECEIPT_SCHEMA_VERSION",
    "EyeGazeSourceHandle",
    "EyeGazeSourceHandleError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "build_gaze_convention_review_receipt",
    "load_eye_gaze_source_handle",
    "require_eye_gaze_source_handle",
    "validate_gaze_convention_review_receipt",
]
