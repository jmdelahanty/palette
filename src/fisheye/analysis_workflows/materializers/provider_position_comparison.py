"""Immutable selector-ineligible comparisons of explicit position providers.

The comparison uses the union of immutable observation ``instance_key`` values.
It never truncates one provider to another, fills a missing provider from a
fallback, or selects a preferred estimator.  Presence, estimator validity, and
the estimator's own failure reason remain separate evidence surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import zarr

from fisheye.analysis_workflows.subject_position_source_handle import (
    SubjectPositionSourceHandle,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
    load_bound_row_identity_contract,
    stamp_and_bind_row_identity_contract,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.subject_position_types import POSITION_FAILURE_REASON_CODES
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
)


PROVIDER_POSITION_COMPARISON_SCHEMA_ID = (
    "palette.provider_position_comparison_run_manifest"
)
PROVIDER_POSITION_COMPARISON_SCHEMA_VERSION = 1
PROVIDER_POSITION_COMPARISON_PARENT_PATH = "analysis/provider_position_comparison_runs"
PROVIDER_POSITION_COMPARISON_MANIFEST_ATTR = "provider_position_comparison_manifest"
PROVIDER_POSITION_COMPARISON_MANIFEST_SHA256_ATTR = (
    "provider_position_comparison_manifest_sha256"
)
PROVIDER_POSITION_COMPARISON_POLICY_ID = (
    "union_instance_key_no_fallback_no_selection.v1"
)
PROVIDER_POSITION_COMPARISON_PUBLISH_SCHEMA_ID = (
    "palette.provider_position_comparison_publish"
)
_RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PROVIDER_ID_RE = re.compile(r"[a-z][a-z0-9_]*")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


class ProviderPositionComparisonError(ValueError):
    """Raised when providers cannot be compared without guessing."""


@dataclass(frozen=True)
class ProviderPositionComparisonPlan:
    source_zarr: Path
    run_name: str
    run_path: str
    scratch_root: Path
    local_zarr: Path
    local_run_path: Path
    parent_selector_attrs: Mapping[str, Any]
    provider_ids: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]


def _run_name(value: str) -> str:
    if type(value) is not str or _RUN_NAME_RE.fullmatch(value) is None:
        raise ProviderPositionComparisonError("run_name is not a safe run name.")
    return value


def _provider_id(value: str) -> str:
    if type(value) is not str or _PROVIDER_ID_RE.fullmatch(value) is None:
        raise ProviderPositionComparisonError(
            "provider IDs must be lowercase snake-case identifiers."
        )
    return value


def _selector_snapshot(parent: Any) -> dict[str, Any]:
    return {
        name: parent.attrs[name] for name in _SELECTOR_ATTRS if name in parent.attrs
    }


def _source_camera_frame(handle: SubjectPositionSourceHandle) -> dict[str, Any]:
    descriptor = handle.coordinate_record.get("coordinate_descriptor")
    if not isinstance(descriptor, Mapping):
        raise ProviderPositionComparisonError(
            f"Provider {handle.run_path!r} lacks a coordinate descriptor."
        )
    frame = descriptor.get("frame_record")
    if not isinstance(frame, Mapping) or set(frame) != {
        "kind",
        "record_ref",
        "record_sha256",
    }:
        raise ProviderPositionComparisonError(
            f"Provider {handle.run_path!r} lacks exact source-camera authority."
        )
    if descriptor.get("geometry_type") != "point_xy" or descriptor.get(
        "source_camera_overlay"
    ) != {"status": "direct"}:
        raise ProviderPositionComparisonError(
            "Position comparison requires direct source-camera point coordinates."
        )
    return dict(frame)


def _readonly(values: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _summary_counts(values: np.ndarray) -> dict[str, int]:
    labels = {
        int(code): str(label) for label, code in POSITION_FAILURE_REASON_CODES.items()
    }
    unique, counts = np.unique(values, return_counts=True)
    return {
        labels.get(int(code), f"unknown_{int(code)}"): int(count)
        for code, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }


def _distance_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "count": 0,
            "median_px": None,
            "p95_px": None,
            "maximum_px": None,
        }
    return {
        "count": int(finite.size),
        "median_px": float(np.median(finite)),
        "p95_px": float(np.quantile(finite, 0.95)),
        "maximum_px": float(np.max(finite)),
    }


def build_provider_position_comparison(
    providers: Sequence[tuple[str, SubjectPositionSourceHandle]],
) -> tuple[tuple[str, ...], dict[str, np.ndarray], dict[str, Any]]:
    """Build union-key arrays and summaries from exact immutable handles."""

    if len(providers) < 2:
        raise ProviderPositionComparisonError("At least two providers are required.")
    provider_ids = tuple(_provider_id(item[0]) for item in providers)
    if len(set(provider_ids)) != len(provider_ids):
        raise ProviderPositionComparisonError("Provider IDs must be unique.")
    handles = tuple(item[1] for item in providers)
    if any(type(handle) is not SubjectPositionSourceHandle for handle in handles):
        raise ProviderPositionComparisonError(
            "Every provider must be an exact subject-position source handle."
        )
    if any(handle.selector_eligible for handle in handles):
        raise ProviderPositionComparisonError(
            "Canary comparison accepts selector-ineligible position runs only."
        )
    frames = tuple(_source_camera_frame(handle) for handle in handles)
    if any(frame != frames[0] for frame in frames[1:]):
        raise ProviderPositionComparisonError(
            "Providers do not share one exact source-camera frame authority."
        )

    key_arrays: list[np.ndarray] = []
    loaded: list[dict[str, np.ndarray]] = []
    for provider_id, handle in providers:
        keys = _readonly(handle.instance_key[:], dtype=np.dtype("<u8"))
        if keys.ndim != 1 or np.unique(keys).size != keys.size:
            raise ProviderPositionComparisonError(
                f"Provider {provider_id!r} has invalid or duplicate instance keys."
            )
        frame_index = _readonly(
            handle.source_acquisition_frame_index[:], dtype=np.dtype("<i8")
        )
        position = _readonly(handle.position_xy[:], dtype=np.dtype("<f4"))
        valid = _readonly(handle.valid[:], dtype=np.dtype("bool"))
        reasons = _readonly(handle.failure_reason_codes[:], dtype=np.dtype("<u2"))
        n_rows = keys.shape[0]
        if (
            frame_index.shape != (n_rows,)
            or position.shape != (n_rows, 2)
            or valid.shape != (n_rows,)
            or reasons.shape != (n_rows,)
        ):
            raise ProviderPositionComparisonError(
                f"Provider {provider_id!r} arrays are not row aligned."
            )
        if np.any(valid & ~np.isfinite(position).all(axis=1)):
            raise ProviderPositionComparisonError(
                f"Provider {provider_id!r} marks non-finite positions valid."
            )
        key_arrays.append(keys)
        loaded.append(
            {
                "keys": keys,
                "frames": frame_index,
                "position": position,
                "valid": valid,
                "reasons": reasons,
            }
        )

    union_keys = np.unique(np.concatenate(key_arrays)).astype("<u8", copy=False)
    p_count = len(providers)
    n_union = int(union_keys.size)
    present = np.zeros((p_count, n_union), dtype=bool)
    source_rows = np.full((p_count, n_union), -1, dtype="<i8")
    provider_frames = np.full((p_count, n_union), -1, dtype="<i8")
    positions = np.full((p_count, n_union, 2), np.nan, dtype="<f4")
    valid = np.zeros((p_count, n_union), dtype=bool)
    reasons = np.zeros((p_count, n_union), dtype="<u2")
    canonical_frames = np.full(n_union, -1, dtype="<i8")
    provider_summaries: list[dict[str, Any]] = []

    for provider_index, ((provider_id, handle), values) in enumerate(
        zip(providers, loaded, strict=True)
    ):
        union_index = np.searchsorted(union_keys, values["keys"])
        if not np.array_equal(union_keys[union_index], values["keys"]):
            raise AssertionError("Union-key construction lost a provider key.")
        existing = canonical_frames[union_index]
        conflict = (existing >= 0) & (existing != values["frames"])
        if np.any(conflict):
            raise ProviderPositionComparisonError(
                f"Provider {provider_id!r} disagrees on acquisition frame identity."
            )
        canonical_frames[union_index] = values["frames"]
        present[provider_index, union_index] = True
        source_rows[provider_index, union_index] = np.arange(
            values["keys"].shape[0], dtype="<i8"
        )
        provider_frames[provider_index, union_index] = values["frames"]
        positions[provider_index, union_index] = values["position"]
        valid[provider_index, union_index] = values["valid"]
        reasons[provider_index, union_index] = values["reasons"]
        valid_count = int(np.count_nonzero(values["valid"]))
        provider_summaries.append(
            {
                "provider_id": provider_id,
                "run_path": handle.run_path,
                "manifest_sha256": handle.manifest_sha256,
                "estimator_id": handle.estimator_record["estimator_id"],
                "estimator_sha256": handle.estimator_sha256,
                "source_sha256": handle.source_sha256,
                "coordinate_sha256": handle.coordinate_sha256,
                "row_count": int(values["keys"].size),
                "valid_count": valid_count,
                "valid_fraction_of_provider_rows": (
                    float(valid_count / values["keys"].size)
                    if values["keys"].size
                    else None
                ),
                "coverage_fraction_of_union": (
                    float(values["keys"].size / n_union) if n_union else None
                ),
                "failure_reason_counts": _summary_counts(values["reasons"]),
            }
        )

    if n_union and np.any(canonical_frames < 0):  # pragma: no cover - construction
        raise AssertionError("Union rows lack acquisition frame identity.")

    pair_indices: list[tuple[int, int]] = []
    pair_ids: list[str] = []
    for left in range(p_count):
        for right in range(left + 1, p_count):
            pair_indices.append((left, right))
            pair_ids.append(f"{provider_ids[left]}__vs__{provider_ids[right]}")
    q_count = len(pair_indices)
    pair_present = np.zeros((q_count, n_union), dtype=bool)
    pair_valid = np.zeros((q_count, n_union), dtype=bool)
    pair_delta = np.full((q_count, n_union, 2), np.nan, dtype="<f4")
    pair_distance = np.full((q_count, n_union), np.nan, dtype="<f4")
    pair_summaries: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(pair_indices):
        pair_present[pair_index] = present[left] & present[right]
        pair_valid[pair_index] = pair_present[pair_index] & valid[left] & valid[right]
        rows = pair_valid[pair_index]
        pair_delta[pair_index, rows] = positions[left, rows] - positions[right, rows]
        pair_distance[pair_index, rows] = np.linalg.norm(
            pair_delta[pair_index, rows].astype(np.float64), axis=1
        ).astype("<f4")
        pair_summaries.append(
            {
                "pair_id": pair_ids[pair_index],
                "left_provider_id": provider_ids[left],
                "right_provider_id": provider_ids[right],
                "both_present_count": int(np.count_nonzero(pair_present[pair_index])),
                "both_valid_count": int(np.count_nonzero(rows)),
                "distance": _distance_summary(pair_distance[pair_index]),
            }
        )

    arrays = {
        "rows/instance_key": _readonly(union_keys),
        "rows/source_acquisition_frame_index": _readonly(canonical_frames),
        "provider_present": _readonly(present),
        "provider_source_row_index": _readonly(source_rows),
        "provider_source_acquisition_frame_index": _readonly(provider_frames),
        "provider_position_xy": _readonly(positions),
        "provider_valid": _readonly(valid),
        "provider_failure_reason_codes": _readonly(reasons),
        "pair_provider_indices": _readonly(np.asarray(pair_indices, dtype="<u2")),
        "pair_both_present": _readonly(pair_present),
        "pair_both_valid": _readonly(pair_valid),
        "pair_delta_xy": _readonly(pair_delta),
        "pair_distance_px": _readonly(pair_distance),
    }
    summary = {
        "union_row_count": n_union,
        "providers": provider_summaries,
        "pairs": pair_summaries,
        "pair_ids": pair_ids,
        "source_camera_frame": frames[0],
    }
    return provider_ids, arrays, summary


def _array_declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "sha256": array_values_sha256(values),
        }
        for path, values in sorted(arrays.items())
    ]


def plan_provider_position_comparison_run(
    analysis_zarr: str | Path,
    providers: Sequence[tuple[str, SubjectPositionSourceHandle]],
    *,
    run_name: str,
    scratch_root: str | Path,
    software_record: Mapping[str, Any],
) -> ProviderPositionComparisonPlan:
    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    name = _run_name(run_name)
    run_path = f"{PROVIDER_POSITION_COMPARISON_PARENT_PATH}/{name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        raise FileExistsError(f"Immutable comparison run exists: {target}")
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    try:
        parent = root[PROVIDER_POSITION_COMPARISON_PARENT_PATH]
        selectors = _selector_snapshot(parent)
    except KeyError:
        selectors = {}
    provider_ids, arrays, summary = build_provider_position_comparison(providers)
    software = json_attr_safe(dict(software_record))
    if not isinstance(software, dict) or not software:
        raise ProviderPositionComparisonError("software_record must be nonempty.")
    payload = {
        "run_name": name,
        "run_path": run_path,
        "status": RUN_STATUS_COMPLETE,
        "stage_selector_eligible": False,
        "policy": {
            "policy_id": PROVIDER_POSITION_COMPARISON_POLICY_ID,
            "union_axis": "sorted_unique_instance_key",
            "missing_provider_policy": "explicit_provider_present_false",
            "invalid_estimator_policy": "preserve_failure_reason_no_fallback",
            "selection": "none",
        },
        "provider_ids": list(provider_ids),
        "summary": summary,
        "arrays": _array_declarations(arrays),
        "software": software,
    }
    manifest = {
        "schema_id": PROVIDER_POSITION_COMPARISON_SCHEMA_ID,
        "schema_version": PROVIDER_POSITION_COMPARISON_SCHEMA_VERSION,
        "payload_sha256": canonical_json_sha256(payload),
        "payload": payload,
    }
    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch / f"provider_position_comparison_{uuid.uuid4().hex}.zarr"
    return ProviderPositionComparisonPlan(
        source_zarr=archive,
        run_name=name,
        run_path=run_path,
        scratch_root=scratch,
        local_zarr=local,
        local_run_path=local.joinpath(*run_path.split("/")),
        parent_selector_attrs=selectors,
        provider_ids=provider_ids,
        arrays=arrays,
        manifest=manifest,
    )


def _validate_run(
    path: Path, *, expected_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)
    manifest = run.attrs.get(PROVIDER_POSITION_COMPARISON_MANIFEST_ATTR)
    if manifest != expected_manifest:
        raise ProviderPositionComparisonError(
            "Comparison manifest is missing or stale."
        )
    digest = canonical_json_sha256(manifest)
    if run.attrs.get(PROVIDER_POSITION_COMPARISON_MANIFEST_SHA256_ATTR) != digest:
        raise ProviderPositionComparisonError("Comparison manifest digest is stale.")
    payload = manifest["payload"]
    if canonical_json_sha256(payload) != manifest["payload_sha256"]:
        raise ProviderPositionComparisonError("Comparison payload digest is stale.")
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ProviderPositionComparisonError("Comparison run is incomplete.")
    if run.attrs.get("stage_selector_eligible") is not False:
        raise ProviderPositionComparisonError(
            "Comparison run became selector eligible."
        )
    declarations = payload["arrays"]
    for declaration in declarations:
        values = np.asarray(run[declaration["path"]][:])
        observed = {
            "path": declaration["path"],
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "sha256": array_values_sha256(values),
        }
        if observed != declaration:
            raise ProviderPositionComparisonError(
                f"Comparison array drifted: {declaration['path']}."
            )
    identity = load_bound_row_identity_contract(run["rows"], run["rows/instance_key"])
    if identity.contract.domain != OBSERVATION_INSTANCE_DOMAIN:
        raise ProviderPositionComparisonError(
            "Comparison row identity domain is invalid."
        )
    return {
        "valid": True,
        "errors": [],
        "manifest_sha256": digest,
        "payload_sha256": manifest["payload_sha256"],
        "row_identity_sha256": identity.record_sha256,
        "union_row_count": int(payload["summary"]["union_row_count"]),
    }


def _write_local(plan: ProviderPositionComparisonPlan) -> None:
    root = zarr.open_group(
        str(plan.local_zarr), mode="w-", zarr_format=3, use_consolidated=False
    )
    parent = root.require_group("analysis").require_group(
        "provider_position_comparison_runs"
    )
    run = parent.create_group(plan.run_name)
    mark_run_started(run, run_name=plan.run_name, stage="provider_position_comparison")
    for path, values in plan.arrays.items():
        chunks = tuple(max(1, min(int(size), 16384)) for size in values.shape)
        run.create_array(path, data=values, chunks=chunks)
    stamp_and_bind_row_identity_contract(
        run["rows"],
        run["rows/instance_key"],
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=plan.arrays["rows/instance_key"],
        ),
    )
    run.attrs[PROVIDER_POSITION_COMPARISON_MANIFEST_ATTR] = plan.manifest
    run.attrs[PROVIDER_POSITION_COMPARISON_MANIFEST_SHA256_ATTR] = (
        canonical_json_sha256(plan.manifest)
    )
    run.attrs["stage_selector_eligible"] = False
    run.attrs["comparison_policy_id"] = PROVIDER_POSITION_COMPARISON_POLICY_ID
    mark_run_complete(
        run,
        parent_group=None,
        run_name=plan.run_name,
        allow_missing_run_provenance=True,
        missing_run_provenance_reason="comparison_manifest_binds_exact_software_record",
    )
    _validate_run(plan.local_run_path, expected_manifest=plan.manifest)


def publish_provider_position_comparison_run(
    plan: ProviderPositionComparisonPlan,
    *,
    copy_backend: str = "python",
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Atomically publish one immutable comparison without changing selectors."""

    _write_local(plan)
    acceptance: dict[str, Any] = {}

    def validate(path: Path) -> Mapping[str, Any]:
        return _validate_run(path, expected_manifest=plan.manifest)

    def prepare(root: Any) -> tuple[Any]:
        return (
            root.require_group("analysis").require_group(
                "provider_position_comparison_runs"
            ),
        )

    def complete(_root: Any, _parent: Any, run: Any) -> None:
        run.attrs["stage_selector_eligible"] = False
        mark_run_complete(
            run,
            parent_group=None,
            run_name=plan.run_name,
            allow_missing_run_provenance=True,
            missing_run_provenance_reason=(
                "comparison_manifest_binds_exact_software_record"
            ),
        )

    def verify(root: Any) -> None:
        parent = root[PROVIDER_POSITION_COMPARISON_PARENT_PATH]
        if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
            raise RuntimeError("Comparison publication changed parent selectors.")
        _validate_run(
            plan.source_zarr.joinpath(*plan.run_path.split("/")),
            expected_manifest=plan.manifest,
        )

    def finalize(_root: Any, _parent: Any, _run: Any) -> None:
        consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        acceptance["metadata"] = validate_direct_consolidated_subtree(
            plan.source_zarr, subtree_path=plan.run_path
        ).to_json()
        consolidated_root = open_zarr_root(
            plan.source_zarr, mode="r", use_consolidated=True
        )
        if plan.run_path not in consolidated_root:
            raise RuntimeError("Consolidated metadata omitted the comparison run.")

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.source_zarr.joinpath(*plan.run_path.split("/")),
            run_name=plan.run_name,
            lock_suffix="provider-position-comparison",
            publish_schema_id=PROVIDER_POSITION_COMPARISON_PUBLISH_SCHEMA_ID,
            policy=PROVIDER_POSITION_COMPARISON_POLICY_ID,
            rollback_policy="retain_failed_tombstone_leave_selectors_untouched",
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        payload_metadata={
            "provider_ids": list(plan.provider_ids),
            "selector_eligible": False,
            "selection": "none",
        },
        activate_run=finalize,
        repair_failed_publication_visibility=lambda _target: (
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
        ),
    )
    result = {
        "status": "complete",
        "run_path": plan.run_path,
        "manifest_sha256": canonical_json_sha256(plan.manifest),
        "payload_sha256": plan.manifest["payload_sha256"],
        "provider_ids": list(plan.provider_ids),
        "summary": plan.manifest["payload"]["summary"],
        "selector_eligible": False,
        "selection": "none",
        "publication": publication,
        "validation": acceptance,
    }
    if not keep_scratch and plan.local_zarr.exists():
        shutil.rmtree(plan.local_zarr)
    return json_attr_safe(result)


__all__ = [
    "PROVIDER_POSITION_COMPARISON_PARENT_PATH",
    "PROVIDER_POSITION_COMPARISON_POLICY_ID",
    "ProviderPositionComparisonError",
    "ProviderPositionComparisonPlan",
    "build_provider_position_comparison",
    "plan_provider_position_comparison_run",
    "publish_provider_position_comparison_run",
]
