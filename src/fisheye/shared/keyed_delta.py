"""Memory-bounded keyed delta planning for immutable materializations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fisheye.shared.row_source_signature import ROW_SOURCE_SIGNATURE_WIDTH_BYTES
from fisheye.shared.zarr.columnar import store_array


KEYED_DELTA_PLAN_SCHEMA_ID = "palette.keyed_delta_plan"
KEYED_DELTA_PLAN_SCHEMA_VERSION = 1
DEFAULT_DELTA_PLAN_SHARD_ROWS = 131_072

ACTION_CODE_MAP = {
    "copy": 0,
    "compute": 1,
    "preserve_manual": 2,
}
REASON_CODE_MAP = {
    "unchanged": 0,
    "added": 1,
    "source_changed": 2,
    "signature_spec_changed": 3,
    "no_reuse_source": 4,
    "preserved_manual": 5,
}
OMIT_REASON_CODE_MAP = {"deleted_from_target": 0}


class KeyedDeltaPlanError(ValueError):
    """Raised when a keyed materialization plan cannot be proven safe."""


@dataclass(frozen=True)
class KeyedDeltaPlan:
    """Compact row actions for one complete target materialization."""

    target_instance_keys: np.ndarray
    target_row_indices: np.ndarray
    source_row_indices: np.ndarray
    action_codes: np.ndarray
    reason_codes: np.ndarray
    omitted_instance_keys: np.ndarray
    omitted_source_row_indices: np.ndarray
    omitted_reason_codes: np.ndarray
    target_signature_spec_digest: str
    source_signature_spec_digest: str | None

    @property
    def row_count(self) -> int:
        return int(self.target_instance_keys.shape[0])

    def summary(self) -> dict[str, Any]:
        action_counts = {
            name: int(np.count_nonzero(self.action_codes == code))
            for name, code in ACTION_CODE_MAP.items()
        }
        reason_counts = {
            name: int(np.count_nonzero(self.reason_codes == code))
            for name, code in REASON_CODE_MAP.items()
        }
        return {
            "schema_id": KEYED_DELTA_PLAN_SCHEMA_ID,
            "schema_version": KEYED_DELTA_PLAN_SCHEMA_VERSION,
            "target_row_count": self.row_count,
            "source_row_count": int(
                np.count_nonzero(self.source_row_indices >= 0)
                + self.omitted_source_row_indices.shape[0]
            ),
            "omitted_row_count": int(self.omitted_instance_keys.shape[0]),
            "action_counts": action_counts,
            "reason_counts": reason_counts,
            "target_signature_spec_digest": self.target_signature_spec_digest,
            "source_signature_spec_digest": self.source_signature_spec_digest,
            "signature_specs_match": (
                self.source_signature_spec_digest
                == self.target_signature_spec_digest
                if self.source_signature_spec_digest is not None
                else None
            ),
        }


def _keys(values: np.ndarray, *, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise KeyedDeltaPlanError(f"{label} instance_key values must be integers.")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise KeyedDeltaPlanError(f"{label} instance_key values must be nonnegative.")
    keys = np.asarray(raw, dtype=np.uint64).reshape(-1)
    if int(np.unique(keys).shape[0]) != int(keys.shape[0]):
        raise KeyedDeltaPlanError(f"{label} instance_key values must be unique.")
    return keys


def _signatures(
    values: np.ndarray,
    *,
    row_count: int,
    label: str,
) -> np.ndarray:
    signatures = np.asarray(values)
    if signatures.dtype != np.dtype(np.uint8):
        raise KeyedDeltaPlanError(f"{label} source signatures must use uint8 dtype.")
    if signatures.shape != (int(row_count), ROW_SOURCE_SIGNATURE_WIDTH_BYTES):
        raise KeyedDeltaPlanError(
            f"{label} source signatures must have shape ({row_count}, 32)."
        )
    return np.ascontiguousarray(signatures)


def _digest(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise KeyedDeltaPlanError(f"{label} signature spec digest must be SHA-256 hex.")
    return digest


def _source_matches(
    *,
    target_keys: np.ndarray,
    source_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target match mask and source row indexes without a Python dict."""

    source_order = np.argsort(source_keys, kind="stable")
    sorted_source = source_keys[source_order]
    positions = np.searchsorted(sorted_source, target_keys)
    in_bounds = positions < sorted_source.shape[0]
    matched = np.zeros(target_keys.shape[0], dtype=bool)
    matched[in_bounds] = sorted_source[positions[in_bounds]] == target_keys[in_bounds]
    source_rows = np.full(target_keys.shape[0], -1, dtype=np.int64)
    source_rows[matched] = source_order[positions[matched]].astype(np.int64, copy=False)
    return matched, source_rows


def _omitted_source_rows(
    *,
    target_keys: np.ndarray,
    source_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_order = np.argsort(target_keys, kind="stable")
    sorted_target = target_keys[target_order]
    positions = np.searchsorted(sorted_target, source_keys)
    in_bounds = positions < sorted_target.shape[0]
    retained = np.zeros(source_keys.shape[0], dtype=bool)
    retained[in_bounds] = sorted_target[positions[in_bounds]] == source_keys[in_bounds]
    omitted_rows = np.flatnonzero(~retained).astype(np.int64, copy=False)
    return source_keys[omitted_rows], omitted_rows


def build_keyed_delta_plan(
    *,
    target_instance_keys: np.ndarray,
    target_source_signatures: np.ndarray,
    target_signature_spec_digest: str,
    source_instance_keys: np.ndarray | None = None,
    source_row_signatures: np.ndarray | None = None,
    source_signature_spec_digest: str | None = None,
    preserve_manual_target_mask: np.ndarray | None = None,
) -> KeyedDeltaPlan:
    """Plan keyed copy/compute actions using compact NumPy sorting/searching."""

    target_keys = _keys(target_instance_keys, label="Target")
    target_signatures = _signatures(
        target_source_signatures,
        row_count=int(target_keys.shape[0]),
        label="Target",
    )
    target_digest = _digest(
        target_signature_spec_digest,
        label="Target",
    )
    if preserve_manual_target_mask is None:
        preserve_mask = np.zeros(target_keys.shape[0], dtype=bool)
    else:
        preserve_mask = np.asarray(preserve_manual_target_mask, dtype=bool).reshape(-1)
        if preserve_mask.shape[0] != target_keys.shape[0]:
            raise KeyedDeltaPlanError(
                "preserve_manual_target_mask does not match target row count."
            )

    source_inputs = (
        source_instance_keys,
        source_row_signatures,
        source_signature_spec_digest,
    )
    source_declared = [value is not None for value in source_inputs]
    if any(source_declared) and not all(source_declared):
        raise KeyedDeltaPlanError(
            "Source keys, signatures, and signature spec digest must be supplied together."
        )

    target_rows = np.arange(target_keys.shape[0], dtype=np.int64)
    actions = np.full(target_keys.shape[0], ACTION_CODE_MAP["compute"], dtype=np.uint8)
    reasons = np.full(
        target_keys.shape[0],
        REASON_CODE_MAP["no_reuse_source"],
        dtype=np.uint8,
    )
    source_rows = np.full(target_keys.shape[0], -1, dtype=np.int64)
    omitted_keys = np.empty((0,), dtype=np.uint64)
    omitted_rows = np.empty((0,), dtype=np.int64)
    omitted_reasons = np.empty((0,), dtype=np.uint8)
    normalized_source_digest: str | None = None

    if all(source_declared):
        source_keys = _keys(np.asarray(source_instance_keys), label="Source")
        source_signatures = _signatures(
            np.asarray(source_row_signatures),
            row_count=int(source_keys.shape[0]),
            label="Source",
        )
        normalized_source_digest = _digest(
            source_signature_spec_digest,
            label="Source",
        )
        matched, source_rows = _source_matches(
            target_keys=target_keys,
            source_keys=source_keys,
        )
        omitted_keys, omitted_rows = _omitted_source_rows(
            target_keys=target_keys,
            source_keys=source_keys,
        )
        omitted_reasons = np.full(
            omitted_keys.shape[0],
            OMIT_REASON_CODE_MAP["deleted_from_target"],
            dtype=np.uint8,
        )
        reasons[~matched] = REASON_CODE_MAP["added"]
        if normalized_source_digest != target_digest:
            reasons[matched] = REASON_CODE_MAP["signature_spec_changed"]
        else:
            matched_target_rows = np.flatnonzero(matched)
            matched_source_rows = source_rows[matched]
            signature_equal = np.all(
                target_signatures[matched_target_rows]
                == source_signatures[matched_source_rows],
                axis=1,
            )
            reusable_target_rows = matched_target_rows[signature_equal]
            changed_target_rows = matched_target_rows[~signature_equal]
            actions[reusable_target_rows] = ACTION_CODE_MAP["copy"]
            reasons[reusable_target_rows] = REASON_CODE_MAP["unchanged"]
            reasons[changed_target_rows] = REASON_CODE_MAP["source_changed"]
            manual_rows = reusable_target_rows[preserve_mask[reusable_target_rows]]
            actions[manual_rows] = ACTION_CODE_MAP["preserve_manual"]
            reasons[manual_rows] = REASON_CODE_MAP["preserved_manual"]

    plan = KeyedDeltaPlan(
        target_instance_keys=target_keys,
        target_row_indices=target_rows,
        source_row_indices=source_rows,
        action_codes=actions,
        reason_codes=reasons,
        omitted_instance_keys=omitted_keys,
        omitted_source_row_indices=omitted_rows,
        omitted_reason_codes=omitted_reasons,
        target_signature_spec_digest=target_digest,
        source_signature_spec_digest=normalized_source_digest,
    )
    validate_keyed_delta_plan(plan)
    return plan


def validate_keyed_delta_plan(plan: KeyedDeltaPlan) -> None:
    """Fail closed unless a plan covers every target row exactly once."""

    row_count = plan.row_count
    validated_target_keys = _keys(plan.target_instance_keys, label="Target plan")
    if validated_target_keys.shape != (row_count,):
        raise KeyedDeltaPlanError("Target plan instance_key is not one-dimensional.")
    aligned = (
        plan.target_row_indices,
        plan.source_row_indices,
        plan.action_codes,
        plan.reason_codes,
    )
    if any(np.asarray(values).shape != (row_count,) for values in aligned):
        raise KeyedDeltaPlanError("Target plan arrays do not share one row axis.")
    if not np.array_equal(plan.target_row_indices, np.arange(row_count, dtype=np.int64)):
        raise KeyedDeltaPlanError("Target plan does not cover each target row exactly once.")
    if np.any(~np.isin(plan.action_codes, list(ACTION_CODE_MAP.values()))):
        raise KeyedDeltaPlanError("Target plan contains an unknown action code.")
    if np.any(~np.isin(plan.reason_codes, list(REASON_CODE_MAP.values()))):
        raise KeyedDeltaPlanError("Target plan contains an unknown reason code.")
    reusable = plan.action_codes != ACTION_CODE_MAP["compute"]
    if np.any(plan.source_row_indices[reusable] < 0):
        raise KeyedDeltaPlanError("Reusable target rows lack source row indexes.")
    valid_pairs = np.zeros(row_count, dtype=bool)
    valid_pairs |= (plan.action_codes == ACTION_CODE_MAP["copy"]) & (
        plan.reason_codes == REASON_CODE_MAP["unchanged"]
    )
    valid_pairs |= (plan.action_codes == ACTION_CODE_MAP["preserve_manual"]) & (
        plan.reason_codes == REASON_CODE_MAP["preserved_manual"]
    )
    for reason_name in (
        "added",
        "source_changed",
        "signature_spec_changed",
        "no_reuse_source",
    ):
        valid_pairs |= (plan.action_codes == ACTION_CODE_MAP["compute"]) & (
            plan.reason_codes == REASON_CODE_MAP[reason_name]
        )
    if not np.all(valid_pairs):
        raise KeyedDeltaPlanError("Target plan contains an inconsistent action/reason pair.")
    unmatched_reasons = np.isin(
        plan.reason_codes,
        [REASON_CODE_MAP["added"], REASON_CODE_MAP["no_reuse_source"]],
    )
    if np.any(plan.source_row_indices[unmatched_reasons] != -1):
        raise KeyedDeltaPlanError("New/no-source target rows unexpectedly reference source rows.")
    matched_reasons = np.isin(
        plan.reason_codes,
        [
            REASON_CODE_MAP["unchanged"],
            REASON_CODE_MAP["source_changed"],
            REASON_CODE_MAP["signature_spec_changed"],
            REASON_CODE_MAP["preserved_manual"],
        ],
    )
    if np.any(plan.source_row_indices[matched_reasons] < 0):
        raise KeyedDeltaPlanError("Matched target rows lack source row indexes.")
    matched_source_rows = plan.source_row_indices[matched_reasons]
    if np.unique(matched_source_rows).shape[0] != matched_source_rows.shape[0]:
        raise KeyedDeltaPlanError("More than one target row references the same source row.")
    omitted_count = int(plan.omitted_instance_keys.shape[0])
    if plan.omitted_source_row_indices.shape != (omitted_count,) or (
        plan.omitted_reason_codes.shape != (omitted_count,)
    ):
        raise KeyedDeltaPlanError("Omitted-row plan arrays are inconsistent.")
    if omitted_count:
        if np.unique(plan.omitted_instance_keys).shape[0] != omitted_count:
            raise KeyedDeltaPlanError("Omitted instance_key values are not unique.")
        if np.unique(plan.omitted_source_row_indices).shape[0] != omitted_count or np.any(
            plan.omitted_source_row_indices < 0
        ):
            raise KeyedDeltaPlanError("Omitted source row indexes are invalid.")
        if np.intersect1d(
            plan.target_instance_keys,
            plan.omitted_instance_keys,
            assume_unique=True,
        ).size:
            raise KeyedDeltaPlanError("A key cannot be both targeted and omitted.")
        if np.any(
            plan.omitted_reason_codes != OMIT_REASON_CODE_MAP["deleted_from_target"]
        ):
            raise KeyedDeltaPlanError("Omitted-row plan contains an unknown reason code.")


def write_keyed_delta_plan(
    group: Any,
    plan: KeyedDeltaPlan,
    *,
    shard_rows: int = DEFAULT_DELTA_PLAN_SHARD_ROWS,
) -> None:
    """Persist a plan as independently sharded, TensorStore-readable columns."""

    validate_keyed_delta_plan(plan)
    arrays = {
        "instance_key": plan.target_instance_keys,
        "target_row_index": plan.target_row_indices,
        "source_row_index": plan.source_row_indices,
        "action_codes": plan.action_codes,
        "reason_codes": plan.reason_codes,
        "omitted_instance_key": plan.omitted_instance_keys,
        "omitted_source_row_index": plan.omitted_source_row_indices,
        "omitted_reason_codes": plan.omitted_reason_codes,
    }
    for name, values in arrays.items():
        store_array(
            group,
            name,
            np.asarray(values),
            shard_rows=int(shard_rows),
        )
    group.attrs.update(
        {
            "schema_id": KEYED_DELTA_PLAN_SCHEMA_ID,
            "schema_version": KEYED_DELTA_PLAN_SCHEMA_VERSION,
            "action_code_map": dict(ACTION_CODE_MAP),
            "reason_code_map": dict(REASON_CODE_MAP),
            "omit_reason_code_map": dict(OMIT_REASON_CODE_MAP),
            "summary": plan.summary(),
            "shard_rows_requested": int(shard_rows),
        }
    )


__all__ = [
    "KEYED_DELTA_PLAN_SCHEMA_ID",
    "KEYED_DELTA_PLAN_SCHEMA_VERSION",
    "DEFAULT_DELTA_PLAN_SHARD_ROWS",
    "ACTION_CODE_MAP",
    "REASON_CODE_MAP",
    "OMIT_REASON_CODE_MAP",
    "KeyedDeltaPlanError",
    "KeyedDeltaPlan",
    "build_keyed_delta_plan",
    "validate_keyed_delta_plan",
    "write_keyed_delta_plan",
]
