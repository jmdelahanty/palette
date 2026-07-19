"""Exact keyed base-plus-replacement planning for immutable snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from fisheye.shared.keyed_delta import ACTION_CODE_MAP, KeyedDeltaPlan, build_keyed_delta_plan


KEYED_REPLACEMENT_SCHEMA_ID = "palette.keyed_replacement_plan"
KEYED_REPLACEMENT_SCHEMA_VERSION = 1
REPLACEMENT_SOURCE_BASE = np.int16(-1)


class KeyedReplacementError(ValueError):
    """Raised when replacement rows cannot complete one exact target rowset."""


@dataclass(frozen=True)
class KeyedReplacementPlan:
    """Map every target row to one base row or one replacement-run row."""

    delta_plan: KeyedDeltaPlan
    source_run_indices: np.ndarray
    source_row_indices: np.ndarray

    @property
    def target_row_count(self) -> int:
        return int(self.delta_plan.target_instance_keys.shape[0])

    @property
    def replacement_target_rows(self) -> np.ndarray:
        return np.flatnonzero(self.source_run_indices != REPLACEMENT_SOURCE_BASE).astype(
            np.int64, copy=False
        )

    def summary(self) -> dict[str, Any]:
        replacement_counts = {
            str(index): int(np.count_nonzero(self.source_run_indices == index))
            for index in sorted(
                int(value)
                for value in np.unique(self.source_run_indices)
                if int(value) >= 0
            )
        }
        return {
            "schema_id": KEYED_REPLACEMENT_SCHEMA_ID,
            "schema_version": KEYED_REPLACEMENT_SCHEMA_VERSION,
            "target_row_count": self.target_row_count,
            "base_row_count": int(
                np.count_nonzero(self.source_run_indices == REPLACEMENT_SOURCE_BASE)
            ),
            "replacement_row_count": int(self.replacement_target_rows.shape[0]),
            "replacement_counts_by_run_index": replacement_counts,
            "delta_plan": self.delta_plan.summary(),
        }


def _unique_keys(values: np.ndarray, *, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise KeyedReplacementError(f"{label} instance_key values must be integers.")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise KeyedReplacementError(f"{label} instance_key values must be nonnegative.")
    keys = np.asarray(raw, dtype=np.uint64).reshape(-1)
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise KeyedReplacementError(f"{label} instance_key values must be unique.")
    return keys


def _match_rows(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    order = np.argsort(haystack, kind="stable")
    sorted_keys = haystack[order]
    positions = np.searchsorted(sorted_keys, needles)
    if np.any(positions >= sorted_keys.shape[0]):
        raise KeyedReplacementError("Replacement contains a key outside the target rowset.")
    matched = sorted_keys[positions] == needles
    if not np.all(matched):
        raise KeyedReplacementError("Replacement contains a key outside the target rowset.")
    return order[positions].astype(np.int64, copy=False)


def build_keyed_replacement_plan(
    *,
    target_instance_keys: np.ndarray,
    target_source_signatures: np.ndarray,
    target_signature_spec_digest: str,
    base_instance_keys: np.ndarray,
    base_source_signatures: np.ndarray,
    base_signature_spec_digest: str,
    replacement_instance_keys: Sequence[np.ndarray],
) -> KeyedReplacementPlan:
    """Require replacements for exactly the target rows unsafe to reuse.

    Replacement runs may be supplied in any order and their rows may be
    reordered.  A key may occur in only one replacement run.  Extra rerun rows
    are rejected as well as missing changed/new rows, keeping publication
    deterministic and making accidental mixed inference packages visible.
    """

    delta_plan = build_keyed_delta_plan(
        target_instance_keys=np.asarray(target_instance_keys),
        target_source_signatures=np.asarray(target_source_signatures),
        target_signature_spec_digest=target_signature_spec_digest,
        source_instance_keys=np.asarray(base_instance_keys),
        source_row_signatures=np.asarray(base_source_signatures),
        source_signature_spec_digest=base_signature_spec_digest,
    )
    target_keys = _unique_keys(delta_plan.target_instance_keys, label="Target")
    expected_rows = np.flatnonzero(
        delta_plan.action_codes == ACTION_CODE_MAP["compute"]
    ).astype(np.int64, copy=False)
    expected_keys = target_keys[expected_rows]

    source_run_indices = np.full(
        target_keys.shape[0], REPLACEMENT_SOURCE_BASE, dtype=np.int16
    )
    source_row_indices = np.asarray(delta_plan.source_row_indices, dtype=np.int64).copy()
    all_replacement_keys: list[np.ndarray] = []
    for run_index, values in enumerate(replacement_instance_keys):
        keys = _unique_keys(np.asarray(values), label=f"Replacement {run_index}")
        target_rows = _match_rows(target_keys, keys)
        if np.any(source_run_indices[target_rows] != REPLACEMENT_SOURCE_BASE):
            raise KeyedReplacementError(
                "One instance_key occurs in more than one replacement run."
            )
        source_run_indices[target_rows] = np.int16(run_index)
        source_row_indices[target_rows] = np.arange(keys.shape[0], dtype=np.int64)
        all_replacement_keys.append(keys)

    supplied_keys = (
        np.concatenate(all_replacement_keys)
        if all_replacement_keys
        else np.empty((0,), dtype=np.uint64)
    )
    if supplied_keys.shape[0] != expected_keys.shape[0] or not np.array_equal(
        np.sort(supplied_keys), np.sort(expected_keys)
    ):
        missing = np.setdiff1d(expected_keys, supplied_keys, assume_unique=False)
        extra = np.setdiff1d(supplied_keys, expected_keys, assume_unique=False)
        raise KeyedReplacementError(
            "Replacement keys do not exactly equal the rows requiring computation: "
            f"missing={int(missing.shape[0])}, extra={int(extra.shape[0])}."
        )
    if expected_rows.size and np.any(source_run_indices[expected_rows] < 0):
        raise KeyedReplacementError("A required replacement row has no replacement source.")
    reusable = delta_plan.action_codes != ACTION_CODE_MAP["compute"]
    if np.any(source_run_indices[reusable] != REPLACEMENT_SOURCE_BASE):
        raise KeyedReplacementError("An unchanged row was unexpectedly replaced.")
    if np.any(source_row_indices < 0):
        raise KeyedReplacementError("The replacement plan leaves a target row unresolved.")
    return KeyedReplacementPlan(
        delta_plan=delta_plan,
        source_run_indices=source_run_indices,
        source_row_indices=source_row_indices,
    )


__all__ = [
    "KEYED_REPLACEMENT_SCHEMA_ID",
    "KEYED_REPLACEMENT_SCHEMA_VERSION",
    "REPLACEMENT_SOURCE_BASE",
    "KeyedReplacementError",
    "KeyedReplacementPlan",
    "build_keyed_replacement_plan",
]
