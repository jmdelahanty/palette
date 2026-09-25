"""Plan and apply the sharded physical layout of one training/review run.

The shared byte planner (``plan_storage``) resolves each array's outer shard
from the ``training_review_run_v1`` profile while keeping the producer's
existing inner chunk shape as the declared access unit. The existing
``copy_completed_run_to_sharded`` performs the copy with exact decoded-value
validation. Logical values, attrs, and row identity are unchanged; only the
physical layout (and the recorded layout attrs) differ.

Ownership: the run is rewritten by exactly one writer in a private scratch
location before publication. Editable review runs are later mutated only by
the serialized review writer (the run lock), so each shard has one writer.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.storage_intent import (
    AccessPattern,
    ArrayIntent,
    StoragePlan,
    WriteMode,
)
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import (
    TRAINING_REVIEW_RUN_V1,
    StorageProfile,
)
from fisheye.shared.zarr_sharded_copy import (
    SHARD_POLICY_MULTI_CHUNK_CAPPED,
    ShardedArrayLayout,
    copy_completed_run_to_sharded,
)

REVIEW_STORAGE_PLAN_ATTR = "review_storage_plan"
REVIEW_STORAGE_PLAN_SCHEMA = "palette.training_review_run_storage_plan.v1"


def _iter_arrays(group: zarr.Group, prefix: str = ""):
    for name, array in sorted(group.arrays(), key=lambda item: item[0]):
        yield (f"{prefix}/{name}" if prefix else str(name)), array
    for name, child in sorted(group.groups(), key=lambda item: item[0]):
        yield from _iter_arrays(child, f"{prefix}/{name}" if prefix else str(name))


def plan_training_review_run(
    run: zarr.Group,
    *,
    mutable: bool,
    profile: StorageProfile = TRAINING_REVIEW_RUN_V1,
) -> dict[str, StoragePlan]:
    """Resolve one plan per array; inner chunks stay the producer's chunks."""

    plans: dict[str, StoragePlan] = {}
    for path, array in _iter_arrays(run):
        dtype = np.dtype(array.dtype)
        shape = tuple(int(value) for value in array.shape)
        if dtype.hasobject or dtype.kind in "VUSO" or not shape or 0 in shape:
            continue  # Left exactly as written (no fixed-width row layout).
        chunks = tuple(
            min(int(chunk), dimension) for chunk, dimension in zip(array.chunks, shape)
        )
        plans[path] = plan_storage(
            ArrayIntent(
                shape=shape,
                dtype=dtype,
                access=AccessPattern.PER_ROW,
                write_mode=(
                    WriteMode.RANDOM_UPDATE if mutable else WriteMode.IMMUTABLE
                ),
                access_unit_shape=chunks,
                name=path,
            ),
            profile,
        )
    return plans


def _plan_record(profile: StorageProfile, mutable: bool, plans) -> dict[str, Any]:
    return {
        "schema_id": REVIEW_STORAGE_PLAN_SCHEMA,
        "storage_profile": profile.as_manifest(),
        "write_mode": (
            WriteMode.RANDOM_UPDATE if mutable else WriteMode.IMMUTABLE
        ).value,
        "arrays": {
            path: {
                "requested_chunk_shape": list(plan.access_unit_shape),
                "chunk_shape": list(plan.chunk_shape or ()),
                "shard_shape": list(plan.shard_shape) if plan.shard_shape else None,
                "write_ownership": plan.write_ownership,
                "estimated_payload_objects": plan.estimated_payload_objects,
            }
            for path, plan in sorted(plans.items())
        },
    }


def shard_training_review_run(
    run_path: str | Path,
    *,
    mutable: bool,
    profile: StorageProfile = TRAINING_REVIEW_RUN_V1,
) -> Mapping[str, Any]:
    """Rewrite one complete, privately owned local run into its planned shards.

    The caller must be the only writer of ``run_path`` (a scratch payload
    before atomic publication). Returns the sharded-copy report.
    """

    run_path = Path(run_path).resolve()
    run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
    plans = plan_training_review_run(run, mutable=mutable, profile=profile)
    for path, plan in plans.items():
        if tuple(plan.chunk_shape or ()) != tuple(plan.access_unit_shape):
            raise RuntimeError(f"Review storage plan changed inner chunks: {path}")
    layouts = {
        path: ShardedArrayLayout(
            outer_shards=plan.shard_shape, layout_profile=profile.profile_id
        )
        for path, plan in plans.items()
        if plan.shard_shape is not None
    }
    record = _plan_record(profile, mutable, plans)
    staging = run_path.with_name(run_path.name + ".sharding")
    report = copy_completed_run_to_sharded(
        run_path,
        staging,
        row_count_array=None,
        shard_rows=max((plan.shard_shape[0] for plan in plans.values() if plan.shard_shape), default=1),
        array_layouts=layouts,
        shard_policy=SHARD_POLICY_MULTI_CHUNK_CAPPED,
        workers=1,
    )
    staged = zarr.open_group(str(staging), mode="r+", use_consolidated=False)
    staged.attrs[REVIEW_STORAGE_PLAN_ATTR] = record
    shutil.rmtree(run_path)
    staging.rename(run_path)
    return report
