"""The policy-owned Zarr v3 array-creation boundary."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.array_contracts import ArrayContract
from fisheye.shared.zarr.codec_profiles import CodecProfile, get_codec_profile
from fisheye.shared.zarr.storage_intent import StoragePlan


_RESERVED_ATTRIBUTES = frozenset(
    {
        "logical_schema_id",
        "logical_schema_version",
        "storage_policy_version",
        "storage_profile_id",
        "codec_profile_id",
        "access_pattern",
        "write_mode",
    }
)


def _data_codecs(profile: CodecProfile) -> tuple[Any, ...]:
    if profile.profile_id != "zstd_fast_v1":
        raise ValueError(f"Codec profile {profile.profile_id!r} has no Zarr adapter.")
    return (
        zarr.codecs.BytesCodec(endian=profile.serializer_endian),
        zarr.codecs.ZstdCodec(
            level=profile.compression_level,
            checksum=profile.checksum,
        ),
    )


def _encoding_kwargs(
    profile: CodecProfile,
    plan: StoragePlan,
) -> dict[str, object]:
    data_codecs = _data_codecs(profile)
    if plan.shard_shape is None:
        return {
            "chunks": plan.chunk_shape,
            "serializer": data_codecs[0],
            "compressors": list(data_codecs[1:]),
            "filters": None,
        }
    sharding_codec = zarr.codecs.ShardingCodec(
        chunk_shape=plan.chunk_shape,
        codecs=data_codecs,
        index_codecs=(
            zarr.codecs.BytesCodec(endian=profile.shard_index_serializer_endian),
            zarr.codecs.Crc32cCodec(),
        ),
        index_location=profile.shard_index_location,
    )
    return {
        "chunks": plan.shard_shape,
        "serializer": sharding_codec,
        "compressors": None,
        "filters": None,
    }


def create_array_from_plan(
    group: Any,
    *,
    name: str,
    contract: ArrayContract,
    plan: StoragePlan,
    fill_value: Any,
    attributes: Mapping[str, object] | None = None,
) -> Any:
    """Create one fresh Zarr v3 array from an exact logical and physical plan."""

    leaf_name = str(name).strip()
    if not leaf_name or "/" in leaf_name:
        raise ValueError("Array factory name must be one nonempty path component.")
    if plan.chunk_shape is None:
        raise ValueError("The shared array factory does not yet support scalars.")
    if (
        plan.logical_schema_id != contract.schema_id
        or plan.logical_schema_version != contract.schema_version
    ):
        raise ValueError("Storage plan logical identity does not match contract.")
    expected_dtype = contract.dtype.numpy_dtype
    if expected_dtype is None:
        raise ValueError("Variable-width contracts require a dedicated array factory.")
    if np.dtype(plan.logical_dtype) != np.dtype(expected_dtype):
        raise ValueError("Storage plan dtype does not match logical contract.")
    shape_errors = contract.validate_shape(plan.logical_shape)
    if shape_errors:
        raise ValueError(
            "Storage plan shape does not match logical contract: "
            + "; ".join(shape_errors)
        )

    supplied_attributes = dict(attributes or {})
    conflicts = sorted(_RESERVED_ATTRIBUTES.intersection(supplied_attributes))
    if conflicts:
        raise ValueError(
            f"Array attributes cannot override reserved storage keys: {conflicts}."
        )
    codec_profile = get_codec_profile(plan.codec_profile_id)
    group_zarr_format = getattr(getattr(group, "metadata", None), "zarr_format", None)
    if group_zarr_format != codec_profile.zarr_format:
        raise ValueError(
            "Shared array factory requires a Zarr "
            f"v{codec_profile.zarr_format} destination group."
        )
    array_attributes = {
        **supplied_attributes,
        "logical_schema_id": contract.schema_id,
        "logical_schema_version": contract.schema_version,
        "storage_policy_version": plan.policy_version,
        "storage_profile_id": plan.profile_id,
        "codec_profile_id": codec_profile.profile_id,
        "access_pattern": plan.access_pattern,
        "write_mode": plan.write_mode,
    }
    kwargs: dict[str, object] = {
        "shape": plan.logical_shape,
        "dtype": np.dtype(expected_dtype),
        "fill_value": fill_value,
        "attributes": array_attributes,
        "dimension_names": contract.axis_names,
        "overwrite": False,
        **_encoding_kwargs(codec_profile, plan),
    }
    return group.create_array(leaf_name, **kwargs)


__all__ = ["create_array_from_plan"]
