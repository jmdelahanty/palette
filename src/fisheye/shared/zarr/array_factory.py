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


def array_metadata_declaration_from_plan(
    *,
    contract: ArrayContract,
    plan: StoragePlan,
    fill_value: Any,
    attributes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the exact Zarr-v3 declaration expected from the array factory."""

    codec_profile = get_codec_profile(plan.codec_profile_id)
    supplied_attributes = dict(attributes or {})
    conflicts = sorted(_RESERVED_ATTRIBUTES.intersection(supplied_attributes))
    if conflicts:
        raise ValueError(
            f"Array attributes cannot override reserved storage keys: {conflicts}."
        )
    serializer: dict[str, object] = {"name": codec_profile.serializer_name}
    if np.dtype(plan.logical_dtype).itemsize > 1:
        serializer["configuration"] = {"endian": codec_profile.serializer_endian}
    data_codecs: list[dict[str, object]] = [
        serializer,
        {
            "name": codec_profile.compressor_name,
            "configuration": {
                "level": codec_profile.compression_level,
                "checksum": codec_profile.checksum,
            },
        },
    ]
    if plan.shard_shape is None:
        codecs: list[dict[str, object]] = data_codecs
        outer_chunk_shape = plan.chunk_shape
    else:
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": list(plan.chunk_shape or ()),
                    "codecs": data_codecs,
                    "index_codecs": [
                        {
                            "name": codec_profile.shard_index_serializer_name,
                            "configuration": {
                                "endian": (codec_profile.shard_index_serializer_endian)
                            },
                        },
                        {"name": codec_profile.shard_index_checksum_name},
                    ],
                    "index_location": codec_profile.shard_index_location,
                },
            }
        ]
        outer_chunk_shape = plan.shard_shape
    return {
        "shape": list(plan.logical_shape),
        "data_type": str(np.dtype(plan.logical_dtype)),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(outer_chunk_shape or ())},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": {
            **supplied_attributes,
            "logical_schema_id": contract.schema_id,
            "logical_schema_version": contract.schema_version,
            "storage_policy_version": plan.policy_version,
            "storage_profile_id": plan.profile_id,
            "codec_profile_id": codec_profile.profile_id,
            "access_pattern": plan.access_pattern,
            "write_mode": plan.write_mode,
        },
        "dimension_names": list(contract.axis_names),
        "storage_transformers": [],
    }


def validate_array_metadata_declaration_from_plan(
    declaration: Mapping[str, Any],
    *,
    contract: ArrayContract,
    plan: StoragePlan,
    fill_value: Any,
) -> tuple[str, ...]:
    """Validate physical Zarr metadata against one resolved storage plan."""

    attributes = declaration.get("attributes")
    if not isinstance(attributes, Mapping):
        return ("array metadata attributes must be an object",)
    nonreserved_attributes = {
        str(key): value
        for key, value in attributes.items()
        if key not in _RESERVED_ATTRIBUTES
    }
    expected = array_metadata_declaration_from_plan(
        contract=contract,
        plan=plan,
        fill_value=fill_value,
        attributes=nonreserved_attributes,
    )
    observed = {
        key: value
        for key, value in declaration.items()
        if key not in {"zarr_format", "node_type", "consolidated_metadata"}
    }
    if observed != expected:
        return ("array metadata differs from the resolved storage plan",)
    return ()


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


__all__ = [
    "array_metadata_declaration_from_plan",
    "create_array_from_plan",
    "validate_array_metadata_declaration_from_plan",
]
