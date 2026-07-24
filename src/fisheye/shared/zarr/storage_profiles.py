"""Named, versioned byte budgets for Palette Zarr storage planning."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fisheye.shared.zarr.storage_intent import AccessPattern


KIB = 1024
MIB = 1024 * KIB


@dataclass(frozen=True)
class StorageProfile:
    """Byte and object-count budgets for a class of stored artifacts."""

    profile_id: str
    target_chunk_bytes: int
    min_chunk_bytes: int
    max_chunk_bytes: int
    eager_max_bytes: int
    target_shard_bytes: int
    per_row_target_shard_bytes: int
    max_shard_bytes: int
    max_payload_objects: int
    codec_profile_id: str
    shard_immutable: bool = True
    shard_owned_appends: bool = True

    def __post_init__(self) -> None:
        positive_fields = {
            "target_chunk_bytes": self.target_chunk_bytes,
            "min_chunk_bytes": self.min_chunk_bytes,
            "max_chunk_bytes": self.max_chunk_bytes,
            "eager_max_bytes": self.eager_max_bytes,
            "target_shard_bytes": self.target_shard_bytes,
            "per_row_target_shard_bytes": self.per_row_target_shard_bytes,
            "max_shard_bytes": self.max_shard_bytes,
            "max_payload_objects": self.max_payload_objects,
        }
        for name, value in positive_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}.")
        if self.min_chunk_bytes > self.target_chunk_bytes:
            raise ValueError("min_chunk_bytes cannot exceed target_chunk_bytes.")
        if self.target_chunk_bytes > self.max_chunk_bytes:
            raise ValueError("target_chunk_bytes cannot exceed max_chunk_bytes.")
        if self.target_shard_bytes > self.max_shard_bytes:
            raise ValueError("target_shard_bytes cannot exceed max_shard_bytes.")
        if self.per_row_target_shard_bytes > self.max_shard_bytes:
            raise ValueError(
                "per_row_target_shard_bytes cannot exceed max_shard_bytes."
            )

    def shard_target_bytes(self, access: AccessPattern) -> int:
        """Return the outer-shard target for an access class."""

        if access is AccessPattern.PER_ROW:
            return int(self.per_row_target_shard_bytes)
        return int(self.target_shard_bytes)

    def as_manifest(self) -> dict[str, object]:
        """Return the exact JSON-safe byte and object budgets."""

        return {
            "schema_id": "palette.storage_profile",
            "schema_version": 1,
            "profile_id": self.profile_id,
            "target_chunk_bytes": self.target_chunk_bytes,
            "min_chunk_bytes": self.min_chunk_bytes,
            "max_chunk_bytes": self.max_chunk_bytes,
            "eager_max_bytes": self.eager_max_bytes,
            "target_shard_bytes": self.target_shard_bytes,
            "per_row_target_shard_bytes": self.per_row_target_shard_bytes,
            "max_shard_bytes": self.max_shard_bytes,
            "max_payload_objects": self.max_payload_objects,
            "codec_profile_id": self.codec_profile_id,
            "shard_immutable": self.shard_immutable,
            "shard_owned_appends": self.shard_owned_appends,
        }


SCRATCH_COMPUTE_V1 = StorageProfile(
    profile_id="scratch_compute_v1",
    target_chunk_bytes=1 * MIB,
    min_chunk_bytes=512 * KIB,
    max_chunk_bytes=2 * MIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=32 * MIB,
    per_row_target_shard_bytes=128 * MIB,
    max_shard_bytes=512 * MIB,
    max_payload_objects=16_384,
    codec_profile_id="zstd_fast_v1",
    shard_immutable=False,
    shard_owned_appends=False,
)

EDITABLE_LOCAL_V1 = StorageProfile(
    profile_id="editable_local_v1",
    target_chunk_bytes=1 * MIB,
    min_chunk_bytes=512 * KIB,
    max_chunk_bytes=2 * MIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=32 * MIB,
    per_row_target_shard_bytes=128 * MIB,
    max_shard_bytes=512 * MIB,
    max_payload_objects=16_384,
    codec_profile_id="zstd_fast_v1",
    shard_immutable=False,
    shard_owned_appends=False,
)

PUBLISHED_HTTP_V1 = StorageProfile(
    profile_id="published_http_v1",
    target_chunk_bytes=1 * MIB,
    min_chunk_bytes=512 * KIB,
    max_chunk_bytes=2 * MIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=32 * MIB,
    per_row_target_shard_bytes=256 * MIB,
    max_shard_bytes=512 * MIB,
    max_payload_objects=4_096,
    codec_profile_id="zstd_fast_v1",
)

TRAINING_IMMUTABLE_V1 = StorageProfile(
    profile_id="training_immutable_v1",
    target_chunk_bytes=1 * MIB,
    min_chunk_bytes=512 * KIB,
    max_chunk_bytes=2 * MIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=32 * MIB,
    per_row_target_shard_bytes=128 * MIB,
    max_shard_bytes=512 * MIB,
    max_payload_objects=4_096,
    codec_profile_id="zstd_fast_v1",
)


STORAGE_PROFILES = {
    profile.profile_id: profile
    for profile in (
        SCRATCH_COMPUTE_V1,
        EDITABLE_LOCAL_V1,
        PUBLISHED_HTTP_V1,
        TRAINING_IMMUTABLE_V1,
    )
}


def get_storage_profile(profile_id: str) -> StorageProfile:
    """Resolve a named profile or raise a useful error."""

    try:
        return STORAGE_PROFILES[str(profile_id)]
    except KeyError as exc:
        choices = ", ".join(sorted(STORAGE_PROFILES))
        raise ValueError(
            f"Unknown storage profile {profile_id!r}; expected one of: {choices}."
        ) from exc


def make_benchmark_storage_profile(
    *,
    base: StorageProfile = PUBLISHED_HTTP_V1,
    target_chunk_bytes: int,
    target_shard_bytes: int,
    shard_immutable: bool,
) -> StorageProfile:
    """Derive one exact byte-sweep candidate without row-count constants."""

    chunk_bytes = int(target_chunk_bytes)
    shard_bytes = int(target_shard_bytes)
    if chunk_bytes <= 0 or shard_bytes <= 0:
        raise ValueError("Benchmark chunk and shard byte targets must be positive.")
    if shard_bytes < chunk_bytes:
        raise ValueError("Benchmark shard target cannot be smaller than chunk target.")
    layout = "sharded" if shard_immutable else "regular"
    profile_id = (
        f"{base.profile_id}__benchmark_{layout}"
        f"__chunk_{chunk_bytes}__shard_{shard_bytes}"
    )
    return replace(
        base,
        profile_id=profile_id,
        target_chunk_bytes=chunk_bytes,
        min_chunk_bytes=chunk_bytes,
        max_chunk_bytes=chunk_bytes,
        target_shard_bytes=shard_bytes,
        per_row_target_shard_bytes=shard_bytes,
        max_shard_bytes=shard_bytes,
        shard_immutable=bool(shard_immutable),
    )
