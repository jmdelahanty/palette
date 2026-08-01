"""Named, versioned byte budgets for Palette Zarr storage planning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from fisheye.shared.zarr.storage_intent import AccessPattern


KIB = 1024
MIB = 1024 * KIB


def _normalize_chunk_targets_by_access(
    values: Mapping[AccessPattern | str, int]
    | tuple[tuple[AccessPattern | str, int], ...],
) -> tuple[tuple[str, int], ...]:
    normalized: dict[str, int] = {}
    items = values.items() if isinstance(values, Mapping) else values
    for raw_access, raw_target in items:
        access = AccessPattern(raw_access).value
        if access in normalized:
            raise ValueError(
                f"Duplicate chunk-byte override for access pattern {access!r}."
            )
        target = int(raw_target)
        if target <= 0:
            raise ValueError(
                "Access-specific target chunk bytes must be positive; "
                f"got {raw_target!r} for {access!r}."
            )
        normalized[access] = target
    return tuple(
        (access.value, normalized[access.value])
        for access in AccessPattern
        if access.value in normalized
    )


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
    target_chunk_bytes_by_access: tuple[tuple[str, int], ...] = ()

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
        object.__setattr__(
            self,
            "target_chunk_bytes_by_access",
            _normalize_chunk_targets_by_access(
                self.target_chunk_bytes_by_access
            ),
        )

    def chunk_byte_budget(self, access: AccessPattern) -> tuple[int, int, int]:
        """Return ``(target, minimum, maximum)`` for one access class.

        Access-specific targets are exact benchmark/profile decisions. Arrays
        without an override retain the profile's ordinary target and bounds.
        """

        resolved = AccessPattern(access).value
        overrides = dict(self.target_chunk_bytes_by_access)
        if resolved in overrides:
            target = int(overrides[resolved])
            return target, target, target
        return (
            int(self.target_chunk_bytes),
            int(self.min_chunk_bytes),
            int(self.max_chunk_bytes),
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
            "schema_version": 2,
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
            "target_chunk_bytes_by_access": dict(
                self.target_chunk_bytes_by_access
            ),
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

# Promoted from the full-duration canonical and refined-detection gates.  The
# narrow random/windowed payload uses 128 KiB inner chunks, while the retained
# eager frame index keeps 1 MiB inner chunks.  Both live in 8 MiB indexed
# shards when the concrete array is large enough to benefit from sharding.
DETECTION_PUBLISHED_ACCESS_AWARE_V1 = StorageProfile(
    profile_id="detection_published_access_aware_v1",
    target_chunk_bytes=128 * KIB,
    min_chunk_bytes=128 * KIB,
    max_chunk_bytes=128 * KIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=8 * MIB,
    per_row_target_shard_bytes=8 * MIB,
    max_shard_bytes=8 * MIB,
    max_payload_objects=4_096,
    codec_profile_id="zstd_fast_v1",
    shard_immutable=True,
    shard_owned_appends=True,
    target_chunk_bytes_by_access=((AccessPattern.EAGER, 1 * MIB),),
)

# Candidate presentation-cache profile for row-addressed subject-mask display
# products such as fixed-count sampled contours.  The 128 KiB inner target
# bounds one-frame read amplification, while 8 MiB indexed shards keep the
# immutable recording-level object count low.  This profile remains a
# selector-ineligible candidate until a real Crimson mounted-read gate promotes
# it; the semantic contract does not depend on that later promotion decision.
SUBJECT_MASK_PRESENTATION_CANDIDATE_V1 = StorageProfile(
    profile_id="subject_mask_presentation_candidate_v1",
    target_chunk_bytes=128 * KIB,
    min_chunk_bytes=128 * KIB,
    max_chunk_bytes=128 * KIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=8 * MIB,
    per_row_target_shard_bytes=8 * MIB,
    max_shard_bytes=8 * MIB,
    max_payload_objects=4_096,
    codec_profile_id="zstd_fast_v1",
    shard_immutable=True,
    shard_owned_appends=True,
)

# Explicit rollback for immutable detection snapshots.  This is the genuine
# 1 MiB unsharded benchmark control, not the generic sharded HTTP profile.
DETECTION_REGULAR_ROLLBACK_V1 = StorageProfile(
    profile_id="detection_regular_rollback_v1",
    target_chunk_bytes=1 * MIB,
    min_chunk_bytes=1 * MIB,
    max_chunk_bytes=1 * MIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=32 * MIB,
    per_row_target_shard_bytes=32 * MIB,
    max_shard_bytes=32 * MIB,
    max_payload_objects=4_096,
    codec_profile_id="zstd_fast_v1",
    shard_immutable=False,
    shard_owned_appends=True,
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
        DETECTION_PUBLISHED_ACCESS_AWARE_V1,
        SUBJECT_MASK_PRESENTATION_CANDIDATE_V1,
        DETECTION_REGULAR_ROLLBACK_V1,
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


def storage_profile_from_manifest(value: Mapping[str, Any]) -> StorageProfile:
    """Parse one exact v2 profile and enforce registered-profile identity."""

    expected_fields = {
        "schema_id",
        "schema_version",
        "profile_id",
        "target_chunk_bytes",
        "min_chunk_bytes",
        "max_chunk_bytes",
        "eager_max_bytes",
        "target_shard_bytes",
        "per_row_target_shard_bytes",
        "max_shard_bytes",
        "max_payload_objects",
        "codec_profile_id",
        "shard_immutable",
        "shard_owned_appends",
        "target_chunk_bytes_by_access",
    }
    if set(value) != expected_fields:
        raise ValueError("storage_profile has an unexpected field set")
    if (
        value.get("schema_id") != "palette.storage_profile"
        or value.get("schema_version") != 2
    ):
        raise ValueError("storage_profile schema identity mismatch")
    integer_fields = (
        "target_chunk_bytes",
        "min_chunk_bytes",
        "max_chunk_bytes",
        "eager_max_bytes",
        "target_shard_bytes",
        "per_row_target_shard_bytes",
        "max_shard_bytes",
        "max_payload_objects",
    )
    if any(type(value.get(name)) is not int for name in integer_fields):
        raise TypeError("storage_profile byte/object budgets must be exact integers")
    if (
        type(value.get("shard_immutable")) is not bool
        or type(value.get("shard_owned_appends")) is not bool
    ):
        raise TypeError("storage_profile shard flags must be exact booleans")
    profile_id = value.get("profile_id")
    codec_profile_id = value.get("codec_profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("storage profile_id cannot be empty")
    if not isinstance(codec_profile_id, str) or not codec_profile_id.strip():
        raise ValueError("codec_profile_id cannot be empty")
    overrides = value.get("target_chunk_bytes_by_access")
    if not isinstance(overrides, Mapping):
        raise TypeError("target_chunk_bytes_by_access must be an object")
    if any(
        type(key) is not str or type(target) is not int
        for key, target in overrides.items()
    ):
        raise TypeError(
            "target_chunk_bytes_by_access requires string keys and integer values"
        )
    profile = StorageProfile(
        profile_id=profile_id.strip(),
        target_chunk_bytes=value["target_chunk_bytes"],
        min_chunk_bytes=value["min_chunk_bytes"],
        max_chunk_bytes=value["max_chunk_bytes"],
        eager_max_bytes=value["eager_max_bytes"],
        target_shard_bytes=value["target_shard_bytes"],
        per_row_target_shard_bytes=value["per_row_target_shard_bytes"],
        max_shard_bytes=value["max_shard_bytes"],
        max_payload_objects=value["max_payload_objects"],
        codec_profile_id=codec_profile_id.strip(),
        shard_immutable=value["shard_immutable"],
        shard_owned_appends=value["shard_owned_appends"],
        target_chunk_bytes_by_access=tuple(overrides.items()),
    )
    if profile.as_manifest() != dict(value):
        raise ValueError("storage_profile is not in canonical persisted form")
    registered = STORAGE_PROFILES.get(profile.profile_id)
    if registered is not None and profile != registered:
        raise ValueError(
            f"registered storage profile {profile.profile_id!r} differs from "
            "its frozen definition"
        )
    return profile


def make_benchmark_storage_profile(
    *,
    base: StorageProfile = PUBLISHED_HTTP_V1,
    target_chunk_bytes: int,
    target_shard_bytes: int,
    shard_immutable: bool,
    target_chunk_bytes_by_access: Mapping[AccessPattern | str, int] | None = None,
) -> StorageProfile:
    """Derive one exact byte-sweep candidate without row-count constants."""

    chunk_bytes = int(target_chunk_bytes)
    shard_bytes = int(target_shard_bytes)
    if chunk_bytes <= 0 or shard_bytes <= 0:
        raise ValueError("Benchmark chunk and shard byte targets must be positive.")
    if shard_bytes < chunk_bytes:
        raise ValueError("Benchmark shard target cannot be smaller than chunk target.")
    layout = "sharded" if shard_immutable else "regular"
    normalized_overrides = _normalize_chunk_targets_by_access(
        target_chunk_bytes_by_access or {}
    )
    largest_chunk_target = max(
        (chunk_bytes, *(value for _access, value in normalized_overrides))
    )
    if shard_immutable and shard_bytes < largest_chunk_target:
        raise ValueError(
            "Benchmark shard target cannot be smaller than any chunk target."
        )
    access_suffix = "".join(
        f"__{access}_chunk_{value}"
        for access, value in normalized_overrides
    )
    profile_id = (
        f"{base.profile_id}__benchmark_{layout}"
        f"__chunk_{chunk_bytes}{access_suffix}__shard_{shard_bytes}"
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
        target_chunk_bytes_by_access=normalized_overrides,
    )
