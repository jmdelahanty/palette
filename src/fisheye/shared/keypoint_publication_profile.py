"""Declared publication profiles for recording-level keypoint products."""

from __future__ import annotations


STRICT_V2_KEYPOINT_PUBLICATION_PROFILE = "strict_v2"
COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE = "compatibility_ordinary_v1"
KEYPOINT_PUBLICATION_PROFILES = (
    STRICT_V2_KEYPOINT_PUBLICATION_PROFILE,
    COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE,
)


def require_compatibility_keypoint_shard_aggregate(profile: object) -> str:
    """Reject use of the ordinary shard aggregator as a canonical producer."""

    value = str(profile or "").strip()
    if value != COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE:
        raise ValueError(
            "finalize_keypoint_shards is a compatibility-only producer and cannot "
            "satisfy strict-v2 canonical keypoint publication. Pass the exact "
            f"compatibility profile {COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE!r} "
            "only for an explicitly noncanonical workflow."
        )
    return value


__all__ = [
    "COMPATIBILITY_KEYPOINT_SHARD_AGGREGATE_PROFILE",
    "KEYPOINT_PUBLICATION_PROFILES",
    "STRICT_V2_KEYPOINT_PUBLICATION_PROFILE",
    "require_compatibility_keypoint_shard_aggregate",
]
