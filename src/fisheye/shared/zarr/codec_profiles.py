"""Exact, versioned codec identities used by shared storage profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodecProfile:
    """Zarr-format, serializer, and compressor metadata contract."""

    profile_id: str
    zarr_format: int
    serializer_name: str
    serializer_endian: str
    compressor_name: str
    compression_level: int
    checksum: bool
    shard_index_serializer_name: str
    shard_index_serializer_endian: str
    shard_index_checksum_name: str
    shard_index_location: str

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("Codec profile ID cannot be empty.")
        if self.zarr_format != 3:
            raise ValueError("Shared codec profiles currently require Zarr v3.")
        if self.serializer_name != "bytes":
            raise ValueError("Unsupported serializer in shared codec profile.")
        if self.serializer_endian not in {"little", "big"}:
            raise ValueError("Byte serializer endian must be little or big.")
        if self.compressor_name != "zstd":
            raise ValueError("Unsupported compressor in shared codec profile.")
        if type(self.compression_level) is not int:
            raise TypeError("compression_level must be an exact integer.")
        if type(self.checksum) is not bool:
            raise TypeError("checksum must be an exact boolean.")
        if self.shard_index_serializer_name != "bytes":
            raise ValueError("Unsupported shard-index serializer.")
        if self.shard_index_serializer_endian not in {"little", "big"}:
            raise ValueError("Shard-index endian must be little or big.")
        if self.shard_index_checksum_name != "crc32c":
            raise ValueError("Unsupported shard-index checksum codec.")
        if self.shard_index_location not in {"start", "end"}:
            raise ValueError("Shard-index location must be start or end.")

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.codec_profile",
            "schema_version": 1,
            "profile_id": self.profile_id,
            "zarr_format": self.zarr_format,
            "codec_chain": [
                {
                    "name": self.serializer_name,
                    "configuration": {"endian": self.serializer_endian},
                },
                {
                    "name": self.compressor_name,
                    "configuration": {
                        "level": self.compression_level,
                        "checksum": self.checksum,
                    },
                },
            ],
            "sharding_index": {
                "codec_chain": [
                    {
                        "name": self.shard_index_serializer_name,
                        "configuration": {"endian": self.shard_index_serializer_endian},
                    },
                    {"name": self.shard_index_checksum_name},
                ],
                "location": self.shard_index_location,
            },
        }


ZSTD_FAST_V1 = CodecProfile(
    profile_id="zstd_fast_v1",
    zarr_format=3,
    serializer_name="bytes",
    serializer_endian="little",
    compressor_name="zstd",
    compression_level=0,
    checksum=False,
    shard_index_serializer_name="bytes",
    shard_index_serializer_endian="little",
    shard_index_checksum_name="crc32c",
    shard_index_location="end",
)


CODEC_PROFILES = {ZSTD_FAST_V1.profile_id: ZSTD_FAST_V1}


def get_codec_profile(profile_id: str) -> CodecProfile:
    try:
        return CODEC_PROFILES[str(profile_id)]
    except KeyError as exc:
        choices = ", ".join(sorted(CODEC_PROFILES))
        raise ValueError(
            f"Unknown codec profile {profile_id!r}; expected one of: {choices}."
        ) from exc


__all__ = [
    "CODEC_PROFILES",
    "ZSTD_FAST_V1",
    "CodecProfile",
    "get_codec_profile",
]
