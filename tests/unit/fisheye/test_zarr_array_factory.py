from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.array_contracts import DETECTION_FRAME_INDICES_V1
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.codec_profiles import ZSTD_FAST_V1
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


class _FakeGroup:
    def __init__(self) -> None:
        self.name = None
        self.kwargs = None
        self.metadata = SimpleNamespace(zarr_format=3)

    def create_array(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        return object()


def _plan(shape: int = 1_000_000):
    intent = DETECTION_FRAME_INDICES_V1.storage_intent(
        shape=(shape,),
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        access_unit_shape=(1,),
        shard_axes=(0,),
        name="instances/frame_indices",
        dimensions={"n_instances": shape},
    )
    return plan_storage(intent, PUBLISHED_HTTP_V1)


def test_codec_profile_is_exact_and_json_safe() -> None:
    manifest = ZSTD_FAST_V1.as_manifest()

    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["zarr_format"] == 3
    assert manifest["codec_chain"] == [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {
            "name": "zstd",
            "configuration": {"level": 0, "checksum": False},
        },
    ]
    assert manifest["sharding_index"] == {
        "codec_chain": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "crc32c"},
        ],
        "location": "end",
    }


def test_array_factory_translates_exact_plan_without_raw_layout_choices() -> None:
    group = _FakeGroup()
    plan = _plan()

    create_array_from_plan(
        group,
        name="frame_indices",
        contract=DETECTION_FRAME_INDICES_V1,
        plan=plan,
        fill_value=0,
    )

    assert group.name == "frame_indices"
    assert group.kwargs["shape"] == (1_000_000,)
    assert group.kwargs["dtype"] == np.dtype("int32")
    assert group.kwargs["chunks"] == plan.shard_shape
    assert group.kwargs["dimension_names"] == ("instance",)
    assert group.kwargs["compressors"] is None
    assert group.kwargs["filters"] is None
    serializer = group.kwargs["serializer"]
    assert isinstance(serializer, zarr.codecs.ShardingCodec)
    assert serializer.chunk_shape == plan.chunk_shape
    assert isinstance(serializer.codecs[0], zarr.codecs.BytesCodec)
    assert isinstance(serializer.codecs[1], zarr.codecs.ZstdCodec)
    assert isinstance(serializer.index_codecs[0], zarr.codecs.BytesCodec)
    assert isinstance(serializer.index_codecs[1], zarr.codecs.Crc32cCodec)
    assert serializer.index_location.value == "end"
    assert group.kwargs["attributes"]["logical_schema_id"] == (
        "palette.array.detection.frame_indices"
    )


def test_array_factory_constructs_exact_regular_codec_chain() -> None:
    group = _FakeGroup()
    plan = _plan(1_000)

    create_array_from_plan(
        group,
        name="frame_indices",
        contract=DETECTION_FRAME_INDICES_V1,
        plan=plan,
        fill_value=0,
    )

    assert plan.shard_shape is None
    assert group.kwargs["chunks"] == plan.chunk_shape
    assert group.kwargs["filters"] is None
    assert isinstance(group.kwargs["serializer"], zarr.codecs.BytesCodec)
    assert isinstance(group.kwargs["compressors"][0], zarr.codecs.ZstdCodec)


def test_array_factory_rejects_path_names_and_reserved_attribute_overrides() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="path component"):
        create_array_from_plan(
            object(),
            name="instances/frame_indices",
            contract=DETECTION_FRAME_INDICES_V1,
            plan=plan,
            fill_value=0,
        )
    with pytest.raises(ValueError, match="reserved storage keys"):
        create_array_from_plan(
            object(),
            name="frame_indices",
            contract=DETECTION_FRAME_INDICES_V1,
            plan=plan,
            fill_value=0,
            attributes={"storage_profile_id": "override"},
        )


def test_array_factory_rejects_non_v3_destination_group() -> None:
    group = SimpleNamespace(metadata=SimpleNamespace(zarr_format=2))

    with pytest.raises(ValueError, match="requires a Zarr v3"):
        create_array_from_plan(
            group,
            name="frame_indices",
            contract=DETECTION_FRAME_INDICES_V1,
            plan=_plan(),
            fill_value=0,
        )
