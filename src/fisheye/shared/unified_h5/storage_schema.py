"""Lossless native byte surfaces using Palette's existing physical planner."""

from fisheye.shared.zarr.array_contracts import ArrayContract, UINT8
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import get_storage_profile

STORAGE_SCHEMA = "palette.unified_h5_native_storage"
STORAGE_VERSION = 1
PAYLOAD_CONTRACT = ArrayContract(
    schema_id="palette.unified_h5_native_payload",
    schema_version=1,
    dtype=UINT8,
    shape_template=("n_bytes",),
    axis_names=("byte",),
    description="Exact native packed-LE bytes, or uint64-length-prefixed native variable strings.",
)
MANIFEST_CONTRACT = ArrayContract(
    schema_id="palette.unified_h5_native_manifest",
    schema_version=1,
    dtype=UINT8,
    shape_template=("n_utf8_bytes",),
    axis_names=("byte",),
    description="Canonical UTF-8 native type/attribute/payload inventory and source admission bindings.",
)


def payload_plan(size, contract=PAYLOAD_CONTRACT):
    intent = contract.storage_intent(
        shape=(size,),
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        whole_shard_writes=False,
    )
    # Regular, bounded local chunks. A single importer owns all physical writes;
    # no Dask workers or partial-shard ownership assumptions are introduced.
    return plan_storage(intent, get_storage_profile("scratch_compute_v1"))


def create_bytes(group, name, size, *, contract=PAYLOAD_CONTRACT):
    return create_array_from_plan(
        group,
        name=name,
        contract=contract,
        plan=payload_plan(size, contract),
        fill_value=0,
    )
