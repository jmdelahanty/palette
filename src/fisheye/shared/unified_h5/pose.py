"""Citrus pose observations (``pose_observations``), contract v1.

The component is optional per recording and required once declared
(agent-contracts ``citrus-pose-observation-v1``, merged 49142e6). Declared
means one ``required`` outcome row, the ``/observations/pose`` group and its
manifest receipt; absent means none of the three. The schemas and table
descriptors are pinned by digest in ``fisheye/shared/contracts``.

A declared pose that failed already fails the whole file (``session_status``
FAILED), which the integrity layer refuses; this layer never admits a file
with pose dropped.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from importlib.resources import files as resource_files
import json

import numpy as np

from .common import parse_json, require
from .hdf5_types import dataset_bytes

COMPONENT = "pose_observations"
GROUP = "/observations/pose"
RECEIPT = GROUP + "/receipt_json"
CONTRACT = GROUP + "/contract_json"
TABLES = ("updates", "objects", "keypoints")
_PINNED = {
    "contract": (
        "citrus_pose_observation_contract.v1.schema.json",
        "e9a2d50a40995f4955d61d50d256e9b10f0d9c76f9908c4d1df3218e9e1af1e7",
    ),
    "receipt": (
        "citrus_pose_observation_receipt.v1.schema.json",
        "2fde5432880d0f9feae18b3c1823e54256efb0514a21cfafb41f1c7a1c30fdc7",
    ),
    "tables": (
        "citrus_pose_observation_tables.v1.json",
        "72820be7efd3aff7633799481ec961609e6750f3868d6a915f3a2925497a2329",
    ),
}


@lru_cache(maxsize=None)
def _pinned(name: str) -> dict:
    file_name, digest = _PINNED[name]
    data = resource_files("fisheye.shared").joinpath("contracts").joinpath(file_name).read_bytes()
    require(sha256(data).hexdigest() == digest, "packaged_contract_drift:" + file_name)
    return json.loads(data)


@lru_cache(maxsize=None)
def _validator(name: str):
    from jsonschema import Draft202012Validator

    return Draft202012Validator(_pinned(name), format_checker=Draft202012Validator.FORMAT_CHECKER)


def _document(h5, path: str, schema: str) -> dict:
    dataset = h5[path]
    require(dataset.shape == () and dataset.dtype.kind == "O", f"pose_json_not_scalar_string:{path}")
    document = parse_json(dataset_bytes(dataset), label=path)
    error = next(iter(_validator(schema).iter_errors(document)), None)
    require(error is None, f"pose_{schema}_schema:{error.message[:160] if error else ''}")
    return document


def _text(value):
    """String attributes as text; any other type stays distinct (never "1" == 1)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else ("<non-string>", repr(value))


def _table(h5, name: str, descriptor: dict):
    dataset = h5[f"{GROUP}/{name}"]
    dtype = dataset.dtype
    fields = descriptor["fields"]
    require(
        dataset.ndim == 1
        and dtype.itemsize == descriptor["row_bytes"]
        and dtype.names == tuple(field["name"] for field in fields)
        and all(
            dtype.fields[f["name"]][1] == f["offset_bytes"]
            and dtype.fields[f["name"]][0] == np.dtype(f["numpy_dtype"])
            for f in fields
        ),
        f"pose_table_layout_mismatch:{name}",
    )
    attributes = {key: _text(value) for key, value in dataset.attrs.items()}
    require(attributes == descriptor["attributes"], f"pose_table_attributes_mismatch:{name}")
    return dataset


def _column(dataset, name: str) -> np.ndarray:
    return np.asarray(dataset.fields(name)[:], dtype=np.uint64)


def _check_row_references(updates, objects, keypoints) -> None:
    """Each update owns a contiguous run of objects, each object of keypoints."""

    for owner, start_field, count_field, child, back_field in (
        (updates, "objects_start", "logged_object_count", objects, "update_row"),
        (objects, "keypoints_start", "keypoint_count", keypoints, "object_row"),
    ):
        starts, counts = _column(owner, start_field), _column(owner, count_field)
        expected_starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.uint64)
        owning = counts > 0  # a row that owns nothing may record any start
        require(
            np.array_equal(starts[owning], expected_starts[owning])
            and int(counts.sum()) == child.shape[0],
            f"pose_row_offsets_not_conserved:{child.name}",
        )
        back = _column(child, back_field)
        require(
            np.array_equal(back, np.repeat(np.arange(owner.shape[0], dtype=np.uint64), counts.astype(np.int64))),
            f"pose_row_owner_mismatch:{child.name}",
        )
    require(
        int(_column(objects, "keypoint_count").max(initial=0)) <= 16,
        "pose_keypoint_count_exceeds_v1_limit",
    )


def validate_pose_observations(h5, outcomes: dict, receipt_components: set) -> int | None:
    """Total pose rows when the component is declared and complete, else None.

    ``outcomes`` maps component id to its outcome row; ``receipt_components``
    names the component receipts the dependency manifest covers.
    """

    declared = (COMPONENT in outcomes, GROUP in h5, COMPONENT in receipt_components)
    if not any(declared):
        return None
    require(all(declared), "pose_declaration_incomplete")
    outcome = outcomes[COMPONENT]
    require(outcome["requirement"] == "required", "pose_outcome_not_required")
    _document(h5, CONTRACT, "contract")
    receipt = _document(h5, RECEIPT, "receipt")
    require(receipt["status"] == "complete", "pose_receipt_not_complete")
    descriptors = _pinned("tables")["tables"]
    tables = {name: _table(h5, name, descriptors[name]) for name in TABLES}
    for name, dataset in tables.items():
        require(
            receipt[f"{name}_expected"] == receipt[f"{name}_written"] == dataset.shape[0],
            f"pose_receipt_count_mismatch:{name}",
        )
    total = sum(dataset.shape[0] for dataset in tables.values())
    require(
        outcome["status"] == "complete"
        and outcome["expected_rows"] == outcome["written_rows"] == total
        and outcome["dropped_rows"] == 0,
        "pose_outcome_count_mismatch",
    )
    _check_row_references(tables["updates"], tables["objects"], tables["keypoints"])
    return total


__all__ = ["COMPONENT", "RECEIPT", "validate_pose_observations"]
