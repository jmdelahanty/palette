from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.metadata_equivalence import (
    ZarrMetadataEquivalenceError,
    validate_direct_consolidated_subtree,
)


def _archive(path: Path, *, consolidate: bool = True) -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    run = root.require_group("analysis/example_runs/run_1")
    run.attrs["schema_id"] = "palette.example"
    nested = run.require_group("nested")
    nested.attrs["meaning"] = "exact"
    run.create_array(
        "values",
        data=np.arange(8, dtype=np.int32),
        chunks=(4,),
    )
    nested.create_array(
        "valid",
        data=np.asarray([True, False], dtype=np.bool_),
        chunks=(2,),
    )
    if consolidate:
        zarr.consolidate_metadata(str(path))
    return path


def _root_document(path: Path) -> dict[str, object]:
    return json.loads((path / "zarr.json").read_text(encoding="utf-8"))


def _write_root_document(path: Path, value: object) -> None:
    (path / "zarr.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_exact_subtree_direct_and_consolidated_metadata_pass(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "analysis.zarr")

    receipt = validate_direct_consolidated_subtree(
        archive,
        subtree_path="analysis/example_runs/run_1",
    )

    assert receipt.node_count == 4
    assert receipt.group_count == 2
    assert receipt.array_count == 2
    assert len(receipt.declarations_sha256) == 64
    assert receipt.to_json()["subtree_path"] == "analysis/example_runs/run_1"


def test_missing_consolidated_metadata_fails_closed(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "analysis.zarr", consolidate=False)

    with pytest.raises(
        ZarrMetadataEquivalenceError,
        match="no persisted consolidated_metadata",
    ):
        validate_direct_consolidated_subtree(
            archive,
            subtree_path="analysis/example_runs/run_1",
        )


def test_stale_direct_array_declaration_fails_closed(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "analysis.zarr")
    metadata_path = archive / "analysis/example_runs/run_1/values/zarr.json"
    direct = json.loads(metadata_path.read_text(encoding="utf-8"))
    direct["attributes"]["tampered"] = True
    metadata_path.write_text(json.dumps(direct), encoding="utf-8")

    with pytest.raises(
        ZarrMetadataEquivalenceError,
        match="declaration differs",
    ):
        validate_direct_consolidated_subtree(
            archive,
            subtree_path="analysis/example_runs/run_1",
        )


def test_unexpected_inline_descendant_fails_closed(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "analysis.zarr")
    root = _root_document(archive)
    envelope = root["consolidated_metadata"]
    assert isinstance(envelope, dict)
    metadata = envelope["metadata"]
    assert isinstance(metadata, dict)
    metadata["analysis/example_runs/run_1/foreign"] = deepcopy(
        metadata["analysis/example_runs/run_1/nested"]
    )
    _write_root_document(archive, root)

    with pytest.raises(
        ZarrMetadataEquivalenceError,
        match="unexpected_inline",
    ):
        validate_direct_consolidated_subtree(
            archive,
            subtree_path="analysis/example_runs/run_1",
        )


def test_array_cannot_claim_group_consolidation_envelope(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "analysis.zarr")
    root = _root_document(archive)
    envelope = root["consolidated_metadata"]
    assert isinstance(envelope, dict)
    metadata = envelope["metadata"]
    assert isinstance(metadata, dict)
    array = metadata["analysis/example_runs/run_1/values"]
    assert isinstance(array, dict)
    array["consolidated_metadata"] = None
    _write_root_document(archive, root)

    with pytest.raises(
        ZarrMetadataEquivalenceError,
        match="Only Zarr groups",
    ):
        validate_direct_consolidated_subtree(
            archive,
            subtree_path="analysis/example_runs/run_1",
        )


def test_symlink_inside_selected_subtree_is_rejected(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "analysis.zarr")
    outside = tmp_path / "outside"
    outside.mkdir()
    (archive / "analysis/example_runs/run_1/linked").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ZarrMetadataEquivalenceError, match="contains a symlink"):
        validate_direct_consolidated_subtree(
            archive,
            subtree_path="analysis/example_runs/run_1",
        )


@pytest.mark.parametrize("subtree", ("", "/", "../outside", "a//b"))
def test_noncanonical_subtree_path_is_rejected(
    tmp_path: Path,
    subtree: str,
) -> None:
    archive = _archive(tmp_path / "analysis.zarr")

    with pytest.raises(ValueError, match="canonical subtree"):
        validate_direct_consolidated_subtree(
            archive,
            subtree_path=subtree,
        )
