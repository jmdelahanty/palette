from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import zarr
from zarr.errors import UnstableSpecificationWarning

from fisheye.utils.audit_zarr_string_encodings import audit_archive
from fisheye.utils.backfill_zarr_string_encodings import main


def _create_legacy_archive(zarr_path: Path) -> None:
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["zarr_purpose"] = "training"
    source_index = root.require_group("source_index")

    # Legacy fixed-width arrays that we intentionally rewrite.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        dataset_ids = source_index.create_array(
            "source_dataset_id",
            data=np.array(["dataset_a", "dataset_b"], dtype="<U16"),
            chunks=(2,),
        )
        source_index.create_array(
            "source_zarr_path",
            data=np.array(["/a.zarr", "/b.zarr"], dtype="<U32"),
            chunks=(2,),
        )
        # Fixed unicode outside allowlist should not be rewritten.
        source_index.create_array(
            "other_label",
            data=np.array(["left", "right"], dtype="<U8"),
            chunks=(2,),
        )
    dataset_ids.attrs["note"] = "keep-me"


def test_backfill_string_encodings_dry_run_reports_and_does_not_mutate(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "legacy_dry_run.zarr"
    _create_legacy_archive(zarr_path)

    before = audit_archive(zarr_path, zarr_use_filter="any")
    assert before["counts"]["fixed_unicode"] == 3

    rc = main([str(zarr_path)])
    assert rc == 0

    after = audit_archive(zarr_path, zarr_use_filter="any")
    assert after["counts"]["fixed_unicode"] == 3
    assert after["counts"]["vlen_utf8"] == 0

    text = capsys.readouterr().out
    assert "rewritable=2" in text
    assert "rewritten=0" in text


def test_backfill_string_encodings_apply_rewrites_allowlisted_paths_only(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "legacy_apply.zarr"
    _create_legacy_archive(zarr_path)

    rc = main([str(zarr_path), "--apply"])
    assert rc == 0

    report = audit_archive(zarr_path, zarr_use_filter="any")
    assert report["counts"]["fixed_unicode"] == 1
    assert report["counts"]["vlen_utf8"] == 2

    root = zarr.open_group(store=zarr_path, mode="r")
    source_index = root["source_index"]
    assert source_index["source_dataset_id"].attrs.get("note") == "keep-me"
    assert np.asarray(source_index["source_dataset_id"][:], dtype=object).tolist() == ["dataset_a", "dataset_b"]

    # Idempotent second apply.
    rc2 = main([str(zarr_path), "--apply"])
    assert rc2 == 0
    text = capsys.readouterr().out
    assert "rewritten=0" in text
