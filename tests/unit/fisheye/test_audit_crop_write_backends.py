from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.utils.audit_crop_write_backends import _collect_rows, _summarize


def _make_archive(
    root: Path,
    recording: str,
    zarr_name: str,
    *,
    zarr_purpose: str | None,
) -> tuple[Path, zarr.Group]:
    zarr_path = root / recording / "zarr" / zarr_name
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(zarr_path), mode="w")
    if zarr_purpose is not None:
        group.attrs["zarr_purpose"] = zarr_purpose
    return zarr_path, group


def test_collect_rows_latest_only_and_summary(tmp_path: Path) -> None:
    zarr_path, root = _make_archive(tmp_path, "rec_a", "rec_a_analysis.zarr", zarr_purpose="analysis")
    crop_parent = root.create_group("crop_runs")
    old = crop_parent.create_group("crop_old")
    old.attrs["write_backend"] = "standard_zarr"
    old.attrs["video_source_type"] = "external"
    new = crop_parent.create_group("crop_new")
    new.attrs["write_backend_requested"] = "kvikio"
    new.attrs["write_backend_effective"] = "kvikio_gds"
    new.attrs["write_backend"] = "kvikio_gds"
    new.attrs["video_source_type"] = "external"
    crop_parent.attrs["latest"] = "crop_new"

    rows = _collect_rows([tmp_path], recursive=True, zarr_use_filter="analysis", latest_only=True)
    assert len(rows) == 1
    assert rows[0].zarr_path == str(zarr_path)
    assert rows[0].crop_run == "crop_new"
    assert rows[0].requested_backend == "kvikio"
    assert rows[0].effective_backend == "kvikio_gds"

    summary = _summarize(rows)
    assert summary["total_runs"] == 1
    assert summary["effective_backend_counts"] == {"kvikio_gds": 1}


def test_collect_rows_infers_requested_backend_for_legacy_attrs(tmp_path: Path) -> None:
    _zarr_path, root = _make_archive(tmp_path, "rec_a", "rec_a_analysis.zarr", zarr_purpose="analysis")
    crop_parent = root.create_group("crop_runs")
    run = crop_parent.create_group("crop_legacy")
    run.attrs["write_backend"] = "standard_zarr"
    crop_parent.attrs["latest"] = "crop_legacy"

    rows = _collect_rows([tmp_path], recursive=True, zarr_use_filter="analysis", latest_only=True)
    assert len(rows) == 1
    assert rows[0].requested_backend == "standard"
    assert rows[0].effective_backend == "standard_zarr"


def test_collect_rows_zarr_use_filter(tmp_path: Path) -> None:
    _make_archive(tmp_path, "rec_a", "rec_a_analysis.zarr", zarr_purpose="analysis")
    _make_archive(tmp_path, "rec_b", "rec_b_training.zarr", zarr_purpose="training")

    # add one crop run per archive
    for zarr_path in tmp_path.rglob("zarr/*.zarr"):
        root = zarr.open_group(str(zarr_path), mode="a")
        crop_parent = root.create_group("crop_runs")
        run = crop_parent.create_group("crop_1")
        run.attrs["write_backend"] = "standard_zarr"
        crop_parent.attrs["latest"] = "crop_1"

    rows_analysis = _collect_rows([tmp_path], recursive=True, zarr_use_filter="analysis", latest_only=True)
    rows_training = _collect_rows([tmp_path], recursive=True, zarr_use_filter="training", latest_only=True)
    rows_any = _collect_rows([tmp_path], recursive=True, zarr_use_filter="any", latest_only=True)

    assert len(rows_analysis) == 1
    assert len(rows_training) == 1
    assert len(rows_any) == 2
