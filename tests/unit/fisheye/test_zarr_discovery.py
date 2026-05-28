from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared.zarr_discovery import (
    discover_filesystem_zarrs,
    iter_filesystem_zarrs,
    load_path_list,
)


def test_iter_filesystem_zarrs_accepts_explicit_zarr_directory(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()

    assert list(iter_filesystem_zarrs([archive], recursive=False)) == [archive]


def test_iter_filesystem_zarrs_nonrecursive_matches_recording_layout(tmp_path: Path) -> None:
    direct = tmp_path / "direct_analysis.zarr"
    recording_layout = tmp_path / "recording" / "zarr" / "recording_analysis.zarr"
    nested = tmp_path / "recording" / "nested" / "ignored_analysis.zarr"
    direct.mkdir()
    recording_layout.mkdir(parents=True)
    nested.mkdir(parents=True)

    discovered = set(iter_filesystem_zarrs([tmp_path], recursive=False))

    assert discovered == {direct, recording_layout}


def test_iter_filesystem_zarrs_recursive_finds_nested_zarrs(tmp_path: Path) -> None:
    first = tmp_path / "recording" / "zarr" / "recording_analysis.zarr"
    second = tmp_path / "recording" / "nested" / "also_analysis.zarr"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    assert set(discover_filesystem_zarrs([tmp_path], recursive=True)) == {first, second}


def test_iter_filesystem_zarrs_ignores_missing_roots(tmp_path: Path) -> None:
    existing = tmp_path / "existing_analysis.zarr"
    existing.mkdir()

    discovered = list(
        iter_filesystem_zarrs([tmp_path / "missing", existing], recursive=True)
    )

    assert discovered == [existing]


def test_load_path_list_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    path_list = tmp_path / "paths.txt"
    path_list.write_text(
        "\n"
        "# comment\n"
        "  /tmp/one.zarr  \n"
        "\n"
        "/tmp/two.zarr\n",
        encoding="utf-8",
    )

    assert load_path_list(path_list) == [Path("/tmp/one.zarr"), Path("/tmp/two.zarr")]


def test_load_path_list_wraps_non_missing_read_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path_list = tmp_path / "paths.txt"

    def _raise_read_error(self: Path, *_args: object, **_kwargs: object) -> str:
        raise OSError("boom")

    monkeypatch.setattr(Path, "read_text", _raise_read_error)

    with pytest.raises(RuntimeError, match=f"Failed to read {path_list}"):
        load_path_list(path_list, wrap_errors=True)


def test_load_path_list_preserves_missing_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_path_list(tmp_path / "missing.txt", wrap_errors=True)
