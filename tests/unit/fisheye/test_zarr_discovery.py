from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared.zarr_discovery import (
    discover_filesystem_zarrs,
    discover_registry_zarr_entries,
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


def test_iter_filesystem_zarrs_accepts_explicit_zarr_file(tmp_path: Path) -> None:
    archive = tmp_path / "recording_training.zarr"
    archive.write_text("placeholder", encoding="utf-8")

    assert list(iter_filesystem_zarrs([archive], recursive=False)) == [archive]
    assert (
        list(
            iter_filesystem_zarrs(
                [archive],
                recursive=False,
                include_zarr_files=False,
            )
        )
        == []
    )


def test_iter_filesystem_zarrs_dedupes_by_resolved_path(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    alias = tmp_path / "alias_analysis.zarr"
    try:
        alias.symlink_to(archive, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not support directory symlinks")

    assert list(iter_filesystem_zarrs([archive, alias], recursive=False)) == [archive]
    assert list(iter_filesystem_zarrs([archive, alias], recursive=False, dedupe=False)) == [
        archive,
        alias,
    ]


def test_iter_filesystem_zarrs_can_preserve_under_zarr_dir_policy(tmp_path: Path) -> None:
    direct = tmp_path / "direct_analysis.zarr"
    recording_layout = tmp_path / "recording" / "zarr" / "recording_analysis.zarr"
    direct.mkdir()
    recording_layout.mkdir(parents=True)

    discovered = set(
        iter_filesystem_zarrs(
            [tmp_path],
            recursive=False,
            pattern_policy="under_zarr_dir",
        )
    )

    assert discovered == {recording_layout}


def test_iter_filesystem_zarrs_can_require_zarr_root_metadata(tmp_path: Path) -> None:
    empty = tmp_path / "empty_analysis.zarr"
    valid = tmp_path / "valid_analysis.zarr"
    empty.mkdir()
    valid.mkdir()
    (valid / "zarr.json").write_text("{}", encoding="utf-8")

    discovered = list(
        iter_filesystem_zarrs(
            [tmp_path],
            recursive=False,
            require_zarr_root=True,
        )
    )

    assert discovered == [valid]


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


def test_registry_discovery_filters_protocol_name_case_insensitively(
    tmp_path: Path,
) -> None:
    class _FakeRegistry:
        def __init__(self, _path: Path) -> None:
            self.closed = False

        def query_datasets(self, **_kwargs: object) -> list[dict[str, object]]:
            return [
                {
                    "zarr_path": str(tmp_path / "keep_analysis.zarr"),
                    "camera_id": "1",
                    "protocol_name": "GoodCopBadCop",
                },
                {
                    "zarr_path": str(tmp_path / "drop_analysis.zarr"),
                    "camera_id": "2",
                    "protocol_name": "RedScare",
                },
            ]

        def close(self) -> None:
            self.closed = True

    entries = discover_registry_zarr_entries(
        registry_path=tmp_path / "registry.sqlite",
        scope_paths=[],
        protocol_name="goodcopbadcop",
        registry_cls=_FakeRegistry,
    )

    assert [entry.zarr_path.name for entry in entries] == ["keep_analysis.zarr"]


def test_registry_discovery_excludes_missing_protocol_when_filtering(
    tmp_path: Path,
) -> None:
    class _FakeRegistry:
        def __init__(self, _path: Path) -> None:
            pass

        def query_datasets(self, **_kwargs: object) -> list[dict[str, object]]:
            return [
                {
                    "zarr_path": str(tmp_path / "unknown_analysis.zarr"),
                    "camera_id": None,
                    "protocol_name": None,
                }
            ]

        def close(self) -> None:
            pass

    entries = discover_registry_zarr_entries(
        registry_path=tmp_path / "registry.sqlite",
        scope_paths=[],
        protocol_name="GoodCopBadCop",
        registry_cls=_FakeRegistry,
    )

    assert entries == []


def test_registry_discovery_can_select_normalized_chaser_capability(
    tmp_path: Path,
) -> None:
    class _Cursor:
        def fetchall(self) -> list[tuple[str]]:
            return [("with-chaser",)]

    class _Connection:
        def execute(self, _sql: str) -> _Cursor:
            return _Cursor()

    class _FakeRegistry:
        def __init__(self, _path: Path) -> None:
            self.conn = _Connection()

        def query_datasets(self, **_kwargs: object) -> list[dict[str, object]]:
            return [
                {
                    "dataset_id": "with-chaser",
                    "zarr_path": str(tmp_path / "keep_analysis.zarr"),
                    "camera_id": "1",
                    "protocol_name": "AnyChaserProtocol",
                },
                {
                    "dataset_id": "without-chaser",
                    "zarr_path": str(tmp_path / "drop_analysis.zarr"),
                    "camera_id": "2",
                    "protocol_name": "GoodCopBadCop",
                },
            ]

        def close(self) -> None:
            pass

    entries = discover_registry_zarr_entries(
        registry_path=tmp_path / "registry.sqlite",
        scope_paths=[],
        require_chaser_metadata=True,
        registry_cls=_FakeRegistry,
    )

    assert [entry.zarr_path.name for entry in entries] == ["keep_analysis.zarr"]
