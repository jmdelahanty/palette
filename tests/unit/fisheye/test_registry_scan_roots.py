"""Tests for registry recursive zarr root discovery."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import _find_zarr_roots


def test_find_zarr_roots_ignores_nested_zarr_json(tmp_path: Path) -> None:
    dataset_a = tmp_path / "a.zarr"
    dataset_a.mkdir(parents=True)
    (dataset_a / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}', encoding="utf-8")

    nested_array = dataset_a / "raw_video" / "images_ds"
    nested_array.mkdir(parents=True)
    (nested_array / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}', encoding="utf-8")

    dataset_b = tmp_path / "b.zarr"
    dataset_b.mkdir(parents=True)
    (dataset_b / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")

    roots = _find_zarr_roots(tmp_path)
    root_paths = {path.resolve() for path in roots}

    assert root_paths == {dataset_a.resolve(), dataset_b.resolve()}
