from __future__ import annotations

from pathlib import Path

from scripts import check_zarr_open_group_modes as mod


def _write_module(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_ast_census_distinguishes_explicit_metadata_modes(tmp_path: Path) -> None:
    source = tmp_path / "src/fisheye"
    _write_module(
        source / "reader.py",
        "import zarr\n"
        "def read(path):\n"
        "    zarr.open_group(path, mode='r', use_consolidated=True)\n"
        "    zarr.open_group(path, mode='r', consolidated=False)\n"
        "    zarr.open_group(path, mode='r')\n",
    )

    calls = mod.collect_bare_open_group_calls(source, repo_root=tmp_path)

    assert len(calls) == 1
    assert calls[0].relative_path == "src/fisheye/reader.py"
    assert calls[0].symbol == "read"
    assert calls[0].line == 5


def test_ratchet_rejects_added_bare_calls_and_tightens_removed_calls(tmp_path: Path) -> None:
    source = tmp_path / "src/fisheye"
    module = source / "reader.py"
    baseline = tmp_path / "baseline.json"
    _write_module(
        module,
        "import zarr\ndef read(path):\n    return zarr.open_group(path, mode='r')\n",
    )
    mod._write_baseline(
        baseline,
        mod.collect_bare_open_group_calls(source, repo_root=tmp_path),
    )

    assert mod.check_zarr_open_group_modes(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 0

    _write_module(
        module,
        "import zarr\n"
        "def read(path):\n"
        "    first = zarr.open_group(path, mode='r')\n"
        "    return zarr.open_group(path, mode='a')\n",
    )
    assert mod.check_zarr_open_group_modes(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 1

    _write_module(
        module,
        "import zarr\n"
        "def read(path):\n"
        "    return zarr.open_group(path, mode='r', use_consolidated=True)\n",
    )
    assert mod.check_zarr_open_group_modes(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 0
    assert mod._read_baseline(baseline) == {}
