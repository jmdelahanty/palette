from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fisheye.diagnostics.prepare_storage_benchmark_fixture import main
from fisheye.shared.zarr.benchmark_fixture import (
    inventory_tree,
    plan_benchmark_fixture,
    publish_benchmark_fixture,
    require_safe_fixture_destination,
)


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "disposable.zarr"
    chunk = source / "values" / "c" / "0"
    chunk.parent.mkdir(parents=True)
    (source / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}),
        encoding="utf-8",
    )
    chunk.write_bytes(b"payload")
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "purpose": "disposable_storage_benchmark",
                "destination": str(source),
                "canonical": False,
                "registry_registered": False,
                "selector_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    return source, manifest


def _thaw(root: Path) -> None:
    os.chmod(root, 0o755)
    for path in root.rglob("*"):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def test_fixture_plan_is_read_only_and_rejects_outside_destination(
    tmp_path: Path,
) -> None:
    source, source_manifest = _source_fixture(tmp_path)
    benchmark_root = tmp_path / "benchmarks"
    destination = benchmark_root / "canonical_detection_storage" / "fixtures" / "f1"

    plan = plan_benchmark_fixture(
        fixture_id="f1",
        source=source,
        source_manifest_path=source_manifest,
        destination=destination,
        benchmark_root=benchmark_root,
    )

    assert plan["status"] == "planned"
    assert plan["payload_io_performed"] is False
    assert plan["source_inventory"] == inventory_tree(source).as_manifest()
    assert not destination.exists()
    with pytest.raises(ValueError, match="must be below"):
        require_safe_fixture_destination(
            tmp_path / "outside" / "fixtures" / "f1",
            benchmark_root=benchmark_root,
        )


def test_fixture_publish_copies_exact_tree_and_freezes_it(tmp_path: Path) -> None:
    source, source_manifest = _source_fixture(tmp_path)
    source_before = inventory_tree(source)
    benchmark_root = tmp_path / "benchmarks"
    destination = benchmark_root / "canonical_detection_storage" / "fixtures" / "f1"

    result = publish_benchmark_fixture(
        fixture_id="f1",
        source=source,
        source_manifest_path=source_manifest,
        destination=destination,
        benchmark_root=benchmark_root,
    )

    try:
        copied = destination / "source.zarr"
        assert result["status"] == "published_immutable"
        assert result["exact_relative_path_size_content_match"] is True
        assert inventory_tree(source) == source_before
        assert inventory_tree(copied).tree_sha256 == source_before.tree_sha256
        assert json.loads(
            (destination / "fixture_manifest.json").read_text(encoding="utf-8")
        )["fixture_id"] == "f1"
        assert (copied.stat().st_mode & 0o777) == 0o555
        assert all(
            (path.stat().st_mode & 0o777) == (0o555 if path.is_dir() else 0o444)
            for path in destination.rglob("*")
        )
        with pytest.raises(FileExistsError, match="already exists"):
            publish_benchmark_fixture(
                fixture_id="f1",
                source=source,
                source_manifest_path=source_manifest,
                destination=destination,
                benchmark_root=benchmark_root,
            )
    finally:
        _thaw(destination)


def test_fixture_rejects_canonical_source_manifest(tmp_path: Path) -> None:
    source, source_manifest = _source_fixture(tmp_path)
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["canonical"] = True
    source_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical=false"):
        plan_benchmark_fixture(
            fixture_id="f1",
            source=source,
            source_manifest_path=source_manifest,
            destination=(
                tmp_path
                / "benchmarks"
                / "canonical_detection_storage"
                / "fixtures"
                / "f1"
            ),
            benchmark_root=tmp_path / "benchmarks",
        )


def test_fixture_cli_defaults_to_plan_only(tmp_path: Path, capsys) -> None:
    source, source_manifest = _source_fixture(tmp_path)
    benchmark_root = tmp_path / "benchmarks"
    destination = benchmark_root / "canonical_detection_storage" / "fixtures" / "f1"

    assert (
        main(
            [
                str(source),
                str(destination),
                "--source-manifest",
                str(source_manifest),
                "--benchmark-root",
                str(benchmark_root),
                "--fixture-id",
                "f1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    assert not destination.exists()
