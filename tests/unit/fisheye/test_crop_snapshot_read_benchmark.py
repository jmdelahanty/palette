from __future__ import annotations

import json
from pathlib import Path

from fisheye.diagnostics.benchmark_crop_snapshot_reads import (
    benchmark_crop_snapshot_reads,
)
from tests.unit.fisheye.test_crop_consumer import _strict_crop


def test_crop_read_benchmark_verifies_manifest_hashes_and_reopen_modes(
    tmp_path: Path,
) -> None:
    crop = _strict_crop(tmp_path)
    archive = Path(str(crop.store.root)).parent.parent
    output = tmp_path / "benchmark.json"

    result = benchmark_crop_snapshot_reads(
        archive,
        run_id="strict_crop",
        repetitions=2,
        batch_rows=2,
        window_rows=2,
        evict_cache=False,
        output_json=output,
    )

    assert result["status"] == "passed"
    assert len(result["passes"]) == 2
    assert all(
        len(item["full_scan"]["array_sha256"]) == 13 for item in result["passes"]
    )
    assert all(len(item["windows"]) == 3 for item in result["passes"])
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
