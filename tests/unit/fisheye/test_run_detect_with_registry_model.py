from __future__ import annotations

from pathlib import Path

import pytest
import zarr

from fisheye.utils import run_detect_with_registry_model as mod


def test_pick_best_candidate_enforces_unique_when_tied() -> None:
    candidates = [
        mod.Candidate(
            run_id="run_a",
            set_id="detect_a",
            model_path="/tmp/a.pt",
            created_utc="2026-02-09T00:00:00+00:00",
            status="success",
            dataset_count=10,
            weighted_score=0.9,
            feature_match_counts={"rig_id": 10},
            feature_weights_used=1.0,
        ),
        mod.Candidate(
            run_id="run_b",
            set_id="detect_b",
            model_path="/tmp/b.pt",
            created_utc="2026-02-09T00:00:01+00:00",
            status="success",
            dataset_count=10,
            weighted_score=0.9,
            feature_match_counts={"rig_id": 10},
            feature_weights_used=1.0,
        ),
    ]
    with pytest.raises(SystemExit, match="Top candidate score tied"):
        mod._pick_best_candidate(candidates, require_unique=True)  # noqa: SLF001


def test_write_model_resolution_provenance_updates_detect_run_attrs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.require_group("detect_runs")
    detect_run = detect_parent.require_group("detect_20260209_000000")
    detect_run.attrs["provenance"] = {"stage": "detect"}

    payload = {
        "mode": "registry",
        "task": "detect",
        "registry_path": "/nvme1/palette_registry.sqlite",
        "recording_id": "2026-01-28T19-22-28Z_arena_1",
        "resolved_at_utc": "2026-02-09T09:00:00+00:00",
        "selected": {
            "run_id": "detect_run_001",
            "set_id": "detect_set_001",
            "model_path": "/nvme1/models/detect/model.pt",
            "score": 0.88,
            "created_utc": "2026-02-09T08:59:00+00:00",
        },
        "candidates": [{"run_id": "detect_run_001", "score": 0.88}],
    }

    mod._write_model_resolution_provenance(  # noqa: SLF001
        zarr_path=zarr_path,
        run_name="detect_20260209_000000",
        payload=payload,
    )

    root2 = zarr.open_group(str(zarr_path), mode="r")
    detect2 = root2["detect_runs"]["detect_20260209_000000"]
    assert detect2.attrs.get("model_resolution_mode") == "registry"
    assert detect2.attrs.get("model_resolution_task") == "detect"
    assert detect2.attrs.get("model_resolution_selected_run_id") == "detect_run_001"
    assert detect2.attrs.get("model_resolution_selected_set_id") == "detect_set_001"
    provenance = detect2.attrs.get("provenance")
    assert isinstance(provenance, dict)
    assert "model_resolution" in provenance
