from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows import (
    historical_protocol_semantic_stimulus_successor as successor,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _semantic_documents() -> tuple[str, str, str]:
    semantic = {
        "identity": {
            "iti_stimulus_mode_id": 99,
            "steps": [
                {
                    "duration": {"scale": "1e-3", "unit": "s", "value": 1000},
                    "parameters": {"color_type_id": 0},
                    "post_stimulus_iti": {
                        "scale": "1e-3",
                        "unit": "s",
                        "value": 0,
                    },
                    "stimulus_mode_id": 4,
                },
                {
                    "duration": {"scale": "1e-3", "unit": "s", "value": 2000},
                    "parameters": {},
                    "post_stimulus_iti": {
                        "scale": "1e-3",
                        "unit": "s",
                        "value": 0,
                    },
                    "stimulus_mode_id": 12,
                },
            ],
        },
        "normalization_policy": "citrus.protocol.semantic.v1",
        "schema_id": "citrus.protocol.semantic",
        "schema_version": 1,
    }
    semantic_text = _json(semantic)
    semantic_hash = "sha256:" + sha256(semantic_text.encode()).hexdigest()
    trial_text = _json(
        {
            "normalization_policy": "citrus.protocol.trial_index.v1",
            "protocol_semantic_hash": semantic_hash,
            "schema_id": "citrus.protocol.trial_index",
            "schema_version": 1,
            "steps": [
                {
                    "duration_s": 1.0,
                    "features": {
                        "color_name": "black",
                        "resolved_color": {
                            "color_space": "srgb",
                            "rgba8": [0, 0, 0, 255],
                        },
                    },
                    "index_status": "detailed",
                    "post_stimulus_iti_s": 0.0,
                    "step_index": 0,
                    "stimulus_family": "solid_color",
                    "stimulus_mode": "SOLID_BLACK",
                    "stimulus_mode_id": 4,
                },
                {
                    "duration_s": 2.0,
                    "features": {},
                    "index_status": "detailed",
                    "post_stimulus_iti_s": 0.0,
                    "step_index": 1,
                    "stimulus_family": "chaser",
                    "stimulus_mode": "CHASER",
                    "stimulus_mode_id": 12,
                },
            ],
        }
    )
    return semantic_hash, semantic_text, trial_text


def _write_raw_h5(path: Path, *, schema_version: int = 1) -> None:
    semantic_hash, semantic_text, trial_text = _semantic_documents()
    if schema_version == 2:
        trial = json.loads(trial_text)
        trial["schema_version"] = 2
        trial["normalization_policy"] = "citrus.protocol.trial_index.v2"
        trial_text = _json(trial)
    with h5py.File(path, "w") as h5:
        protocol = h5.create_group("protocol_snapshot")
        protocol.create_dataset("protocol_semantic_hash", data=semantic_hash)
        protocol.create_dataset("protocol_semantic_json", data=semantic_text)
        protocol.create_dataset("protocol_trial_index_json", data=trial_text)
        if schema_version == 2:
            protocol.attrs.update(
                {
                    "schema_id": "citrus.protocol.snapshot",
                    "schema_version": 2,
                    "policy_id": "citrus.protocol.snapshot.v2",
                    "contract_status": "valid",
                }
            )
            protocol.create_dataset(
                "protocol_trial_index_hash",
                data="sha256:" + sha256(trial_text.encode()).hexdigest(),
            )


def _archive(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "analysis.zarr"
    raw_h5 = tmp_path / "raw.h5"
    _write_raw_h5(raw_h5)
    root = zarr.open_group(str(archive), mode="w")
    root.attrs["recording_id"] = "recording-1"
    parent = root.require_group("analysis").require_group("stimulus_runs")
    parent.attrs.update({"latest": "source-v1", "latest_complete": "source-v1"})
    run = parent.create_group("source-v1")
    run.attrs.update(
        {
            "stage_selector_eligible": True,
            "source_h5": str(tmp_path / "canonical.h5"),
        }
    )
    mark_run_started(run, run_name="source-v1", stage="stimulus")
    steps = run.create_group("steps")
    for index, (mode_id, mode, duration) in enumerate(
        ((4, "SOLID_BLACK", 1.0), (12, "CHASER", 2.0))
    ):
        step = steps.create_group(f"step_{index}")
        step.attrs.update(
            {
                "step_index": index,
                "stimulus_mode_id": mode_id,
                "stimulus_mode": mode,
                "duration_s": duration,
                "start_camera_frame": index * 100,
                "end_camera_frame": (index + 1) * 100,
            }
        )
    run.create_array(
        "payload",
        data=np.asarray([1, 2, 3], dtype=np.int16),
        chunks=(3,),
    )
    provenance = build_writer_run_provenance(
        command="fixture",
        params={},
        input_run_ids={},
    )
    mark_run_complete(
        run,
        parent_group=parent,
        run_name="source-v1",
        run_provenance=provenance,
    )
    consolidate_metadata_capture_expected_warnings(archive)
    return archive, raw_h5


def test_plan_and_local_copy_preserve_arrays_and_bind_exact_semantics(
    tmp_path: Path,
) -> None:
    archive, raw_h5 = _archive(tmp_path)
    plan = successor.plan_historical_protocol_semantic_stimulus_successor(
        archive,
        source_run_name="source-v1",
        run_name="semantic-v1",
        raw_h5=raw_h5,
    )

    assert plan.receipt()["selector_activation"] == "none"
    assert plan.snapshot.trial_index_integrity_status == (
        "palette_computed_not_producer_asserted"
    )
    local = tmp_path / "local-run.zarr"
    successor._write_local_run(plan, local)
    run = zarr.open_group(str(local), mode="r", use_consolidated=False)
    assert run.attrs["schema_id"] == successor.SCHEMA_ID
    assert run.attrs["schema_version"] == successor.SCHEMA_VERSION
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["selector_eligible"] is False
    assert run.attrs["protocol_semantic_status"] == "verified"
    assert np.asarray(run["payload"][:]).tolist() == [1, 2, 3]
    assert run["steps/step_1"].attrs["protocol_semantic_step_index"] == 1
    assert successor._validate_run(local, plan=plan)["valid"] is True

    parent = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "analysis/stimulus_runs"
    ]
    assert parent.attrs["latest"] == "source-v1"
    assert "semantic-v1" not in parent


def test_plan_rejects_modern_v2_snapshot(tmp_path: Path) -> None:
    archive, raw_h5 = _archive(tmp_path)
    _write_raw_h5(raw_h5, schema_version=2)

    with pytest.raises(
        successor.HistoricalProtocolSemanticStimulusSuccessorError,
        match="sealed frame-bound acquisition contract",
    ):
        successor.plan_historical_protocol_semantic_stimulus_successor(
            archive,
            source_run_name="source-v1",
            run_name="semantic-v1",
            raw_h5=raw_h5,
        )
