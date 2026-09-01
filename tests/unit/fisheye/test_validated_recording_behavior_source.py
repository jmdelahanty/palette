from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

import fisheye.analysis_workflows.validated_recording_behavior_source as subject
from apps.marimo.components.core_behavior import (
    ValidatedCoreBehaviorSource,
    collect_projection,
    load_core_behavior_projection,
)
from fisheye.analysis_workflows.materializers.provider_track_motion import (
    PROVIDER_TRACK_MOTION_MANIFEST_ATTR,
    PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR,
)
from fisheye.analysis_workflows.validated_recording_behavior_bundle import (
    CAPABILITY_KEYS,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR


def _complete(scope: str, key: str) -> dict[str, Any]:
    return {
        "state": "complete",
        "reason_code": None,
        "detail": None,
        "binding_scope": scope,
        "binding_key": key,
    }


def _unavailable() -> dict[str, Any]:
    return {
        "state": "unavailable",
        "reason_code": "not_persisted",
        "detail": "synthetic source fixture",
        "binding_scope": None,
        "binding_key": None,
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    run_path = "analysis/track_kinematics_runs/provider/motion-v1"
    run = root.require_group(run_path)
    arrays = {
        "angular_sample_reason_code": np.zeros(7, dtype=np.uint16),
        "angular_sample_valid": np.ones(7, dtype=bool),
        "heading_degrees": np.arange(7, dtype=np.float32) * 10.0,
        "linear_sample_reason_code": np.zeros(7, dtype=np.uint16),
        "linear_sample_valid": np.ones(7, dtype=bool),
        "position_source_valid": np.ones(7, dtype=bool),
        "positions_mm": np.arange(14, dtype=np.float32).reshape(7, 2),
        "source_acquisition_frame_index": np.arange(100, 107, dtype=np.int64),
        "speed_smoothed_mm": np.arange(7, dtype=np.float32) + 0.5,
        "time_seconds": np.arange(7, dtype=np.float32) / 10.0,
        "track_ids": np.asarray([2, 7], dtype=np.int64),
        "track_row_offsets": np.asarray([0, 3, 7], dtype=np.int64),
        "track_sample_key": np.arange(7, dtype=np.int64),
    }
    for name, values in arrays.items():
        run.create_array(name, data=values, chunks=values.shape)

    source_authority = {
        "record": {"authority": "synthetic-provider-motion"},
        "sha256": canonical_json_sha256({"authority": "synthetic-provider-motion"}),
    }
    array_records = [
        {
            "path": name,
            "dtype": np.dtype(values.dtype).str,
            "shape": list(values.shape),
            "sha256": sha256_array(values),
        }
        for name, values in sorted(arrays.items())
    ]
    payload = {
        "run_path": run_path,
        "row_axis": "track_sample",
        "status": "complete",
        "stage_selector_eligible": False,
        "source_authority": source_authority,
        "arrays": array_records,
    }
    manifest = {
        "schema_id": "palette.provider_track_motion_run_manifest",
        "schema_version": 1,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    run.attrs.update(
        {
            PROVIDER_TRACK_MOTION_MANIFEST_ATTR: manifest,
            PROVIDER_TRACK_MOTION_MANIFEST_DIGEST_ATTR: manifest["payload_digest"],
            RUN_COMPLETION_STATUS_ATTR: "complete",
            "stage_selector_eligible": False,
        }
    )
    zarr.consolidate_metadata(str(archive))

    projection_receipt = tmp_path / "projection.json"
    projection_receipt.write_text("{}", encoding="utf-8")
    bundle_path = tmp_path / "behavior-bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")
    capabilities = {key: _unavailable() for key in CAPABILITY_KEYS}
    capabilities["provider_motion"] = _complete("source_bindings", "provider_motion")
    capabilities["epoch_behavior"] = _complete(
        "scientific_child_bindings", "epoch_behavior"
    )
    bundle = {
        "analysis_zarr": str(archive.resolve()),
        "recording_id": "synthetic-recording",
        "record_sha256": "b" * 64,
        "projection_receipt": {
            "receipt_path": str(projection_receipt.resolve()),
            "receipt_sha256": "c" * 64,
        },
        "source_bindings": {
            "provider_motion": {
                "binding_type": "epoch_transitive_provider_motion_v1",
                "source": {
                    "run_path": run_path,
                    "manifest_sha256": manifest["payload_digest"],
                    "verification_digest": "d" * 64,
                    "track_id": 7,
                    "track_row_start": 3,
                    "track_row_stop": 7,
                },
                "source_authority": source_authority,
                "published_metadata": {},
                "sealed_by": {},
            }
        },
        "scientific_child_bindings": {
            "epoch_behavior": {
                "receipt_path": str((tmp_path / "epoch.json").resolve()),
                "receipt_sha256": "e" * 64,
                "run_path": "analysis/stimulus_epoch_behavior_summary_runs/epoch-v1",
                "manifest_sha256": "f" * 64,
                "payload_digest": "a" * 64,
            }
        },
        "capabilities": capabilities,
    }

    def read_bundle(path: str | Path, **expected: Any) -> dict[str, Any]:
        assert Path(path).resolve() == bundle_path.resolve()
        if expected.get("expected_analysis_zarr") is not None:
            assert (
                Path(expected["expected_analysis_zarr"]).resolve() == archive.resolve()
            )
        assert expected["validate_current_sources"] is True
        return deepcopy(bundle)

    monkeypatch.setattr(
        subject, "read_validated_recording_behavior_bundle", read_bundle
    )
    monkeypatch.setattr(
        subject,
        "read_exact_chaser_projection_receipt",
        lambda *_args, **_kwargs: {"record_sha256": "c" * 64},
    )
    return bundle_path, bundle, arrays


def test_capability_router_preserves_typed_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, _bundle, _arrays = _fixture(tmp_path, monkeypatch)
    source = subject.ValidatedRecordingBehaviorSource(bundle_path)

    binding = source.require_capability(
        "provider_motion", expected_binding_scope="source_bindings"
    )
    assert binding.binding_key == "provider_motion"
    assert binding.bundle_sha256 == "b" * 64
    with pytest.raises(subject.ValidatedCapabilityUnavailableError) as exc_info:
        source.require_capability("eye_angles")
    assert exc_info.value.state == "unavailable"
    assert exc_info.value.reason_code == "not_persisted"
    assert source.capability_states()["eye_angles"]["detail"] == (
        "synthetic source fixture"
    )


def test_targeted_provider_projection_hashes_only_consumed_arrays_and_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, _bundle, arrays = _fixture(tmp_path, monkeypatch)
    source = subject.ValidatedRecordingBehaviorSource(bundle_path)

    projection = source.provider_motion_track_projection(
        ("time_seconds", "speed_smoothed_mm", "positions_mm")
    )

    assert projection.track_id == 7
    assert projection.row_count == 4
    np.testing.assert_array_equal(
        projection.arrays["time_seconds"], arrays["time_seconds"][3:7]
    )
    np.testing.assert_array_equal(
        projection.arrays["positions_mm"], arrays["positions_mm"][3:7]
    )
    assert projection.arrays["positions_mm"].flags.writeable is False
    assert set(source._verified_provider_arrays) == {
        "track_ids",
        "track_row_offsets",
        "time_seconds",
        "speed_smoothed_mm",
        "positions_mm",
    }
    assert projection.source_paths["speed_smoothed_mm"].endswith("/speed_smoothed_mm")


def test_targeted_provider_projection_rejects_tampered_consumed_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, bundle, _arrays = _fixture(tmp_path, monkeypatch)
    mutable = zarr.open_group(bundle["analysis_zarr"], mode="a", use_consolidated=False)
    mutable["analysis/track_kinematics_runs/provider/motion-v1/speed_smoothed_mm"][
        0
    ] = np.float32(999.0)
    source = subject.ValidatedRecordingBehaviorSource(bundle_path)

    with pytest.raises(
        subject.ValidatedRecordingBehaviorSourceError,
        match="differs from its manifest digest",
    ):
        source.provider_motion_track_projection(("speed_smoothed_mm",))


def test_targeted_provider_projection_rejects_changed_track_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, bundle, _arrays = _fixture(tmp_path, monkeypatch)
    bundle["source_bindings"]["provider_motion"]["source"]["track_row_start"] = 2
    source = subject.ValidatedRecordingBehaviorSource(bundle_path)

    with pytest.raises(
        subject.ValidatedRecordingBehaviorSourceError,
        match="track partition differs",
    ):
        source.provider_motion_catalog()


def test_projection_receipt_must_match_bundle_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, _bundle, _arrays = _fixture(tmp_path, monkeypatch)
    source = subject.ValidatedRecordingBehaviorSource(bundle_path)
    other = tmp_path / "other-projection.json"
    other.write_text("{}", encoding="utf-8")

    with pytest.raises(
        subject.ValidatedRecordingBehaviorSourceError,
        match="differs from the receipt bound",
    ):
        source.exact_projection_receipt_path(explicit_path=other)


def test_projection_receipt_digest_is_rechecked_at_route_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, _bundle, _arrays = _fixture(tmp_path, monkeypatch)
    source = subject.ValidatedRecordingBehaviorSource(bundle_path)
    monkeypatch.setattr(
        subject,
        "read_exact_chaser_projection_receipt",
        lambda *_args, **_kwargs: {"record_sha256": "9" * 64},
    )

    with pytest.raises(
        subject.ValidatedRecordingBehaviorSourceError,
        match="receipt changed after",
    ):
        source.exact_projection_receipt_path()


def test_provider_projection_rejects_structural_and_non_sample_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, _bundle, _arrays = _fixture(tmp_path, monkeypatch)
    source = subject.ValidatedRecordingBehaviorSource(bundle_path)

    with pytest.raises(subject.ValidatedRecordingBehaviorSourceError):
        source.provider_motion_track_projection(("track_ids",))
    with pytest.raises(subject.ValidatedRecordingBehaviorSourceError):
        source.provider_motion_track_projection(("time_seconds", "time_seconds"))


def test_validated_core_behavior_routes_direct_surfaces_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, _bundle, arrays = _fixture(tmp_path, monkeypatch)
    source = subject.ValidatedRecordingBehaviorSource(bundle_path)
    core = ValidatedCoreBehaviorSource(source)

    assert core.available_analysis_ids() == ("speed", "heading", "position")
    assert core.eye_angle_options() == ()
    assert core.tail_kinematics_options() == ()
    assert core.baseline_interval() is None

    speed = load_core_behavior_projection(
        core,
        "speed",
        start_s=0.4,
        stop_s=0.61,
        series_keys=("speed_smoothed_mm",),
    )
    collected = collect_projection(speed)
    np.testing.assert_allclose(
        collected["speed_smoothed_mm"].to_numpy(),
        arrays["speed_smoothed_mm"][4:7],
    )
    assert collected["frame_index"].to_list() == [104, 105, 106]
    assert collected["linear_sample_valid"].to_list() == [True, True, True]
    assert speed.metadata["bundle_sha256"] == "b" * 64
    assert speed.metadata["capability_states"]["eye_angles"] == {
        "state": "unavailable",
        "reason_code": "not_persisted",
        "detail": "synthetic source fixture",
    }
    assert set(speed.metadata["consumed_arrays"]) == {
        "time_seconds",
        "source_acquisition_frame_index",
        "speed_smoothed_mm",
        "linear_sample_valid",
        "linear_sample_reason_code",
    }

    position = load_core_behavior_projection(core, "position")
    assert "position_source_valid" in position.columns
    assert position.row_count == 4

    with pytest.raises(
        subject.ValidatedRecordingBehaviorSourceError,
        match="independent selector discovery is prohibited",
    ):
        core.project_eye_angles()
    with pytest.raises(
        subject.ValidatedRecordingBehaviorSourceError,
        match="independent selector discovery is prohibited",
    ):
        core.eye_angle_catalog()
