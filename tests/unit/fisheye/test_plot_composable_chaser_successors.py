from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fisheye.utils import plot_composable_chaser_successors as plot_module
from fisheye.utils.plot_composable_chaser_successors import (
    ComposableChaserPlotError,
    dashboard_plot_parameters,
    render_dashboard,
)


@dataclass
class _Handle:
    successor_kind: str
    recording_id: str
    scientific_payload_sha256: str
    scientific_manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    deep_audited: bool = True
    verification_mode: str = "deep_audit"
    run_path: str = "analysis/example/run"
    manifest_sha256: str = "d" * 64
    verified_array_names: tuple[str, ...] = ()
    receipt_digest: str | None = None

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def require_verified_authority(self) -> None:
        if self.verification_mode not in {
            "deep_audit",
            "receipt_bound_targeted_array_rehash_v1",
        }:
            raise ValueError("unsupported verification mode")

    def require_verified_arrays(self, names: tuple[str, ...]) -> None:
        missing = set(names).difference(self.arrays)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)!r}")


def _handles() -> tuple[_Handle, _Handle, _Handle]:
    controller_digest = "a" * 64
    bout_digest = "b" * 64
    controller = _Handle(
        successor_kind="controller_chase_trials",
        recording_id="recording-1",
        scientific_payload_sha256=controller_digest,
        scientific_manifest={},
        arrays={
            "start_acquisition_frame_id": np.asarray([100, 200], dtype=np.int64),
            "end_acquisition_frame_id_inclusive": np.asarray(
                [149, 259], dtype=np.int64
            ),
            "trial_ordinal": np.asarray([1, 2], dtype=np.int32),
            "logged_trial_id": np.asarray([11, 12], dtype=np.int64),
            "chaser_identity_code": np.asarray([1, 1], dtype=np.uint16),
            "gap_fraction": np.asarray([0.0, 0.1], dtype=np.float64),
        },
    )
    bout = _Handle(
        successor_kind="generalized_chaser_bout_response",
        recording_id="recording-1",
        scientific_payload_sha256=bout_digest,
        scientific_manifest={
            "sources": {
                "controller_trial_payload_sha256": controller_digest,
                "motion": {
                    "relative_frame_projection": {
                        "missing_relative_frame_count": 3
                    }
                },
            },
            "identity_registries": {
                "semantic_role": {"2": "chaser_training"}
            },
        },
        arrays={
            "summary_role_code": np.asarray([2, 2], dtype=np.uint8),
            "summary_chaser_identity_code": np.asarray([1, 1], dtype=np.uint16),
            "summary_distance_bin_index": np.asarray([0, 1], dtype=np.int16),
            "summary_distance_bin_start_mm": np.asarray([0, 8], dtype=np.float32),
            "summary_distance_bin_end_mm": np.asarray([8, np.inf], dtype=np.float32),
            "summary_bout_rate_per_min": np.asarray([1.5, 2.5], dtype=np.float64),
            "summary_bout_count": np.asarray([2, 3], dtype=np.int64),
        },
    )
    escape = _Handle(
        successor_kind="chaser_escape_freeze",
        recording_id="recording-1",
        scientific_payload_sha256="c" * 64,
        scientific_manifest={
            "sources": {
                "controller_trial_payload_sha256": controller_digest,
                "bout_response_payload_sha256": bout_digest,
            }
        },
        arrays={
            "trial_response_class_code": np.asarray([1, 2], dtype=np.uint8),
            "sweep_trial_row_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "sweep_speed_threshold_mm_s": np.asarray(
                [10, 20, 10, 20], dtype=np.float32
            ),
            "sweep_escape_event_rate_per_min": np.asarray(
                [3, 2, 1, 0], dtype=np.float64
            ),
        },
    )
    return controller, bout, escape


def test_render_dashboard_writes_png_and_pdf(tmp_path: Path) -> None:
    handles = _handles()
    png, pdf = render_dashboard(*handles, output_stem=tmp_path / "dashboard")
    parameters = dashboard_plot_parameters(*handles)

    assert png.is_file() and png.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert parameters["scientific_coordinates"]["bout_distance_bins"] == [
        {"bin_index": 0, "start_mm_inclusive": 0.0, "end_mm_exclusive": 8.0},
        {"bin_index": 1, "start_mm_inclusive": 8.0, "end_mm_exclusive": None},
    ]
    assert parameters["scientific_coordinates"]["escape_speed_thresholds_mm_s"] == [
        10.0,
        20.0,
    ]
    assert parameters["rendering"]["png_dpi"] == 180


def test_render_dashboard_rejects_stale_dependency(tmp_path: Path) -> None:
    controller, bout, escape = _handles()
    escape.scientific_manifest["sources"]["bout_response_payload_sha256"] = "f" * 64

    with pytest.raises(ComposableChaserPlotError, match="stale or mixed"):
        render_dashboard(controller, bout, escape, output_stem=tmp_path / "bad")


def test_main_uses_receipt_bound_targeted_array_rosters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handles = _handles()
    by_kind = {handle.successor_kind: handle for handle in handles}
    calls: list[dict[str, Any]] = []

    def load_handle(_archive: Path, **kwargs: Any) -> _Handle:
        calls.append(kwargs)
        handle = by_kind[kwargs["successor_kind"]]
        handle.deep_audited = False
        handle.verification_mode = "receipt_bound_targeted_array_rehash_v1"
        handle.verified_array_names = tuple(sorted(kwargs["required_array_names"]))
        handle.receipt_digest = "e" * 64
        return handle

    def render_stub(*_handles: _Handle, output_stem: Path) -> tuple[Path, Path]:
        png = output_stem.with_suffix(".png")
        pdf = output_stem.with_suffix(".pdf")
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"png")
        pdf.write_bytes(b"pdf")
        return png, pdf

    monkeypatch.setattr(
        plot_module, "load_composable_chaser_successor_source_handle", load_handle
    )
    monkeypatch.setattr(plot_module, "render_dashboard", render_stub)
    monkeypatch.setattr(plot_module, "dashboard_plot_parameters", lambda *_: {})

    output_dir = tmp_path / "plots"
    assert plot_module.main(
        [
            str(tmp_path / "analysis.zarr"),
            "--run-name",
            "successors-v1",
            "--expected-recording-id",
            "recording-1",
            "--output-dir",
            str(output_dir),
            "--controller-validation-receipt",
            str(tmp_path / "controller.json"),
            "--bout-validation-receipt",
            str(tmp_path / "bout.json"),
            "--escape-validation-receipt",
            str(tmp_path / "escape.json"),
        ]
    ) == 0

    assert len(calls) == 3
    for call in calls:
        assert call["deep_audit"] is False
        assert call["required_array_names"] == plot_module._PLOT_ARRAY_NAMES[
            call["successor_kind"]
        ]
    receipt = json.loads(
        (output_dir / "successors-v1_plot_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["schema_version"] == 3
    assert receipt["plot_policy"]["source_validation"] == (
        "receipt_bound_targeted_array_rehash_v1"
    )
