from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.body_heading import (
    BODY_HEADING_DISPLAY_RECIPE,
    _collapsed_body_heading,
    build_exact_body_heading_output,
)
from apps.marimo.components.chaser_exact.projection import RelativeFrameProjection
from apps.marimo.components.chaser_exact_body_heading_contract import (
    BODY_HEADING_ARRAY_PATHS,
    BODY_HEADING_FRAME_COLLAPSE_POLICY,
    compatible_body_heading_binding,
)


class _Marimo:
    @staticmethod
    def callout(value: str, *, kind: str) -> dict[str, str]:
        return {"value": value, "kind": kind}

    @staticmethod
    def vstack(values: list[Any]) -> list[Any]:
        return values


def _relative(*, mismatched_heading: bool = False) -> RelativeFrameProjection:
    n_frames = 4
    n_chasers = 2
    frame_id = np.repeat(np.arange(n_frames, dtype=np.int64), n_chasers)
    source_row = np.repeat(np.asarray([10, 11, -1, 13], dtype=np.int64), n_chasers)
    heading = np.repeat(
        np.asarray([0.0, 90.0, np.nan, -90.0], dtype=np.float32), n_chasers
    )
    if mismatched_heading:
        heading[3] = np.float32(80.0)
    source_valid = source_row >= 0
    heading_valid = source_valid.copy()
    reason = np.where(heading_valid, 0, 1).astype(np.uint16)
    return RelativeFrameProjection(
        run_path="analysis/chaser_relative_frame_runs/keypoint-v1",
        run_name="keypoint-v1",
        recording_id="recording-1",
        manifest_sha256="a" * 64,
        n_frames=n_frames,
        n_chasers=n_chasers,
        source_authorities=MappingProxyType(
            {
                "body_frame": MappingProxyType(
                    {
                        "source_authority_id": "accepted-body-frame-source",
                        "source_digest": "c" * 64,
                        "provider_id": "accepted-keypoint-body-frame",
                        "provider_digest": "d" * 64,
                    }
                )
            }
        ),
        arrays=MappingProxyType(
            {
                "acquisition_frame_id": frame_id,
                "selection_member": np.ones(n_frames * n_chasers, dtype=bool),
            }
        ),
        body_arrays=MappingProxyType(
            {
                "body_source_row_id": source_row,
                "body_source_row_valid": source_valid,
                "body_heading_deg": heading,
                "body_heading_valid": heading_valid,
                "body_heading_reason_code": reason,
            }
        ),
    )


def _projection(relative: RelativeFrameProjection | None = None) -> Any:
    keypoint = relative or _relative()
    return SimpleNamespace(
        relatives=(keypoint, keypoint),
        recording_id="recording-1",
        epoch_records=(
            {
                "analysis_role": "chaser_pre",
                "window_id": 0,
                "start_frame": 0,
                "end_frame_exclusive": 2,
            },
            {
                "analysis_role": "chaser_training",
                "window_id": 1,
                "start_frame": 2,
                "end_frame_exclusive": 3,
            },
            {
                "analysis_role": "chaser_post",
                "window_id": 2,
                "start_frame": 3,
                "end_frame_exclusive": 4,
            },
        ),
        provenance={"bundle_manifest_sha256": "b" * 64},
    )


def test_body_heading_collapses_exact_chaser_repeats_once_per_frame() -> None:
    output = build_exact_body_heading_output(_Marimo, go, _projection())

    assert len(output) == 2
    figure = output[1]
    assert len(figure.data) == 4
    assert int(np.sum(figure.data[0].customdata)) == 3
    assert int(np.sum(figure.data[1].customdata)) == 2
    assert int(np.sum(figure.data[2].customdata)) == 0
    assert int(np.sum(figure.data[3].customdata)) == 1
    display = figure.layout.meta["body_heading_display"]
    assert display["recipe_id"] == BODY_HEADING_DISPLAY_RECIPE
    assert display["frame_collapse_policy"] == BODY_HEADING_FRAME_COLLAPSE_POLICY
    assert display["body_axis_fallback"] == "prohibited"
    assert display["motion_heading_fallback"] == "prohibited"
    assert display["body_frame_authority"]["provider_digest"] == "d" * 64
    assert display["bin_edges_deg"][0] == -180.0
    assert display["bin_edges_deg"][-1] == 180.0
    assert len(display["bin_edges_deg"]) == 37
    assert display["panels"][0]["candidate_frame_count"] == 4
    assert display["panels"][0]["valid_heading_count"] == 3
    assert display["panels"][0]["missing_body_source_count"] == 1
    assert display["panels"][1]["label"] == "chaser_pre · window 0"


def test_body_heading_rejects_cross_chaser_disagreement() -> None:
    with pytest.raises(ValueError, match="differs by chaser row"):
        _collapsed_body_heading(_relative(mismatched_heading=True))


def test_body_heading_metadata_binding_requires_the_closed_array_roster() -> None:
    expected = {
        "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
        "manifest_sha256": "a" * 64,
    }
    manifest = {
        "schema_binding": {"body_extension_present": True},
        "source_authorities": {
            "body_frame": {
                "source_authority_id": "accepted-body-frame-source",
                "source_digest": "c" * 64,
                "provider_id": "accepted-keypoint-body-frame",
                "provider_digest": "d" * 64,
            }
        },
        "array_declarations": [{"path": path} for path in BODY_HEADING_ARRAY_PATHS],
    }

    binding = compatible_body_heading_binding(
        manifest,
        expected_relative_binding=expected,
    )

    assert binding is not None
    assert binding["array_paths"] == list(BODY_HEADING_ARRAY_PATHS)
    assert binding["frame_collapse_policy"] == BODY_HEADING_FRAME_COLLAPSE_POLICY
    assert binding["motion_heading_fallback"] == "prohibited"
    manifest_without_authority = {
        **manifest,
        "source_authorities": {"body_frame": None},
    }
    assert (
        compatible_body_heading_binding(
            manifest_without_authority,
            expected_relative_binding=expected,
        )
        is None
    )
    manifest["array_declarations"].pop()
    assert (
        compatible_body_heading_binding(
            manifest,
            expected_relative_binding=expected,
        )
        is None
    )
