from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apps.marimo.components.chaser_exact.body_alignment import (
    _body_alignment_values,
    build_exact_body_alignment_output,
)
from apps.marimo.components.chaser_exact.provenance import plain
from apps.marimo.components.chaser_exact_body_alignment_contract import (
    ExactBodyAlignmentContractError,
    validate_body_alignment_scientific_manifest,
)
from apps.marimo.components.chaser_exact_body_alignment_discovery import (
    compatible_body_alignment_binding,
)
from fisheye.analysis.provider_chaser_position_suite import PositionSuiteEpoch
from fisheye.analysis_workflows.chaser_body_alignment_by_distance_successor import (
    ChaserBodyAlignmentByDistanceInput,
    prepare_chaser_body_alignment_by_distance_successor,
)
from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    build_composable_chaser_successor_publication_plan,
    load_composable_chaser_successor_source_handle,
    publish_composable_chaser_successor_run,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root


class _Mo:
    @staticmethod
    def callout(text, *, kind):
        return {"text": text, "kind": kind}

    @staticmethod
    def vstack(items):
        return list(items)


def _inputs() -> ChaserBodyAlignmentByDistanceInput:
    n_frames = 6
    shape = (n_frames, 1)
    return ChaserBodyAlignmentByDistanceInput(
        recording_id="recording",
        relative_frame_run_path="analysis/chaser_relative_frame_runs/keypoint",
        relative_frame_manifest_sha256="a" * 64,
        semantic_selection_run_path=(
            "analysis/protocol_semantic_chaser_selection_runs/semantic"
        ),
        semantic_selection_manifest_sha256="b" * 64,
        n_frames=n_frames,
        n_chasers=1,
        acquisition_frame_id=np.arange(n_frames, dtype=np.int64)[:, None],
        selection_member=np.ones(shape, dtype=bool),
        chaser_occurrence_member=np.ones(shape, dtype=bool),
        chaser_identity_code=np.ones(shape, dtype=np.uint16),
        chaser_behavior_role_code=np.full(shape, 2, dtype=np.uint8),
        chaser_behavior_role_valid=np.ones(shape, dtype=bool),
        relative_distance_physical=np.asarray(
            [2.0, 2.0, 7.0, 7.0, 12.0, 12.0], dtype=np.float32
        )[:, None],
        relative_physical_valid=np.ones(shape, dtype=bool),
        relative_physical_reason_code=np.zeros(shape, dtype=np.uint16),
        body_source_row_id=np.arange(n_frames, dtype=np.int64)[:, None],
        body_source_row_valid=np.ones(shape, dtype=bool),
        body_heading_deg=np.zeros(shape, dtype=np.float32),
        body_heading_valid=np.ones(shape, dtype=bool),
        body_heading_reason_code=np.zeros(shape, dtype=np.uint16),
        body_bearing_deg=np.asarray(
            [0.0, 90.0, -90.0, 180.0, -180.0, 0.0], dtype=np.float32
        )[:, None],
        body_bearing_valid=np.ones(shape, dtype=bool),
        body_bearing_reason_code=np.zeros(shape, dtype=np.uint16),
        epochs=tuple(
            PositionSuiteEpoch(
                analysis_role=role,
                window_id=index,
                source_label=f"source-{index}",
                start_frame=index * 2,
                end_frame=index * 2 + 2,
                source_interval_sha256=str(index + 1) * 64,
            )
            for index, role in enumerate(
                ("chaser_pre", "chaser_training", "chaser_post")
            )
        ),
        fish_position_authority={
            "provider_id": "keypoint.v1",
            "provider_digest": "c" * 64,
        },
        body_frame_authority={
            "provider_id": "body.v1",
            "provider_digest": "d" * 64,
        },
        identity_registries={
            "chaser": {"1": "blue-dot"},
            "behavior_role": {"2": "aggressive"},
        },
        scale_policy={"unit": "mm", "pixels_per_unit": 2.0},
        distance_bin_width_mm=5.0,
    )


def _publish(
    tmp_path: Path,
    *,
    run_name: str = "alignment-v1",
    inputs: ChaserBodyAlignmentByDistanceInput | None = None,
):
    archive = tmp_path / "analysis.zarr"
    if not archive.exists():
        root = open_zarr_root(archive, mode="w-")
        root.attrs["recording_id"] = "recording"
    prepared = prepare_chaser_body_alignment_by_distance_successor(inputs or _inputs())
    plan = build_composable_chaser_successor_publication_plan(
        archive, run_name=run_name, prepared=prepared
    )
    publish_composable_chaser_successor_run(
        plan, scratch_root=tmp_path / f"scratch-{run_name}"
    )
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="chaser_body_alignment_by_distance",
        run_name=run_name,
        deep_audit=True,
    )
    return archive, prepared, handle


def _discovery_sources(prepared):
    sources = prepared.manifest["sources"]
    spatial_sources = {
        "position_providers": [
            {
                "provider_role": "keypoint",
                "relative_frame": dict(sources["relative_frame"]),
            },
            {
                "provider_role": "detection",
                "relative_frame": {
                    "run_path": "analysis/chaser_relative_frame_runs/detection",
                    "manifest_sha256": "e" * 64,
                },
            },
        ],
        "protocol_semantic_selection": dict(sources["protocol_semantic_selection"]),
    }
    relative_manifest = {
        "dimensions": {"n_frames": 6, "n_chasers": 1, "n_rows": 6},
        "source_authorities": {
            "fish_position": dict(sources["fish_position_authority"]),
            "body_frame": dict(sources["body_frame_authority"]),
        },
        "scale_policy": dict(sources["scale_policy"]),
    }
    return spatial_sources, relative_manifest


def test_discovery_and_renderer_use_one_persisted_bin_contract(tmp_path: Path) -> None:
    archive, prepared, handle = _publish(tmp_path)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    spatial_sources, relative_manifest = _discovery_sources(prepared)
    binding = compatible_body_alignment_binding(
        root,
        recording_id="recording",
        spatial_sources=spatial_sources,
        spatial_epoch_records=prepared.manifest["epoch_records"],
        keypoint_relative_manifest=relative_manifest,
    )

    assert binding is not None
    assert binding["run_path"].endswith("/alignment-v1")
    assert binding["distance_bin_recipe"]["edges_mm"] == [
        0.0,
        5.0,
        10.0,
        15.0,
    ]
    projection = SimpleNamespace(
        body_alignment_by_distance=handle,
        provenance={"recording_id": "recording"},
    )
    values = _body_alignment_values(projection)
    assert np.asarray(values["summary_candidate_row_count"]).sum() == 6

    import plotly.graph_objects as go

    output = build_exact_body_alignment_output(_Mo, go, projection)
    assert len(output) == 5
    for figure in output[1:]:
        assert (
            figure.layout.meta["body_alignment_by_distance_display"]["viewer_rebinning"]
            == "prohibited"
        )
        figure.to_plotly_json()


def test_discovery_accepts_independent_receipts_for_same_relative_child(
    tmp_path: Path,
) -> None:
    alignment_receipt = "8" * 64
    inputs = replace(
        _inputs(),
        relative_frame_verification_mode=(
            "receipt_bound_targeted_array_rehash_v1"
        ),
        relative_frame_validation_receipt_sha256=alignment_receipt,
    )
    archive, prepared, _handle = _publish(tmp_path, inputs=inputs)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    spatial_sources, relative_manifest = _discovery_sources(prepared)
    spatial_receipt = "9" * 64
    spatial_sources["position_providers"][0]["relative_frame"] = {
        **spatial_sources["position_providers"][0]["relative_frame"],
        "validation_receipt_sha256": spatial_receipt,
    }

    binding = compatible_body_alignment_binding(
        root,
        recording_id="recording",
        spatial_sources=spatial_sources,
        spatial_epoch_records=prepared.manifest["epoch_records"],
        keypoint_relative_manifest=relative_manifest,
    )

    assert binding is not None
    assert (
        binding["source_relative_frame"]["validation_receipt_sha256"]
        == alignment_receipt
    )
    assert (
        spatial_sources["position_providers"][0]["relative_frame"][
            "validation_receipt_sha256"
        ]
        == spatial_receipt
    )


def test_discovery_fails_closed_on_ambiguous_matching_children(tmp_path: Path) -> None:
    archive, prepared, _handle = _publish(tmp_path, run_name="alignment-a")
    _publish(tmp_path, run_name="alignment-b")
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    spatial_sources, relative_manifest = _discovery_sources(prepared)

    assert (
        compatible_body_alignment_binding(
            root,
            recording_id="recording",
            spatial_sources=spatial_sources,
            spatial_epoch_records=prepared.manifest["epoch_records"],
            keypoint_relative_manifest=relative_manifest,
        )
        is None
    )


def test_contract_rejects_viewer_rebinning_policy_tamper() -> None:
    prepared = prepare_chaser_body_alignment_by_distance_successor(_inputs())
    scientific = {key: value for key, value in prepared.manifest.items()}
    scientific["denominators"] = {
        **dict(scientific["denominators"]),
        "viewer_rebinning": "allowed",
    }
    scientific_body = plain(scientific)
    scientific_body.pop("payload_digest")
    scientific["payload_digest"] = canonical_json_sha256(scientific_body)
    sources = prepared.manifest["sources"]

    with pytest.raises(ExactBodyAlignmentContractError, match="fallback policy"):
        validate_body_alignment_scientific_manifest(
            scientific,
            expected_scientific_payload_sha256=scientific["payload_digest"],
            expected_n_frames=6,
            expected_n_chasers=1,
            expected_relative_binding=sources["relative_frame"],
            expected_semantic_binding=sources["protocol_semantic_selection"],
            expected_fish_position_authority=sources["fish_position_authority"],
            expected_body_frame_authority=sources["body_frame_authority"],
            expected_scale_policy=sources["scale_policy"],
            expected_epoch_records=prepared.manifest["epoch_records"],
        )


def test_renderer_rejects_nonconserving_persisted_support(tmp_path: Path) -> None:
    _archive, _prepared, handle = _publish(tmp_path)
    arrays = {name: np.array(value, copy=True) for name, value in handle.arrays.items()}
    arrays["summary_joint_valid_row_count"][0] += 1

    class _Tampered:
        scientific_manifest = handle.scientific_manifest

        def require_verified_arrays(self, _names):
            return None

        def array(self, name):
            return arrays[name]

    projection = SimpleNamespace(body_alignment_by_distance=_Tampered())
    with pytest.raises(ValueError, match="do not conserve bins"):
        _body_alignment_values(projection)


def test_shared_static_interactive_parser_rejects_fallback_policy_tamper(
    tmp_path: Path,
) -> None:
    _archive, _prepared, handle = _publish(tmp_path)
    scientific = plain(handle.scientific_manifest)
    scientific["denominators"] = {
        **scientific["denominators"],
        "viewer_rebinning": "allowed",
    }
    unsigned = dict(scientific)
    unsigned.pop("payload_digest")
    scientific["payload_digest"] = canonical_json_sha256(unsigned)

    class _Tampered:
        def require_verified_arrays(self, _names):
            return None

        def array(self, name):
            return handle.array(name)

    tampered = _Tampered()
    tampered.scientific_manifest = scientific
    projection = SimpleNamespace(body_alignment_by_distance=tampered)
    with pytest.raises(ValueError, match="fallback policy"):
        _body_alignment_values(projection)
