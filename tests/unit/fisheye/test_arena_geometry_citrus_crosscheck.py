"""Citrus-declared versus unreviewed Palette blind-fit geometry cross-check.

The cross-check is evidence only: it logs raw disagreement between the dish
geometry Citrus declared/applied in the unified H5 and the Palette blind
``top_rim_preferred_v1`` fit.  It never selects, accepts or replaces geometry
and carries no pass/fail flag.  The preservation tests pin the pre-existing
reviewed comparison grammar byte-for-byte.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis_workflows.materializers import (
    arena_geometry_candidates as candidates,
)
from fisheye.analysis_workflows.materializers import (
    arena_geometry_comparison as comparison,
)
from fisheye.analysis_workflows.materializers import (
    arena_geometry_fit_review as fit_review,
)
from fisheye.analysis_workflows.materializers import (
    arena_geometry_selection as selection,
)
from fisheye.registry.registered_geometry_readiness import (
    project_registered_geometry_stages,
)
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.recording_geometry import (
    CircleGeometry,
    CitrusRegistrationStatus,
)
from fisheye.shared.zarr_io import open_zarr_root
from tests.unit.fisheye.test_arena_geometry_candidates import (
    _bound_mask,
    _palette_binding,
)
from tests.unit.fisheye.test_arena_geometry_comparison import (
    _archive_with_candidates,
    _write_candidate,
    _write_nested_detection_source,
)


def _sha(value: object) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _replace_text(value: object, old: str, new: str) -> object:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_text(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_text(item, old, new) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Preservation: the reviewed comparison path stays byte-identical.
# ---------------------------------------------------------------------------

# Captured on the pre-change module (origin/main b04ec0d4 merged into the
# geometry-rim-metrics branch d4790e23) for the deterministic fixture below.
_GOLDEN_REVIEWED = {
    "corroborated_acquisition_v1": {
        "comparison_id": "arena_geometry_comparison_103298f452ba4a426168",
        "comparison_record_sha256": (
            "103298f452ba4a42616819d850260ef41a50feaf6bdde988588eb863e1285a6d"
        ),
        "run_attrs_sha256": (
            "d4fd23915108cd90f099eb18113484ec091ca488a8942401d997c1329ec30083"
        ),
    },
    "manual_review_only_v1": {
        "comparison_id": "arena_geometry_comparison_32dc042bf25714aa162a",
        "comparison_record_sha256": (
            "32dc042bf25714aa162a6d0ecf43c9bc4f857a81482e2b8f10a5afa5e3785417"
        ),
        "run_attrs_sha256": (
            "42e69a22a9041d26f7da82e4037ad8beb8812d97f0a52c4866435596afe345fc"
        ),
    },
}


def _deterministic_reviewed_archive(tmp_path, monkeypatch):
    archive, acquisition, palette = _archive_with_candidates(tmp_path, monkeypatch)
    root = open_zarr_root(archive, mode="a")
    parent = root[f"analysis/{candidates.CANDIDATE_RUNS_PARENT}"]
    record = dict(parent[palette].attrs["candidate_record"])
    del parent[palette]
    record = _replace_text(record, str(tmp_path), "/fixture")
    palette = _write_candidate(parent, record)
    return archive, acquisition, palette, _write_nested_detection_source(archive)


@pytest.mark.parametrize("policy_id", sorted(_GOLDEN_REVIEWED))
def test_reviewed_comparison_bytes_are_unchanged(tmp_path, monkeypatch, policy_id):
    archive, acquisition, palette, detection = _deterministic_reviewed_archive(
        tmp_path, monkeypatch
    )
    plan = comparison.build_arena_geometry_comparison_plan(
        archive,
        acquisition_candidate_run=acquisition,
        palette_candidate_run=palette,
        semantic_compatibility="projected_edges_unresolved",
        policy_id=policy_id,
        detect_source_group_path=detection,
    )
    attrs = {
        key: value
        for key, value in comparison._run_attrs(plan).items()
        if key != "run_provenance"
    }
    observed = {
        "comparison_id": plan.comparison_id,
        "comparison_record_sha256": plan.comparison_record_sha256,
        "run_attrs_sha256": _sha(attrs),
    }
    assert observed == _GOLDEN_REVIEWED[policy_id]
    result = comparison.publish_arena_geometry_comparison(
        plan, scratch_root=tmp_path / "scratch"
    )
    assert result["published"] is True
    run = open_zarr_root(archive, mode="r")[
        f"analysis/{comparison.COMPARISON_RUNS_PARENT}/{plan.comparison_id}"
    ]
    persisted = {key: run.attrs[key] for key in attrs}
    assert _sha(persisted) == _GOLDEN_REVIEWED[policy_id]["run_attrs_sha256"]
    assert run.attrs["schema_id"] == "palette.arena_geometry_comparison_run"
    assert run.attrs["schema_version"] == 1


# ---------------------------------------------------------------------------
# Citrus cross-check fixtures: real probe output, real fit-review publisher.
# ---------------------------------------------------------------------------

_AGREEING = ((256.2, 255.7, 217.5), (256.2, 255.7, 222.0))
_DISAGREEING = ((286.2, 255.7, 190.0), (286.2, 255.7, 195.0))


def _citrus_record(
    circles=_AGREEING,
    *,
    status="exact_match_applied",
    applied=True,
    source_kind="citrus_h5",
):
    physical, gate = circles
    base = _bound_mask()
    mask = replace(
        base.mask,
        source_kind=source_kind,
        source_location="/recording/raw/citsess_test.h5",
        producer_contract_linkage_status="producer_native",
        recovery_receipt_sha256=None,
        physical_inner_rim=CircleGeometry(*physical),
        valid_detection_gate=CircleGeometry(*gate),
        native_width_px=512,
        native_height_px=512,
        citrus_registration_status=CitrusRegistrationStatus(status),
        selected_daily_registration_applied_by_citrus=applied,
    )
    return candidates.build_acquisition_geometry_candidate_record(
        replace(base, mask=mask), recovery_binding=None
    )


def _binding_512():
    coordinate, arena, _source = _palette_binding()
    coordinate = dict(coordinate, native_width_px=512, native_height_px=512)
    video = {
        "total_frames": 1000,
        "source_video": "camera.mp4",
        "file_fingerprint": {"size_bytes": len(b"synthetic-decoder-boundary")},
    }
    return coordinate, arena, video


def _blind_fit_package(tmp_path, monkeypatch, *, recipe=None):
    from fisheye.diagnostics import probe_recording_dish_rim_fit as probe
    from tests.unit.fisheye.test_dish_rim_metrics_publication import _probe_args

    inputs = tmp_path / "probe_inputs"
    inputs.mkdir()
    args = _probe_args(inputs, monkeypatch, recipe=recipe or probe.TOP_RIM_RECIPE)
    return probe.run_probe(args).parent


def _write_detection_512(archive, monkeypatch):
    from fisheye.shared.zarr.detection_schema import CanonicalDetectionDimensions

    def manifest_reader(root, *, group_path, **_kwargs):
        return dict(root[group_path].attrs["run_manifest"])

    monkeypatch.setattr(
        comparison, "require_active_coordinate_canonical_detection", manifest_reader
    )
    monkeypatch.setattr(
        comparison,
        "canonical_detection_dimensions_from_manifest",
        lambda _manifest: CanonicalDetectionDimensions(
            n_frames=3, n_instances=3, source_width=512, source_height=512
        ),
    )
    root = open_zarr_root(archive, mode="a")
    run = root.require_group("detect_runs").create_group("detect_native")
    run.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_video_width": 512,
            "source_video_height": 512,
            "num_frames": 3,
            "source_evidence": {
                "source_pixel_authority": {
                    "record_ref": (
                        "/analysis/coordinate_frames/source_camera/2010093/"
                        "continuous@pixel_frame_authority"
                    ),
                    "record_sha256": "e" * 64,
                }
            },
            "run_manifest": {
                "schema_id": "palette.canonical_detection.run_manifest",
                "schema_version": 3,
                "payload_digest": "f" * 64,
                "payload": {"run_id": "detect_native"},
            },
        }
    )
    instances = run.create_group("instances")
    boxes = instances.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [256.0 / 512.0, 256.0 / 512.0, 0.05, 0.05],
                [60.0 / 512.0, 256.0 / 512.0, 0.05, 0.05],
                [470.0 / 512.0, 256.0 / 512.0, 0.05, 0.05],
            ],
            dtype=np.float32,
        ),
    )
    boxes.attrs["coordinate_descriptor"] = {
        "geometry_type": "bbox_cxcywh",
        "component_units": ["normalized"] * 4,
    }
    instances.create_array("frame_indices", data=np.asarray([0, 1, 2], dtype=np.int32))
    instances.create_array(
        "instance_key", data=np.asarray([10, 20, 30], dtype=np.uint64)
    )
    return "detect_runs/detect_native"


def _crosscheck_archive(
    tmp_path,
    monkeypatch,
    *,
    record=None,
    recipe=None,
    with_detection=True,
):
    import zarr

    archive = tmp_path / "recording.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    parent = root.require_group("analysis").create_group(
        candidates.CANDIDATE_RUNS_PARENT
    )
    acquisition = _write_candidate(parent, record or _citrus_record())
    package_dir = _blind_fit_package(tmp_path, monkeypatch, recipe=recipe)
    plan = fit_review.build_arena_geometry_fit_review_plan(
        archive, review_package_dir=package_dir
    )
    fit_review.publish_arena_geometry_fit_review(
        plan, scratch_root=tmp_path / "fit-scratch"
    )
    monkeypatch.setattr(
        candidates,
        "_source_camera_candidate_binding",
        lambda *_a, **_k: _binding_512(),
    )
    arguments = {
        "acquisition_candidate_run": acquisition,
        "fit_review_run": plan.run_name,
    }
    if with_detection:
        arguments["detect_source_group_path"] = _write_detection_512(
            archive, monkeypatch
        )
    return archive, arguments


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _snapshot(archive):
    return {
        path.relative_to(archive): path.read_bytes()
        for path in archive.rglob("*")
        if path.is_file()
    }


_SOURCE_PREFIXES = (
    "analysis/arena_geometry_runs/",
    "analysis/arena_geometry_fit_runs/",
    "detect_runs/",
)


def test_valid_citrus_dish_crosscheck_needs_no_review_and_publishes_log_only(
    tmp_path, monkeypatch
):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    before = _snapshot(archive)
    plan = comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)
    assert _snapshot(archive) == before  # planning writes nothing
    record = plan.comparison_record
    assert record["schema_version"] == 2
    assert record["comparison_profile"] == comparison.CITRUS_CROSSCHECK_PROFILE_ID
    blind = record["candidate_bindings"]["palette_blind_fit"]
    assert blind["fit_review_status"] == "awaiting_explicit_human_review"
    assert blind["scientific_recipe"]["recipe_id"] == "top_rim_preferred_v1"
    assert record["citrus_geometry"]["state"] == "declared_and_applied"
    assert record["citrus_geometry"]["reason_codes"] == []
    metrics = record["disagreement_metrics"]
    assert metrics["center_offset_native_px"] < 1.0
    assert abs(metrics["radius_delta_vs_citrus_physical_rim_px"]) < 1.0
    assert metrics["circle_iou_vs_citrus_physical_rim"] > 0.99
    gate = record["gate_disagreement"]
    assert gate["status"] == "measured"
    assert gate["row_count"] == 3
    assert gate["exclusive_disagreement_count"] == 0
    assert record["observed_features"]["semantic_compatibility"] == (
        "projected_edges_unresolved"
    )
    assert record["threshold_policy"]["thresholds"] is None
    assert record["decision"]["workflow_action"] == "log_only"
    assert record["decision"]["pass_fail_flag_emitted"] is False
    assert not {"passed", "pass", "thresholds_satisfied", "agreement"} & set(
        _keys(record)
    )

    result = comparison.publish_arena_geometry_citrus_crosscheck(
        plan, scratch_root=tmp_path / "scratch"
    )
    assert result["published"] is True
    root = open_zarr_root(archive, mode="r")
    parent = root[f"analysis/{comparison.COMPARISON_RUNS_PARENT}"]
    run = parent[plan.comparison_id]
    assert plan.comparison_id.startswith("arena_geometry_citrus_crosscheck_")
    assert run.attrs["schema_id"] == comparison.CITRUS_CROSSCHECK_RUN_SCHEMA_ID
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["citrus_geometry_replaced"] is False
    assert "latest" not in parent.attrs and "latest_complete" not in parent.attrs
    after = _snapshot(archive)
    for path, payload in before.items():
        if str(path).startswith(_SOURCE_PREFIXES):
            assert after[path] == payload, path
    again = comparison.publish_arena_geometry_citrus_crosscheck(
        plan, scratch_root=tmp_path / "scratch"
    )
    assert again["status"] == "already_complete"


def test_crosscheck_is_not_a_reviewed_comparison_for_readiness_or_selection(
    tmp_path, monkeypatch
):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    plan = comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)
    comparison.publish_arena_geometry_citrus_crosscheck(
        plan, scratch_root=tmp_path / "scratch"
    )
    root = open_zarr_root(archive, mode="r")
    projected = {
        item.step_name: item
        for item in project_registered_geometry_stages(
            root=root,
            analysis_group=root["analysis"],
            common_details={},
            raw_status="ok",
            calibration_status="ok",
            detect_status="ok",
            detect_quality_status="missing",
        )
    }
    assert projected["arena_geometry_comparison"].status != "ok"
    assert projected["arena_geometry_comparison"].run_name != plan.comparison_id
    with pytest.raises(ValueError, match="Unsupported arena-geometry comparison"):
        comparison.validate_arena_geometry_comparison_record(plan.comparison_record)
    with pytest.raises(ValueError):
        selection.build_arena_geometry_selection_plan(
            archive,
            candidate_run=arguments["acquisition_candidate_run"],
            selected_by="reviewer@example.org",
            decision_reason="must refuse cross-check evidence",
            comparison_run=plan.comparison_id,
        )


def test_disagreeing_citrus_dish_logs_raw_disagreement_without_a_verdict(
    tmp_path, monkeypatch
):
    archive, arguments = _crosscheck_archive(
        tmp_path, monkeypatch, record=_citrus_record(_DISAGREEING)
    )
    plan = comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)
    record = plan.comparison_record
    metrics = record["disagreement_metrics"]
    fitted = record["geometry"]["palette_blind_fit_boundary"]
    assert metrics["center_offset_x_native_px"] == pytest.approx(
        fitted["center_px"]["x"] - 286.2
    )
    assert metrics["center_offset_native_px"] == pytest.approx(30.0, abs=0.5)
    assert metrics["radius_delta_vs_citrus_physical_rim_px"] == pytest.approx(
        fitted["radius_px"] - 190.0
    )
    assert metrics["radius_delta_vs_citrus_detection_gate_px"] == pytest.approx(
        fitted["radius_px"] - 195.0
    )
    assert metrics["circle_iou_vs_citrus_physical_rim"] < 0.8
    gate = record["gate_disagreement"]
    assert gate["palette_only_count"] == 1
    assert gate["acquisition_only_count"] == 0
    assert gate["exclusive_disagreement_fraction"] == pytest.approx(1 / 3)
    # Large disagreement is logged, never converted to a verdict.
    assert record["decision"] == comparison._CITRUS_CROSSCHECK_DECISION
    assert record["citrus_geometry"]["state"] == "declared_and_applied"


def test_gate_disagreement_is_explicitly_not_measured_without_detections(
    tmp_path, monkeypatch
):
    archive, arguments = _crosscheck_archive(
        tmp_path, monkeypatch, with_detection=False
    )
    plan = comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)
    assert plan.comparison_record["gate_disagreement"] == {
        "status": "not_measured",
        "reason": "exact_detection_source_not_supplied",
    }
    assert plan.detect_source_group_path is None


def test_missing_citrus_candidate_is_refused(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    arguments["acquisition_candidate_run"] = "arena-geometry-acquisition-absent"
    with pytest.raises(ValueError, match="candidate is missing"):
        comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)


def test_non_citrus_acquisition_geometry_is_refused(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(
        tmp_path,
        monkeypatch,
        record=_citrus_record(source_kind="orange_recording_folder"),
    )
    with pytest.raises(ValueError, match="Citrus unified H5"):
        comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)


@pytest.mark.parametrize(
    "status",
    [
        "checksum_mismatch",
        "registration_id_mismatch",
        "rim_observation_mismatch",
        "invalid",
    ],
)
def test_invalid_citrus_geometry_is_refused(tmp_path, monkeypatch, status):
    archive, arguments = _crosscheck_archive(
        tmp_path, monkeypatch, record=_citrus_record(status=status, applied=None)
    )
    with pytest.raises(ValueError, match="Citrus H5 geometry is invalid"):
        comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)


def test_applied_claim_without_applied_flag_is_refused(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(
        tmp_path, monkeypatch, record=_citrus_record(applied=False)
    )
    with pytest.raises(ValueError, match="does not declare the selected registration"):
        comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)


@pytest.mark.parametrize("status", ["missing", "camera_arena_target_missing"])
def test_declared_but_unapplied_citrus_geometry_is_recorded_explicitly(
    tmp_path, monkeypatch, status
):
    archive, arguments = _crosscheck_archive(
        tmp_path, monkeypatch, record=_citrus_record(status=status, applied=None)
    )
    plan = comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)
    citrus = plan.comparison_record["citrus_geometry"]
    assert citrus["state"] == "declared_not_proven_applied"
    assert citrus["reason_codes"] == [f"citrus_registration_{status}"]
    assert citrus["citrus_registration_status"] == status
    tampered = deepcopy(dict(plan.comparison_record))
    tampered["citrus_geometry"]["state"] = "declared_and_applied"
    with pytest.raises(ValueError, match="state disagrees"):
        comparison.validate_arena_geometry_citrus_crosscheck_record(tampered)


def test_legacy_recipe_blind_fit_is_refused(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch, recipe="legacy_v1")
    with pytest.raises(ValueError, match="top_rim_preferred_v1"):
        comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)


def test_tampered_blind_fit_package_is_refused(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    root = open_zarr_root(archive, mode="a")
    path = f"analysis/{fit_review.FIT_REVIEW_RUNS_PARENT}/{arguments['fit_review_run']}"
    root[path][fit_review.FIT_REPORT_ARRAY][0] = 0
    with pytest.raises(ValueError, match="Fit-review run is invalid"):
        comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)


def test_tampered_citrus_candidate_is_refused(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    root = open_zarr_root(archive, mode="a")
    run = root[
        f"analysis/{candidates.CANDIDATE_RUNS_PARENT}/"
        f"{arguments['acquisition_candidate_run']}"
    ]
    record = deepcopy(dict(run.attrs["candidate_record"]))
    record["physical_inner_rim"]["geometry"]["radius_px"] = 150.0
    record["valid_detection_region"]["geometry"]["radius_px"] = 155.0
    run.attrs["candidate_record"] = record
    with pytest.raises(ValueError, match="digest is invalid"):
        comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)


def test_wrong_live_coordinate_binding_is_refused(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    coordinate, arena, video = _binding_512()
    coordinate["pixel_frame_record_sha256"] = "f" * 64
    monkeypatch.setattr(
        candidates,
        "_source_camera_candidate_binding",
        lambda *_a, **_k: (coordinate, arena, video),
    )
    with pytest.raises(ValueError, match="coordinate/arena bindings disagree"):
        comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)


def test_source_change_between_plan_and_publish_is_refused(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    plan = comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)
    root = open_zarr_root(archive, mode="a")
    root[arguments["detect_source_group_path"]]["instances"]["bbox_norm_coords"][
        1, 0
    ] = 0.9
    with pytest.raises(RuntimeError, match="Detection source changed"):
        comparison.publish_arena_geometry_citrus_crosscheck(
            plan, scratch_root=tmp_path / "scratch"
        )
    assert not plan.target_run_path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "threshold",
        "pass_flag",
        "same_feature",
        "metric_nan",
        "selection",
        "extra_metric",
    ],
)
def test_record_validator_refuses_verdicts_and_tampering(
    tmp_path, monkeypatch, mutation
):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    plan = comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)
    record = deepcopy(dict(plan.comparison_record))
    if mutation == "threshold":
        record["threshold_policy"]["thresholds"] = {"maximum_center_offset_px": 5.0}
    elif mutation == "pass_flag":
        record["decision"]["passed"] = True
    elif mutation == "same_feature":
        record["observed_features"]["semantic_compatibility"] = "same_feature_confirmed"
    elif mutation == "metric_nan":
        record["disagreement_metrics"]["center_offset_native_px"] = float("nan")
    elif mutation == "selection":
        record["decision"]["candidate_selected"] = True
    elif mutation == "extra_metric":
        record["disagreement_metrics"]["agreement"] = 1.0
    with pytest.raises(ValueError):
        comparison.validate_arena_geometry_citrus_crosscheck_record(record)


def test_published_crosscheck_tamper_is_detected(tmp_path, monkeypatch):
    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    plan = comparison.build_arena_geometry_citrus_crosscheck_plan(archive, **arguments)
    comparison.publish_arena_geometry_citrus_crosscheck(
        plan, scratch_root=tmp_path / "scratch"
    )
    root = open_zarr_root(archive, mode="a")
    run = root[f"analysis/{comparison.COMPARISON_RUNS_PARENT}/{plan.comparison_id}"]
    record = deepcopy(dict(run.attrs["comparison_record"]))
    record["disagreement_metrics"]["center_offset_native_px"] = 0.0
    run.attrs["comparison_record"] = record
    result = comparison.validate_arena_geometry_citrus_crosscheck_run(
        plan.target_run_path, expected_plan=plan, require_complete=True
    )
    assert result["valid"] is False
    with pytest.raises(FileExistsError):
        comparison.publish_arena_geometry_citrus_crosscheck(
            plan, scratch_root=tmp_path / "scratch"
        )


def test_cli_plans_then_publishes_crosscheck(tmp_path, monkeypatch, capsys):
    from fisheye.utils.publish_arena_geometry_citrus_crosscheck import main

    archive, arguments = _crosscheck_archive(tmp_path, monkeypatch)
    argv = [
        str(archive),
        "--acquisition-candidate-run",
        arguments["acquisition_candidate_run"],
        "--fit-review-run",
        arguments["fit_review_run"],
        "--detect-source-group",
        arguments["detect_source_group_path"],
        "--scratch-root",
        str(tmp_path / "scratch"),
    ]
    assert main(argv) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["status"] == "planned"
    assert planned["decision"]["workflow_action"] == "log_only"
    expected = ["--expected-comparison-run", planned["comparison_id"], "--apply"]
    assert main([*argv, *expected]) == 0
    published = json.loads(capsys.readouterr().out)
    assert published["published"] is True
    assert published["comparison_id"] == planned["comparison_id"]
