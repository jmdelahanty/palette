from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.stimulus_response import (
    ProtocolStep,
    _write_stimulus_response_compact_v3,
)
from fisheye.analysis.stimulus_response_io import (
    resolve_stimulus_response_v3_tables,
)
from fisheye.analysis.stimulus_response_storage import (
    STIMULUS_RESPONSE_CANDIDATE_ATTR,
    STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR,
    STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR,
    consolidate_and_validate_stimulus_response_metadata,
    stimulus_response_fill_values,
    validate_stimulus_response_metadata_equivalence,
    validate_stimulus_response_storage_receipt,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from fisheye.shared.zarr.stimulus_response_schema import (
    STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR,
    STIMULUS_RESPONSE_ARRAY_SCHEMA_VERSION,
    STIMULUS_RESPONSE_BYTE_PLANNED_ARRAY_SCHEMA_VERSION,
    STIMULUS_RESPONSE_LAYOUT,
    STIMULUS_RESPONSE_SCHEMA_ID,
    STIMULUS_RESPONSE_SCHEMA_VERSION,
    stimulus_response_array_declarations,
    stimulus_response_array_manifest,
    validate_stimulus_response_v3_run,
)


def _global() -> dict[str, np.ndarray]:
    return {
        "fish_id": np.asarray([4, 9], dtype=np.int32),
        "total_distance_mm": np.asarray([1.0, 2.0], dtype=np.float32),
        "mean_speed_mm_s": np.asarray([3.0, 4.0], dtype=np.float32),
        "total_active_s": np.asarray([5.0, 6.0], dtype=np.float32),
        "fraction_moving": np.asarray([0.5, 0.75], dtype=np.float32),
    }


def _step_metrics() -> dict[str, np.ndarray]:
    return {
        "fish_id": np.asarray([4, 9], dtype=np.int32),
        "total_distance_mm": np.asarray([1.0, 2.0], dtype=np.float32),
        "mean_speed_mm_s": np.asarray([3.0, 4.0], dtype=np.float32),
        "median_speed_mm_s": np.asarray([3.0, 4.0], dtype=np.float32),
        "max_speed_mm_s": np.asarray([5.0, 6.0], dtype=np.float32),
        "fraction_moving": np.asarray([0.5, 0.75], dtype=np.float32),
        "coverage": np.asarray([1.0, 1.0], dtype=np.float32),
    }


def _candidate(
    path: Path,
    *,
    frame_count: int = 0,
    byte_planned: bool = True,
) -> tuple[zarr.Group, zarr.Group]:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    run = (
        root.require_group("analysis")
        .require_group("stimulus_response_runs")
        .create_group("candidate")
    )
    run.attrs.update(
        {
            "schema_id": STIMULUS_RESPONSE_SCHEMA_ID,
            "schema_version": STIMULUS_RESPONSE_SCHEMA_VERSION,
            "layout": STIMULUS_RESPONSE_LAYOUT,
            "stage_selector_eligible": False,
            "palette_run_completion_status": "complete",
        }
    )
    frame_annotations = None
    if frame_count:
        frame_annotations = {
            "step_index": np.zeros(frame_count, dtype=np.int32),
            "stimulus_mode_id": np.full(frame_count, 7, dtype=np.int32),
        }
    _write_stimulus_response_compact_v3(
        run,
        global_metrics=_global(),
        steps=[ProtocolStep(0, "baseline", "SOLID_BLACK", 7, 0, 20, 2.0)],
        step_metrics=[_step_metrics()],
        frame_annotations=frame_annotations,
        step_bout_metrics=None,
        step_grating_data=None,
        step_concentric_data=None,
        step_loom_data=None,
        global_omr_metrics=None,
        storage_profile=PUBLISHED_HTTP_V1 if byte_planned else None,
    )
    return root, run


def test_candidate_uses_exact_byte_planned_declarations_and_semantic_fills(
    tmp_path: Path,
) -> None:
    _root, run = _candidate(tmp_path / "candidate.zarr", frame_count=300_000)

    assert validate_stimulus_response_v3_run(run) == ()
    assert validate_stimulus_response_storage_receipt(run) == ()
    assert run.attrs[STIMULUS_RESPONSE_CANDIDATE_ATTR] == {
        "schema_id": "palette.stimulus_response.storage_candidate",
        "schema_version": 1,
        "profile_id": "published_http_v1",
        "status": "unpromoted_selector_ineligible",
        "write_ownership": "serial_single_writer_whole_shard",
    }
    assert run.attrs[STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR]["schema_version"] == (
        STIMULUS_RESPONSE_BYTE_PLANNED_ARRAY_SCHEMA_VERSION
    )
    receipt = run.attrs[STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR]
    entries = {entry["path"]: entry for entry in receipt["payload"]["arrays"]}
    assert entries["step_index/stimulus_params_json"]["observed_facts"][
        "access_unit_shape"
    ] == [1, 16384]
    assert (
        entries["frame_annotations/step_index"]["declaration"]["access_pattern"]
        == "windowed"
    )
    assert (
        entries["global_per_fish/mean_speed_mm_s"]["declaration"]["access_pattern"]
        == "eager"
    )
    assert run["frame_annotations/step_index"].metadata.fill_value == -1
    assert math.isnan(run["global_per_fish/mean_speed_mm_s"].metadata.fill_value)


def test_legacy_v3_manifest_and_physical_fills_remain_compatible(
    tmp_path: Path,
) -> None:
    _root, run = _candidate(
        tmp_path / "legacy-v3.zarr",
        byte_planned=False,
    )
    assert validate_stimulus_response_v3_run(run) == ()
    assert run.attrs[STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR]["schema_version"] == (
        STIMULUS_RESPONSE_ARRAY_SCHEMA_VERSION
    )
    assert run["global_per_fish/mean_speed_mm_s"].metadata.fill_value == 0.0

    frozen_manifest = stimulus_response_array_manifest(
        bundles=("moving_grating_omr", "looming"),
        byte_planner_adopted=False,
    )
    assert set(frozen_manifest) == {
        "schema_id",
        "schema_version",
        "run_schema_id",
        "run_schema_version",
        "layout",
        "bundles",
        "arrays",
    }
    assert canonical_json_sha256(frozen_manifest) == (
        "e741db1d8431d1258cbf6ec646e8b8f585c6bd8636f8897ed02d55e7f2423613"
    )


def test_semantic_fill_registry_covers_labels_counts_quality_and_text() -> None:
    fills = stimulus_response_fill_values(
        bundles=("moving_grating_omr", "looming"),
    )
    assert fills["step_index/step_name"] == 0
    assert fills["step_index/step_index"] == -1
    assert fills["moving_grating_omr_per_fish/valid_transition_count"] == 0
    assert fills["global_omr_per_fish/total_bouts"] == 0
    assert fills["global_omr_per_fish/total_bout_correct"] == 0
    assert fills["global_omr_per_fish/total_bout_opposing"] == 0
    assert fills["global_omr_per_fish/total_bout_ambiguous"] == 0
    assert fills["moving_grating_omr_per_bout/correct_label"] == 0
    assert fills["moving_grating_omr_per_bout/quality_flag"] == 1
    assert fills["looming_per_trial_per_fish/escaped"] is False
    assert math.isnan(fills["looming_per_fish/escape_probability"])


def test_table_specific_axes_do_not_claim_false_cross_table_row_equality() -> None:
    declarations = stimulus_response_array_declarations(
        bundles=(
            "moving_grating",
            "moving_grating_omr",
            "concentric_grating",
            "concentric_radial_omr",
            "looming",
        ),
        byte_planner_adopted=True,
    )
    axes = {
        declaration.path.split("/", 1)[0]: declaration.contract.axis_names[0]
        for declaration in declarations
    }
    assert axes["step_per_fish"] == "step_fish_rows"
    assert axes["grating_per_fish"] == "grating_fish_rows"
    assert axes["moving_grating_omr_per_fish"] == ("moving_grating_omr_fish_rows")
    assert axes["concentric_per_fish"] == "concentric_fish_rows"
    assert axes["looming_per_fish"] == "looming_fish_rows"


def test_recomputed_receipt_tampering_is_rejected(tmp_path: Path) -> None:
    _root, run = _candidate(tmp_path / "candidate.zarr")
    receipt = copy.deepcopy(run.attrs[STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR])
    receipt["payload"]["arrays"][0]["plan"]["chunk_shape"][0] += 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    run.attrs[STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR] = receipt

    errors = validate_stimulus_response_storage_receipt(run)

    assert any("invalid" in error or "differs" in error for error in errors)


def test_consolidated_metadata_equality_is_required_by_strict_candidate_reader(
    tmp_path: Path,
) -> None:
    root, run = _candidate(tmp_path / "candidate.zarr")
    try:
        resolve_stimulus_response_v3_tables(run)
    except ValueError as exc:
        assert "metadata-equivalence" in str(exc)
    else:  # pragma: no cover - protects the fail-closed assertion
        raise AssertionError("Candidate reader accepted missing equivalence evidence.")

    receipt = consolidate_and_validate_stimulus_response_metadata(
        root,
        run_path="analysis/stimulus_response_runs/candidate",
    )

    assert receipt["payload"]["result"] == ("direct_and_consolidated_metadata_equal")
    direct_root = zarr.open_group(root.store, mode="r", use_consolidated=False)
    consolidated_root = zarr.open_group(root.store, mode="r", use_consolidated=True)
    direct_run = direct_root["analysis/stimulus_response_runs/candidate"]
    consolidated_run = consolidated_root["analysis/stimulus_response_runs/candidate"]
    assert (
        direct_run.attrs[STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR]
        == consolidated_run.attrs[STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR]
    )
    resolved = resolve_stimulus_response_v3_tables(consolidated_run)
    assert resolved.layout == STIMULUS_RESPONSE_LAYOUT


def test_forged_or_stale_metadata_equivalence_is_rejected(tmp_path: Path) -> None:
    root, run = _candidate(tmp_path / "candidate.zarr")
    run.attrs[STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR] = {
        "result": "direct_and_consolidated_metadata_equal"
    }
    assert validate_stimulus_response_metadata_equivalence(run) == (
        "metadata-equivalence receipt is absent or not exact",
    )
    try:
        resolve_stimulus_response_v3_tables(run)
    except ValueError as exc:
        assert "metadata-equivalence receipt" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Forged metadata-equivalence evidence was accepted.")

    consolidate_and_validate_stimulus_response_metadata(
        root,
        run_path="analysis/stimulus_response_runs/candidate",
    )
    run = zarr.open_group(
        root.store,
        mode="a",
        use_consolidated=False,
    )["analysis/stimulus_response_runs/candidate"]
    run.attrs["tampered_after_consolidation"] = True
    assert "metadata-equivalence receipt is stale" in (
        validate_stimulus_response_metadata_equivalence(run)
    )


def test_candidate_markers_cannot_be_downgraded_or_hide_visualizations(
    tmp_path: Path,
) -> None:
    _root, run = _candidate(tmp_path / "candidate.zarr")
    run.create_group("visualizations")
    assert any(
        "visualizations" in error for error in validate_stimulus_response_v3_run(run)
    )
    del run["visualizations"]
    del run.attrs["analysis_storage_profile_role"]
    run.attrs["stimulus_response_array_schema"] = stimulus_response_array_manifest(
        bundles=(),
        byte_planner_adopted=False,
    )
    errors = validate_stimulus_response_v3_run(run)
    assert any("candidate marker set is incomplete" in error for error in errors)
    assert any("byte-planner metadata" in error for error in errors)


def test_live_physical_fill_tampering_is_rejected(tmp_path: Path) -> None:
    store_path = tmp_path / "candidate.zarr"
    _root, _run = _candidate(store_path, frame_count=4)
    metadata_path = (
        store_path
        / "analysis"
        / "stimulus_response_runs"
        / "candidate"
        / "frame_annotations"
        / "step_index"
        / "zarr.json"
    )
    declaration = json.loads(metadata_path.read_text(encoding="utf-8"))
    declaration["fill_value"] = 0
    metadata_path.write_text(
        json.dumps(declaration, sort_keys=True),
        encoding="utf-8",
    )
    reopened = zarr.open_group(
        str(store_path),
        mode="r",
        use_consolidated=False,
    )["analysis/stimulus_response_runs/candidate"]
    assert any(
        "array metadata differs" in error
        for error in validate_stimulus_response_storage_receipt(reopened)
    )
