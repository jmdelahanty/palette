from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    ComposableChaserSuccessorPublicationError,
    build_composable_chaser_successor_publication_plan,
    load_composable_chaser_successor_source_handle,
    publish_composable_chaser_successor_run,
)
from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    ensure_exact_immutable_child_validation_receipt,
)
from fisheye.analysis_workflows.controller_trial_successor import (
    prepare_controller_trial_successor,
)
from fisheye.analysis_workflows.escape_freeze_successor import (
    prepare_escape_freeze_successor,
)
from fisheye.analysis_workflows.full_chaser_profile_successor import (
    prepare_full_chaser_profile_successor,
)
from fisheye.analysis_workflows.gaze_tracking_successor import (
    prepare_gaze_tracking_successor,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import (
    prepare_generalized_bout_response_successor,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from tests.unit.fisheye.test_controller_trial_successor import _source as _trial_source
from tests.unit.fisheye.test_escape_freeze_successor import _source as _escape_source
from tests.unit.fisheye.test_full_chaser_profile_successor import (
    _plan as _full_plan,
    _products as _full_products,
)
from tests.unit.fisheye.test_gaze_tracking_successor import _source as _gaze_source
from tests.unit.fisheye.test_generalized_bout_response_successor import (
    _source as _bout_source,
)


def _archive(tmp_path):
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording-1"
    return archive


def test_controller_trial_publication_is_immutable_and_strictly_readable(tmp_path) -> None:
    archive = _archive(tmp_path)
    prepared = prepare_controller_trial_successor(_trial_source())
    plan = build_composable_chaser_successor_publication_plan(
        archive,
        run_name="controller-v1",
        prepared=prepared,
    )
    receipt = publish_composable_chaser_successor_run(
        plan,
        scratch_root=tmp_path / "scratch",
    )

    assert receipt["status"] == "published_selector_ineligible"
    assert receipt["successor_kind"] == "controller_chase_trials"
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="controller_chase_trials",
        run_name="controller-v1",
        expected_recording_id="recording-1",
        deep_audit=True,
    )
    assert handle.scientific_payload_sha256 == prepared.payload_digest
    np.testing.assert_array_equal(
        handle.array("logged_trial_id"), prepared.array("logged_trial_id")
    )
    binding = handle.module_product_binding(module_id="controller_chase_trials")
    assert binding.schema_id == "palette.analysis.controller_chase_trials"
    assert binding.run_path == handle.run_path
    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    parent = root["analysis/controller_chase_trial_runs"]
    assert not {"latest", "latest_complete", "selected", "authoritative"} & set(
        parent.attrs
    )


def test_deep_audit_rejects_array_tampering(tmp_path) -> None:
    archive = _archive(tmp_path)
    prepared = prepare_controller_trial_successor(_trial_source())
    plan = build_composable_chaser_successor_publication_plan(
        archive, run_name="controller-v1", prepared=prepared
    )
    publish_composable_chaser_successor_run(plan, scratch_root=tmp_path / "scratch")
    root = open_zarr_root(archive, mode="a", use_consolidated=False)
    values = root[f"{plan.run_path}/logged_trial_id"]
    values[0] = 999
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(ComposableChaserSuccessorPublicationError, match="content digest"):
        load_composable_chaser_successor_source_handle(
            archive,
            successor_kind="controller_chase_trials",
            run_name="controller-v1",
            deep_audit=True,
        )


def test_all_successor_types_have_distinct_immutable_parents(tmp_path) -> None:
    archive = _archive(tmp_path)
    controller = prepare_controller_trial_successor(_trial_source())
    bout = prepare_generalized_bout_response_successor(
        replace(
            _bout_source(),
            source_controller_trial_payload_sha256=controller.payload_digest,
        )
    )
    gaze = prepare_gaze_tracking_successor(_gaze_source())
    escape = prepare_escape_freeze_successor(_escape_source())
    profile, selected, applicability = _full_plan(complete=True)
    full = prepare_full_chaser_profile_successor(
        profile=profile,
        applicability=applicability,
        products=_full_products(selected),
    )

    plans = [
        build_composable_chaser_successor_publication_plan(
            archive, run_name=f"candidate-{index}", prepared=prepared
        )
        for index, prepared in enumerate(
            (controller, bout, gaze, escape, full), start=1
        )
    ]
    assert [plan.successor_kind for plan in plans] == [
        "controller_chase_trials",
        "generalized_chaser_bout_response",
        "chaser_gaze_tracking",
        "chaser_escape_freeze",
        "chaser_full_profile",
    ]
    assert len({plan.parent_path for plan in plans}) == 5
    assert all(plan.manifest["selector_eligible"] is False for plan in plans)


def test_selector_alias_run_name_is_rejected(tmp_path) -> None:
    archive = _archive(tmp_path)
    with pytest.raises(ComposableChaserSuccessorPublicationError, match="non-selector"):
        build_composable_chaser_successor_publication_plan(
            archive,
            run_name="latest",
            prepared=prepare_controller_trial_successor(_trial_source()),
        )


def test_full_profile_binding_requires_deep_array_audit(tmp_path) -> None:
    archive = _archive(tmp_path)
    prepared = prepare_controller_trial_successor(_trial_source())
    plan = build_composable_chaser_successor_publication_plan(
        archive, run_name="controller-v1", prepared=prepared
    )
    publish_composable_chaser_successor_run(plan, scratch_root=tmp_path / "scratch")
    shallow = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="controller_chase_trials",
        run_name="controller-v1",
    )

    with pytest.raises(ComposableChaserSuccessorPublicationError, match="deep-audited"):
        shallow.module_product_binding(module_id="controller_chase_trials")


def test_deep_audited_handle_rehydrates_exact_prepared_dependency(tmp_path) -> None:
    archive = _archive(tmp_path)
    prepared = prepare_controller_trial_successor(_trial_source())
    plan = build_composable_chaser_successor_publication_plan(
        archive, run_name="controller-v1", prepared=prepared
    )
    publish_composable_chaser_successor_run(plan, scratch_root=tmp_path / "scratch")
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="controller_chase_trials",
        run_name="controller-v1",
        deep_audit=True,
    )

    reused = handle.prepared_successor()

    assert reused.payload_digest == prepared.payload_digest
    np.testing.assert_array_equal(
        reused.array("trial_row_id_by_source_row"),
        prepared.array("trial_row_id_by_source_row"),
    )
    with pytest.raises(TypeError):
        handle.scientific_manifest["policy"]["fallback"] = "infer"  # type: ignore[index]


def test_exact_child_receipt_avoids_root_parse_and_rejects_array_change(
    tmp_path,
) -> None:
    archive = _archive(tmp_path)
    prepared = prepare_controller_trial_successor(_trial_source())
    plan = build_composable_chaser_successor_publication_plan(
        archive, run_name="controller-v1", prepared=prepared
    )
    publish_composable_chaser_successor_run(
        plan,
        scratch_root=tmp_path / "scratch",
    )
    receipt_path = tmp_path / "exact-child-receipt.json"
    receipt = ensure_exact_immutable_child_validation_receipt(
        archive,
        run_path=plan.run_path,
        manifest_attr="composable_chaser_successor_manifest",
        manifest_digest_attr="composable_chaser_successor_manifest_sha256",
        palette_commit="a" * 40,
        output_json=receipt_path,
        expected_recording_id="recording-1",
    )

    (archive / "zarr.json").write_text("root metadata must not be parsed")
    handle = load_composable_chaser_successor_source_handle(
        archive,
        successor_kind="controller_chase_trials",
        run_name="controller-v1",
        expected_recording_id="recording-1",
        deep_audit=True,
        direct_validation_receipt=receipt_path,
    )
    assert handle.scientific_payload_sha256 == prepared.payload_digest
    assert handle.metadata_equivalence["receipt_sha256"] == receipt["record_sha256"]
    assert handle.metadata_equivalence[
        "archive_root_consolidated_metadata_reparse"
    ] is False

    child = open_zarr_root(
        archive / plan.run_path,
        mode="a",
        use_consolidated=False,
    )
    child["logged_trial_id"][0] = 999
    with pytest.raises(ComposableChaserSuccessorPublicationError, match="content digest"):
        load_composable_chaser_successor_source_handle(
            archive,
            successor_kind="controller_chase_trials",
            run_name="controller-v1",
            expected_recording_id="recording-1",
            deep_audit=True,
            direct_validation_receipt=receipt_path,
        )

    metadata_path = archive / plan.run_path / "zarr.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["attributes"]["receipt_tamper_marker"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(
        ComposableChaserSuccessorPublicationError,
        match="metadata generation changed",
    ):
        load_composable_chaser_successor_source_handle(
            archive,
            successor_kind="controller_chase_trials",
            run_name="controller-v1",
            expected_recording_id="recording-1",
            deep_audit=True,
            direct_validation_receipt=receipt_path,
        )
