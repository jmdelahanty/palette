from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis.chaser_distance_base_schema import (
    ARRAY_COORDINATE_SPACES,
    CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID,
    SEALED_CHASER_DISTANCE_BASE_PATHS,
    build_chaser_distance_base_declarations,
)
from fisheye.analysis.chaser_distance_base_storage import (
    BASE_LOGICAL_HASHES_ATTR,
    BASE_MANIFEST_ATTR,
    BASE_MANIFEST_DIGEST_ATTR,
    base_logical_hashes,
    validate_base_candidate,
)
from fisheye.analysis_workflows.materializers import (
    chaser_distance_base as materializer,
)
from fisheye.analysis_workflows.materializers.chaser_distance_base import (
    CHASER_DISTANCE_EXECUTION_PHASE_ORDER,
    build_chaser_distance_base_candidate_plan,
    materialize_chaser_distance_base_candidate,
    tombstone_chaser_distance_execution_candidate,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings


def _write(group: zarr.Group, path: str, data: np.ndarray) -> None:
    parent = group
    components = path.split("/")
    for component in components[:-1]:
        parent = parent.require_group(component)
    parent.create_array(components[-1], data=np.asarray(data), overwrite=False)


def _text(rows: list[str], width: int = 96) -> np.ndarray:
    out = np.zeros((len(rows), width), dtype=np.uint8)
    for index, text in enumerate(rows):
        value = text.encode("utf-8")
        out[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    return out


def _archive(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w", zarr_format=3, use_consolidated=False)
    parent = root.require_group("analysis/chaser_distance_runs")
    parent.attrs.update(
        {
            "latest": "source",
            "latest_complete": "source",
            "latest_pending": "pending-source",
            "authoritative_run": "source",
            "authoritative_run_provenance": {"manifest_sha256": "a" * 64},
            "publication_policy": "owner_generation_guarded_selectors_then_eligibility_v1",
            "publication_generation": 7,
            "chaser_distance_publication_lease": {"owner": "test"},
        }
    )
    run = parent.create_group("source")
    frames, chasers, windows, bins = 6, 2, 2, 3
    frame_index = np.arange(frames, dtype=np.int64)
    fish = np.column_stack((frame_index, frame_index + 1)).astype(np.float32)
    chaser = np.stack((fish + 2, fish + 4), axis=1).astype(np.float32)
    distance = np.linalg.norm(chaser - fish[:, None, :], axis=2).astype(np.float32)
    arrays = {
        "stimulus_state_key": frame_index,
        "frames/camera_frame_id": frame_index,
        "frames/stimulus_frame_num": frame_index + 10,
        "frames/timestamp_ns": frame_index * 1_000,
        "frames/stimulus_epoch_window_id": np.asarray(
            [0, 0, 0, 1, 1, 1], dtype=np.int32
        ),
        "chasers/chaser_index": np.asarray([0, 1], dtype=np.int16),
        "chasers/stimulus_instance_id_bytes": _text(["chaser:0", "chaser:1"]),
        "chasers/source_track_key_bytes": _text(["chaser_index:0", "chaser_index:1"]),
        "positions/source_detection_row_index": frame_index + 20,
        "positions/fish_centroid_img_xy": fish,
        "positions/fish_centroid_arena_xy": fish + np.float32(0.5),
        "positions/chaser_arena_xy": chaser,
        "positions/fish_valid": np.ones(frames, dtype=bool),
        "positions/chaser_valid": np.ones((frames, chasers), dtype=bool),
        "distances/distance_px": distance,
        "distances/distance_mm": distance / np.float32(2),
        "distances/nearest_chaser_index": np.zeros(frames, dtype=np.int16),
        "distances/nearest_distance_mm": distance[:, 0] / np.float32(2),
        "epoch_summary/window_id": np.asarray([0, 1], dtype=np.int32),
        "epoch_summary/label_bytes": _text(["pre_event", "post_event"]),
        "epoch_summary/start_frame": np.asarray([0, 3], dtype=np.int64),
        "epoch_summary/end_frame": np.asarray([2, 5], dtype=np.int64),
        "epoch_summary/mean_distance_mm": np.ones((windows, chasers), dtype=np.float32),
        "epoch_summary/min_distance_mm": np.ones((windows, chasers), dtype=np.float32),
        "epoch_summary/p05_distance_mm": np.ones((windows, chasers), dtype=np.float32),
        "epoch_summary/p50_distance_mm": np.ones((windows, chasers), dtype=np.float32),
        "epoch_summary/p95_distance_mm": np.ones((windows, chasers), dtype=np.float32),
        "epoch_distributions/bin_edges_mm": np.arange(bins + 1, dtype=np.float32),
        "epoch_distributions/bin_centers_mm": np.arange(bins, dtype=np.float32)
        + np.float32(0.5),
        "epoch_distributions/hist_density": np.ones(
            (windows, chasers, bins), dtype=np.float32
        ),
    }
    assert set(arrays) == set(SEALED_CHASER_DISTANCE_BASE_PATHS)
    for array_path, values in arrays.items():
        _write(run, array_path, values)
    # These are intentionally present but outside the sealed base projection.
    _write(run, "chasers/behavior_class_id", np.zeros(chasers, dtype=np.int8))
    _write(
        run,
        "epoch_summary/valid_frame_count",
        np.ones((windows, chasers), dtype=np.int64),
    )
    run.attrs.update(
        {
            "schema_id": "palette.chaser_distance.v1",
            "schema_version": 1,
            "coordinate_publication_status": "sealed_canonical_v2",
            "stage_selector_eligible": True,
            "palette_run_completion_status": "complete",
        }
    )
    consolidate_metadata_capture_expected_warnings(path)
    return root


def _record(
    attribute_name: str,
    record: dict | None = None,
    *,
    node_path: str = "analysis/chaser_distance_runs/source",
) -> SimpleNamespace:
    return SimpleNamespace(
        record_ref=f"/{node_path}@{attribute_name}",
        record_sha256=(attribute_name[0].encode().hex()[0] if attribute_name else "a")
        * 64,
        record={} if record is None else record,
    )


def _bound() -> SimpleNamespace:
    protected = {
        path: {}
        for path in (
            "stimulus_state_key",
            "frames/camera_frame_id",
            "frames/stimulus_frame_num",
            "frames/timestamp_ns",
            "frames/stimulus_epoch_window_id",
            "positions/source_detection_row_index",
            "positions/fish_centroid_img_xy",
            "positions/fish_centroid_arena_xy",
            "positions/chaser_arena_xy",
            "positions/fish_valid",
            "positions/chaser_valid",
            "distances/distance_px",
            "distances/distance_mm",
            "distances/nearest_chaser_index",
            "distances/nearest_distance_mm",
            "chasers/chaser_index",
            "chasers/stimulus_instance_id_bytes",
            "chasers/source_track_key_bytes",
        )
    }
    epoch = {
        name: {} for name in ("window_id", "label_bytes", "start_frame", "end_frame")
    }
    measurement_paths = (
        "distances/distance_px",
        "distances/distance_mm",
        "distances/nearest_distance_mm",
        "epoch_summary/mean_distance_mm",
        "epoch_summary/min_distance_mm",
        "epoch_summary/p05_distance_mm",
        "epoch_summary/p50_distance_mm",
        "epoch_summary/p95_distance_mm",
        "epoch_distributions/bin_edges_mm",
        "epoch_distributions/bin_centers_mm",
        "epoch_distributions/hist_density",
    )
    coordinate_paths = (
        "positions/fish_centroid_img_xy",
        "positions/fish_centroid_arena_xy",
        "positions/chaser_arena_xy",
    )
    return SimpleNamespace(
        run_path="analysis/chaser_distance_runs/source",
        publication_seal=_record(
            "chaser_distance_publication_seal",
            {"protected_arrays": protected},
        ),
        surface_manifest=_record(
            "chaser_distance_surface_manifest",
            {
                "measurement_surfaces": {path: {} for path in measurement_paths},
                "coordinate_surfaces": {path: {} for path in coordinate_paths},
            },
        ),
        row_identity=_record("row_identity_contract"),
        input_authority=_record("chaser_distance_input_authority"),
        measurement_authority=_record("chaser_distance_measurement_authority"),
        chaser_collection=_record(
            "chaser_collection_authority",
            node_path="analysis/chaser_distance_runs/source/chasers",
        ),
        epoch_window_identity=_record(
            "epoch_window_identity_authority",
            {"published_arrays": epoch},
            node_path="analysis/chaser_distance_runs/source/epoch_summary",
        ),
    )


@pytest.fixture
def patched_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        materializer, "load_bound_chaser_distance_run", lambda *_a, **_k: _bound()
    )


def test_exact_declarations_freeze_all_sealed_semantics(tmp_path: Path) -> None:
    root = _archive(tmp_path / "schema.zarr")
    run = root["analysis/chaser_distance_runs/source"]
    declarations = build_chaser_distance_base_declarations(run)

    assert len(declarations) == 30
    assert (
        tuple(item.path for item in declarations) == SEALED_CHASER_DISTANCE_BASE_PATHS
    )
    by_path = {item.path: item for item in declarations}
    assert by_path["distances/distance_mm"].contract.dtype.numpy_dtype == "float32"
    assert by_path["distances/distance_mm"].contract.units == "mm"
    assert by_path["distances/distance_mm"].contract.coordinate_space is None
    assert by_path["distances/distance_px"].contract.coordinate_space is None
    assert by_path["distances/nearest_distance_mm"].contract.coordinate_space is None
    assert by_path["positions/source_detection_row_index"].fill_semantics.startswith(
        "-1"
    )
    assert (
        by_path["epoch_distributions/hist_density"].authority_role.value
        == "derived_cache"
    )
    assert all(
        item.contract.coordinate_space == ARRAY_COORDINATE_SPACES[item.path]
        for item in declarations
    )


def test_atomic_candidate_preserves_selectors_and_proves_metadata(
    tmp_path: Path,
    patched_bound: None,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _archive(archive)
    result = materialize_chaser_distance_base_candidate(
        archive,
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / "scratch",
        copy_backend="python",
        apply=True,
    )

    assert result["status"] == "complete"
    assert result["local_validation"]["array_count"] == 30
    assert result["local_direct_consolidated_array_count"] == 30
    assert result["archive_direct_consolidated_array_count"] == 30
    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    source_parent = direct["analysis/chaser_distance_runs"]
    assert source_parent.attrs["latest"] == "source"
    assert source_parent.attrs["latest_complete"] == "source"
    assert source_parent.attrs["latest_pending"] == "pending-source"
    assert source_parent.attrs["authoritative_run"] == "source"
    assert source_parent.attrs["authoritative_run_provenance"] == {
        "manifest_sha256": "a" * 64
    }
    assert source_parent.attrs["publication_policy"] == (
        "owner_generation_guarded_selectors_then_eligibility_v1"
    )
    assert source_parent.attrs["publication_generation"] == 7
    assert source_parent.attrs["chaser_distance_publication_lease"] == {"owner": "test"}
    candidate = direct["analysis/chaser_distance_storage_candidates/candidate"]
    assert candidate.attrs["schema_id"] == CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_profile_promoted"] is False
    assert validate_base_candidate(
        candidate,
        source_group=direct["analysis/chaser_distance_runs/source"],
        expected_source_binding=materializer.build_source_authority_binding(
            _bound(), source_group=direct["analysis/chaser_distance_runs/source"]
        ),
    )["valid"]
    consolidated_candidate = consolidated[
        "analysis/chaser_distance_storage_candidates/candidate"
    ]
    assert dict(consolidated_candidate.attrs) == dict(candidate.attrs)
    assert not (tmp_path / "scratch").exists()


def test_typed_execution_stages_source_and_can_be_tombstoned_exactly(
    tmp_path: Path,
    patched_bound: None,
) -> None:
    archive = tmp_path / "typed_analysis.zarr"
    root = _archive(archive)
    source = root["analysis/chaser_distance_runs/source"]
    expected_hashes = base_logical_hashes(source)
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "request_payload_digest": "b" * 64,
    }

    def accept(_root, _parent, candidate):
        assert candidate.attrs["analysis_candidate_execution_binding"] == binding
        assert candidate.attrs["source_staging_mode"] == ("sealed_base_logical_copy_v1")
        return {"accepted": True, "candidate_hashes": base_logical_hashes(candidate)}

    result = materialize_chaser_distance_base_candidate(
        archive,
        source_run="source",
        run_name="typed_candidate",
        scratch_root=tmp_path / "typed-scratch",
        copy_backend="python",
        apply=True,
        stage_source_to_scratch=True,
        execution_binding=binding,
        expected_source_logical_hashes=expected_hashes,
        publication_acceptance_validator=accept,
    )

    assert result["caller_acceptance"] == {
        "accepted": True,
        "candidate_hashes": expected_hashes,
    }
    assert result["source_logical_manifest_sha256"] == canonical_json_sha256(
        expected_hashes
    )
    assert result["published_logical_manifest_sha256"] == (
        result["source_logical_manifest_sha256"]
    )
    assert [phase["name"] for phase in result["runtime_telemetry"]["phases"]] == list(
        CHASER_DISTANCE_EXECUTION_PHASE_ORDER
    )
    assert result["output_storage"]["payload_file_count"] > 0
    assert not (tmp_path / "typed-scratch").exists()

    tombstone = tombstone_chaser_distance_execution_candidate(
        archive,
        run_name="typed_candidate",
        expected_execution_binding=binding,
        failure_phase="post_receipt_validation",
        error_type="RuntimeError",
        error_message="injected post-publication failure",
    )
    assert tombstone["tombstoned"] is True
    for use_consolidated in (False, True):
        candidate = zarr.open_group(
            str(archive), mode="r", use_consolidated=use_consolidated
        )["analysis/chaser_distance_storage_candidates/typed_candidate"]
        assert candidate.attrs["palette_run_completion_status"] == "failed"
        assert candidate.attrs["stage_selector_eligible"] is False
        assert candidate.attrs["storage_candidate_profile_promoted"] is False
        assert candidate.attrs["analysis_candidate_execution_binding"] == binding


def test_typed_execution_acceptance_failure_is_atomic_and_ineligible(
    tmp_path: Path,
    patched_bound: None,
) -> None:
    archive = tmp_path / "acceptance_failure_analysis.zarr"
    root = _archive(archive)
    expected_hashes = base_logical_hashes(root["analysis/chaser_distance_runs/source"])
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "request_payload_digest": "c" * 64,
    }

    def reject(_root, _parent, _candidate):
        raise RuntimeError("injected caller acceptance failure")

    with pytest.raises(
        RuntimeError, match="injected caller acceptance failure"
    ) as error:
        materialize_chaser_distance_base_candidate(
            archive,
            source_run="source",
            run_name="rejected_candidate",
            scratch_root=tmp_path / "rejected-scratch",
            copy_backend="python",
            apply=True,
            stage_source_to_scratch=True,
            execution_binding=binding,
            expected_source_logical_hashes=expected_hashes,
            publication_acceptance_validator=reject,
        )
    assert hasattr(error.value, "palette_runtime_telemetry")
    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    candidate = direct["analysis/chaser_distance_storage_candidates/rejected_candidate"]
    assert candidate.attrs["palette_run_completion_status"] == "failed"
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_profile_promoted"] is False
    source_parent = direct["analysis/chaser_distance_runs"]
    assert source_parent.attrs["latest"] == "source"
    assert source_parent.attrs["latest_complete"] == "source"


def _published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, zarr.Group]:
    monkeypatch.setattr(
        materializer, "load_bound_chaser_distance_run", lambda *_a, **_k: _bound()
    )
    archive = tmp_path / "tamper_analysis.zarr"
    _archive(archive)
    materialize_chaser_distance_base_candidate(
        archive,
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / "scratch-tamper",
        copy_backend="python",
        apply=True,
    )
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    return archive, root["analysis/chaser_distance_storage_candidates/candidate"]


def test_rehashed_array_omission_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive_path, candidate = _published(tmp_path, monkeypatch)
    del candidate["epoch_distributions/hist_density"]
    hashes = dict(candidate.attrs[BASE_LOGICAL_HASHES_ATTR])
    hashes.pop("epoch_distributions/hist_density")
    candidate.attrs[BASE_LOGICAL_HASHES_ATTR] = hashes
    manifest = copy.deepcopy(candidate.attrs[BASE_MANIFEST_ATTR])
    payload = manifest["payload"]
    payload["arrays"] = [
        item
        for item in payload["arrays"]
        if item["path"] != "epoch_distributions/hist_density"
    ]
    payload["array_paths"].remove("epoch_distributions/hist_density")
    payload["source_logical_hashes"].pop("epoch_distributions/hist_density")
    payload["candidate_logical_hashes"].pop("epoch_distributions/hist_density")
    manifest["payload_digest"] = canonical_json_sha256(payload)
    candidate.attrs[BASE_MANIFEST_ATTR] = manifest
    candidate.attrs[BASE_MANIFEST_DIGEST_ATTR] = manifest["payload_digest"]

    validation = validate_base_candidate(candidate)
    assert not validation["valid"]
    assert "hist_density" in " ".join(validation["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "promoted"),
        (
            "authority_boundary",
            {
                "included": "all live arrays",
                "excluded": [],
            },
        ),
    ],
)
def test_rehashed_manifest_policy_tampering_fails_complete_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    archive, candidate = _published(tmp_path, monkeypatch)
    manifest = copy.deepcopy(candidate.attrs[BASE_MANIFEST_ATTR])
    manifest["payload"][field] = value
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    candidate.attrs[BASE_MANIFEST_ATTR] = manifest
    candidate.attrs[BASE_MANIFEST_DIGEST_ATTR] = manifest["payload_digest"]
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    source = root["analysis/chaser_distance_runs/source"]
    validation = validate_base_candidate(
        candidate,
        source_group=source,
        expected_source_binding=materializer.build_source_authority_binding(
            _bound(), source_group=source
        ),
    )
    assert not validation["valid"]
    assert any("complete executable contract" in item for item in validation["errors"])


def test_rehashed_source_pointer_tampering_fails_external_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, candidate = _published(tmp_path, monkeypatch)
    source_binding = copy.deepcopy(
        candidate.attrs["chaser_distance_sealed_base_source_binding"]
    )
    source_binding["publication_seal"]["record_sha256"] = "f" * 64
    candidate.attrs["chaser_distance_sealed_base_source_binding"] = source_binding
    manifest = copy.deepcopy(candidate.attrs[BASE_MANIFEST_ATTR])
    manifest["payload"]["source_binding"] = source_binding
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    candidate.attrs[BASE_MANIFEST_ATTR] = manifest
    candidate.attrs[BASE_MANIFEST_DIGEST_ATTR] = manifest["payload_digest"]
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    source = root["analysis/chaser_distance_runs/source"]
    validation = validate_base_candidate(
        candidate,
        source_group=source,
        expected_source_binding=materializer.build_source_authority_binding(
            _bound(), source_group=source
        ),
    )
    assert not validation["valid"]
    assert any("verified source" in item for item in validation["errors"])


def test_rehashed_source_pointer_omission_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, candidate = _published(tmp_path, monkeypatch)
    source_binding = copy.deepcopy(
        candidate.attrs["chaser_distance_sealed_base_source_binding"]
    )
    source_binding.pop("surface_manifest")
    candidate.attrs["chaser_distance_sealed_base_source_binding"] = source_binding
    manifest = copy.deepcopy(candidate.attrs[BASE_MANIFEST_ATTR])
    manifest["payload"]["source_binding"] = source_binding
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    candidate.attrs[BASE_MANIFEST_ATTR] = manifest
    candidate.attrs[BASE_MANIFEST_DIGEST_ATTR] = manifest["payload_digest"]
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    source = root["analysis/chaser_distance_runs/source"]
    validation = validate_base_candidate(
        candidate,
        source_group=source,
        expected_source_binding=source_binding,
    )
    assert not validation["valid"]
    assert any("unexpected field set" in item for item in validation["errors"])


def test_payload_source_and_lifecycle_tampering_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, candidate = _published(tmp_path, monkeypatch)
    candidate["distances/distance_mm"][0, 0] = np.float32(999)
    assert not validate_base_candidate(candidate)["valid"]

    candidate.attrs["stage_selector_eligible"] = True
    candidate.attrs["storage_candidate_profile_promoted"] = True
    errors = validate_base_candidate(candidate)["errors"]
    assert any("selector-ineligible" in item for item in errors)
    assert any("unpromoted" in item for item in errors)

    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    source = root["analysis/chaser_distance_runs/source"]
    source["distances/distance_px"][0, 0] = np.float32(123)
    source_errors = validate_base_candidate(candidate, source_group=source)["errors"]
    assert any("sealed source" in item for item in source_errors)

    candidate.attrs["storage_candidate_source_run_path"] = (
        "analysis/chaser_distance_runs/other"
    )
    path_errors = validate_base_candidate(candidate)["errors"]
    assert any("source-run binding mismatch" in item for item in path_errors)


def test_plan_rejects_containment_and_aliases(
    tmp_path: Path, patched_bound: None
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _archive(archive)
    with pytest.raises(ValueError, match="disjoint"):
        build_chaser_distance_base_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate",
            scratch_root=archive / "scratch",
        )
    for name in ("latest", "../bad", "bad name"):
        with pytest.raises(ValueError):
            build_chaser_distance_base_candidate_plan(
                archive,
                source_run="source",
                run_name=name,
                scratch_root=tmp_path / f"scratch-{name.replace('/', '_')}",
            )


def test_copy_failure_leaves_source_pointers_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _archive(archive)
    monkeypatch.setattr(
        materializer, "load_bound_chaser_distance_run", lambda *_a, **_k: _bound()
    )
    from fisheye.analysis_workflows.materializers import atomic_run_publisher

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(atomic_run_publisher, "_copy_and_verify", fail_copy)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        materialize_chaser_distance_base_candidate(
            archive,
            source_run="source",
            run_name="failed_candidate",
            scratch_root=tmp_path / "scratch-failure",
            copy_backend="python",
            apply=True,
        )
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent = root["analysis/chaser_distance_runs"]
    assert parent.attrs["latest"] == "source"
    assert parent.attrs["latest_complete"] == "source"
    assert parent.attrs["latest_pending"] == "pending-source"
    assert parent.attrs["authoritative_run"] == "source"
    assert parent.attrs["authoritative_run_provenance"] == {"manifest_sha256": "a" * 64}
    assert parent.attrs["publication_policy"] == (
        "owner_generation_guarded_selectors_then_eligibility_v1"
    )
    assert parent.attrs["publication_generation"] == 7
    assert parent.attrs["chaser_distance_publication_lease"] == {"owner": "test"}
    candidate_parent = root["analysis/chaser_distance_storage_candidates"]
    assert "failed_candidate" not in candidate_parent


def test_source_dtype_tampering_is_rejected_before_planning(
    tmp_path: Path, patched_bound: None
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    root = _archive(archive)
    source = root["analysis/chaser_distance_runs/source"]
    values = np.asarray(source["distances/distance_mm"][:], dtype=np.float64)
    del source["distances/distance_mm"]
    _write(source, "distances/distance_mm", values)
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(ValueError, match="dtype mismatch"):
        build_chaser_distance_base_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "scratch-dtype",
        )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("schema_id", "palette.chaser_distance.v2"),
        ("schema_version", 2),
        ("coordinate_publication_status", "legacy_unsealed"),
        ("palette_run_completion_status", "running"),
        ("stage_selector_eligible", False),
    ],
)
def test_source_identity_and_lifecycle_tampering_is_rejected_before_planning(
    tmp_path: Path,
    patched_bound: None,
    attribute: str,
    value: object,
) -> None:
    archive = tmp_path / f"bad-{attribute}.zarr"
    root = _archive(archive)
    source = root["analysis/chaser_distance_runs/source"]
    source.attrs[attribute] = value
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(ValueError, match="exact complete eligible sealed canonical"):
        build_chaser_distance_base_candidate_plan(
            archive,
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / f"scratch-{attribute}",
        )


def test_candidate_validation_rejects_different_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, candidate = _published(tmp_path, monkeypatch)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    original = root["analysis/chaser_distance_runs/source"]
    other = root["analysis/chaser_distance_runs"].create_group("other")
    for path in SEALED_CHASER_DISTANCE_BASE_PATHS:
        _write(other, path, np.asarray(original[path][:]))
    other.attrs.update(dict(original.attrs))
    expected_binding = materializer.build_source_authority_binding(
        _bound(), source_group=original
    )
    validation = validate_base_candidate(
        candidate,
        source_group=other,
        expected_source_binding=expected_binding,
    )
    assert not validation["valid"]
    assert any("path differs" in item for item in validation["errors"])


def test_zero_epoch_windows_remain_a_valid_sealed_base_shape(
    tmp_path: Path, patched_bound: None
) -> None:
    archive = tmp_path / "zero_windows_analysis.zarr"
    root = _archive(archive)
    source = root["analysis/chaser_distance_runs/source"]
    source["frames/stimulus_epoch_window_id"][:] = np.int32(-1)
    replacements = {
        "epoch_summary/window_id": np.empty((0,), dtype=np.int32),
        "epoch_summary/label_bytes": np.empty((0, 96), dtype=np.uint8),
        "epoch_summary/start_frame": np.empty((0,), dtype=np.int64),
        "epoch_summary/end_frame": np.empty((0,), dtype=np.int64),
        "epoch_summary/mean_distance_mm": np.empty((0, 2), dtype=np.float32),
        "epoch_summary/min_distance_mm": np.empty((0, 2), dtype=np.float32),
        "epoch_summary/p05_distance_mm": np.empty((0, 2), dtype=np.float32),
        "epoch_summary/p50_distance_mm": np.empty((0, 2), dtype=np.float32),
        "epoch_summary/p95_distance_mm": np.empty((0, 2), dtype=np.float32),
        "epoch_distributions/hist_density": np.empty((0, 2, 3), dtype=np.float32),
    }
    for path, values in replacements.items():
        del source[path]
        _write(source, path, values)
    consolidate_metadata_capture_expected_warnings(archive)

    result = materialize_chaser_distance_base_candidate(
        archive,
        source_run="source",
        run_name="zero_window_candidate",
        scratch_root=tmp_path / "scratch-zero",
        copy_backend="python",
        apply=True,
    )

    assert result["status"] == "complete"
    candidate = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "analysis/chaser_distance_storage_candidates/zero_window_candidate"
    ]
    assert candidate["epoch_summary/window_id"].shape == (0,)
    assert candidate["epoch_distributions/hist_density"].shape == (0, 2, 3)


def test_candidate_hashes_remain_exact_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, candidate = _published(tmp_path, monkeypatch)
    source = zarr.open_group(str(archive), mode="r", use_consolidated=False)[
        "analysis/chaser_distance_runs/source"
    ]
    assert base_logical_hashes(candidate) == base_logical_hashes(source)
