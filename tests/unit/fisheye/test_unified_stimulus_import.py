"""Real public importer -> native candidate reader, no production promotion."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import zarr

from fisheye.analysis.import_stimulus_to_zarr import import_stimulus_to_zarr
from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.storage import load_unified_stimulus_candidate
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture, write_receipt


def _import_fixture(
    tmp_path, name="appearance", *, destination=None, run_name="candidate"
):
    source = emit_fixture(tmp_path, name)
    receipt = write_receipt(tmp_path, name)
    destination = destination or tmp_path / "candidate.zarr"
    import_stimulus_to_zarr(
        source,
        destination,
        run_name=run_name,
        overwrite=False,
        verbose=False,
        source_profile="unified_experimental_h5_v1",
        finalization_receipt=receipt,
    )
    return source, destination


def test_unified_profile_is_explicit_and_refuses_before_destination(tmp_path):
    source = emit_fixture(tmp_path)
    destination = tmp_path / "not-created.zarr"
    with pytest.raises(ValueError):
        import_stimulus_to_zarr(
            source, destination, run_name="candidate", overwrite=False, verbose=False
        )
    assert not destination.exists()


@pytest.mark.parametrize("name", ["base", "appearance"])
def test_public_import_preserves_native_tables_and_never_selects(tmp_path, name):
    source = emit_fixture(tmp_path, name)
    receipt = write_receipt(tmp_path, name)
    destination = tmp_path / "candidate.zarr"
    result = import_stimulus_to_zarr(
        source,
        destination,
        run_name="candidate",
        overwrite=False,
        verbose=False,
        source_profile="unified_experimental_h5_v1",
        finalization_receipt=receipt,
    )
    assert result == "candidate"
    root = zarr.open_group(str(destination), mode="r", use_consolidated=True)
    parent = root["analysis/stimulus_runs"]
    assert resolve_latest_complete_run_name(parent) is None
    assert all(
        key not in parent.attrs
        for key in ("latest", "latest_complete", "authoritative_run")
    )
    assert parent["candidate"].attrs["stage_selector_eligible"] is False
    loaded = load_unified_stimulus_candidate(root, run_name="candidate")
    with h5py.File(source, "r") as h5:
        for path in (
            "/frames/stimulus",
            "/components/chaser/states",
            "/correspondence/chaser/sources",
        ):
            np.testing.assert_array_equal(loaded.read_table(path), h5[path][()])
        if name == "appearance":
            np.testing.assert_array_equal(
                loaded.read_table("/components/visual_appearance/states"),
                h5["/components/visual_appearance/states"][()],
            )
        for path in loaded._manifest["nodes"]:
            if isinstance(h5[path], h5py.Dataset):
                actual = loaded.read_dataset(path)
                expected = h5[path][()]
                np.testing.assert_array_equal(actual, expected)
        assert (
            loaded.read_json("/protocol/executed/execution_index_json")["status"]
            == "interrupted"
        )
        assert (
            loaded.typed_attributes("/metadata/session")["session_uuid"]["type"][
                "class"
            ]
            == "string"
        )
        np.testing.assert_array_equal(
            loaded.read_table("/frames/stimulus", 1, 3), h5["/frames/stimulus"][1:3]
        )


def test_missing_external_receipt_does_not_create_destination(tmp_path):
    source = emit_fixture(tmp_path)
    destination = tmp_path / "absent.zarr"
    with pytest.raises((ValueError, FileNotFoundError)):
        import_stimulus_to_zarr(
            source,
            destination,
            run_name="candidate",
            overwrite=False,
            verbose=False,
            source_profile="unified_experimental_h5_v1",
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "options",
    [
        {"source_profile": "future_v99"},
        {
            "source_profile": "unified_experimental_h5_v1",
            "metadata_and_calibration_only": True,
        },
        {"finalization_receipt": "unused.json"},
    ],
)
def test_profile_misuse_is_rejected_before_destination(tmp_path, options):
    destination = tmp_path / "not-created.zarr"
    with pytest.raises(ValueError):
        import_stimulus_to_zarr(
            emit_fixture(tmp_path),
            destination,
            run_name="candidate",
            overwrite=False,
            verbose=False,
            **options,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "payload",
        "native_attrs",
        "array_attrs",
        "manifest",
        "owner",
        "eligible",
        "provenance",
    ],
)
def test_unpatched_reader_rejects_tampering(tmp_path, mutation):
    _, destination = _import_fixture(tmp_path)
    root = zarr.open_group(str(destination), mode="a", use_consolidated=False)
    run = root["analysis/stimulus_runs/candidate"]
    if mutation == "payload":
        array = run["native_h5/frames/stimulus/payload"]
        array[0] = int(array[0]) ^ 1
    elif mutation == "native_attrs":
        run["native_h5/frames/stimulus"].attrs["unlisted"] = True
    elif mutation == "array_attrs":
        run["native_h5/frames/stimulus/payload"].attrs["logical_schema_version"] = 9
    elif mutation == "manifest":
        array = run["native_manifest_json_utf8"]
        array[0] = ord("[")
    elif mutation == "owner":
        run.attrs["stimulus_publication_owner_uuid"] = "foreign-owner"
    elif mutation == "eligible":
        run.attrs["stage_selector_eligible"] = True
    else:
        provenance = dict(run.attrs["run_provenance"])
        provenance["git_sha"] = "foreign-code"
        run.attrs["run_provenance"] = provenance
    # Intentionally keep the old consolidated generation. Even direct-only
    # metadata mutations must not disappear behind the published view.
    published = zarr.open_group(str(destination), mode="r", use_consolidated=True)
    with pytest.raises(UnifiedH5ContractError):
        load_unified_stimulus_candidate(published, run_name="candidate")


def test_loaded_reader_rechecks_payload_and_generation(tmp_path):
    _, destination = _import_fixture(tmp_path)
    loaded = load_unified_stimulus_candidate(
        zarr.open_group(str(destination), mode="r", use_consolidated=True),
        run_name="candidate",
    )
    mutable = zarr.open_group(str(destination), mode="a", use_consolidated=False)
    mutable["analysis/stimulus_runs/candidate/native_h5/frames/stimulus/payload"][
        0
    ] = 255
    with pytest.raises(UnifiedH5ContractError, match="payload_digest"):
        loaded.read_table("/frames/stimulus")
    mutable["analysis/stimulus_runs/candidate"].attrs[
        "stimulus_publication_owner_uuid"
    ] = "foreign"
    with pytest.raises(UnifiedH5ContractError, match="generation_changed"):
        loaded.read_table("/frames/stimulus")


def test_existing_selectors_and_immutable_runs_are_preserved(tmp_path):
    destination = tmp_path / "existing.zarr"
    root = zarr.open_group(str(destination), mode="a", use_consolidated=False)
    parent = root.require_group("analysis/stimulus_runs")
    parent.create_group("existing", attributes={"sentinel": "preserve"})
    selection = {
        "latest": "existing",
        "latest_complete": "existing",
        "authoritative_run": "existing",
        "authoritative_run_provenance": {"sentinel": "preserve"},
    }
    parent.attrs.update(selection)
    source, _ = _import_fixture(tmp_path, destination=destination)
    assert dict(root["analysis/stimulus_runs"].attrs) == selection
    assert dict(root["analysis/stimulus_runs/existing"].attrs) == {
        "sentinel": "preserve"
    }
    for overwrite in (False, True):
        with pytest.raises(UnifiedH5ContractError, match="new_name"):
            import_stimulus_to_zarr(
                source,
                destination,
                run_name="candidate",
                overwrite=overwrite,
                verbose=False,
                source_profile="unified_experimental_h5_v1",
                finalization_receipt=write_receipt(tmp_path),
            )


@pytest.mark.parametrize(
    "failure",
    [
        "copy",
        "completion",
        "consolidation",
        "consolidation_noop",
        "consolidation_persist_then_raise",
    ],
)
def test_owned_failure_tombstone_and_fresh_retry(tmp_path, monkeypatch, failure):
    from fisheye.analysis import import_stimulus_to_zarr as public_module
    from fisheye.analysis import unified_stimulus_import as native_module
    from fisheye.shared.unified_h5 import storage

    def fail(*args, **kwargs):
        raise RuntimeError("injected " + failure)

    with monkeypatch.context() as patch:
        if failure == "copy":
            patch.setattr(storage, "_write_payload", fail)
        elif failure == "completion":
            patch.setattr(native_module, "mark_run_complete", fail)
        elif failure == "consolidation_persist_then_raise":
            consolidate = public_module.consolidate_metadata_capture_expected_warnings

            def persisted(*args):
                consolidate(*args)
                fail()

            patch.setattr(
                public_module,
                "consolidate_metadata_capture_expected_warnings",
                persisted,
            )
        else:
            patch.setattr(
                public_module,
                "consolidate_metadata_capture_expected_warnings",
                (lambda *args: None) if failure.endswith("noop") else fail,
            )
        with pytest.raises((ValueError, RuntimeError)):
            _import_fixture(tmp_path)
    destination = tmp_path / "candidate.zarr"
    root = zarr.open_group(str(destination), mode="r", use_consolidated=False)
    run = root["analysis/stimulus_runs/candidate"]
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert run.attrs["stage_selector_eligible"] is False
    assert "palette_run_completed_at_utc" not in run.attrs
    assert (
        run.attrs["stimulus_publication_tombstone"]["retry_policy"]
        == "new_immutable_run_name_required"
    )
    assert all(
        key not in root["analysis/stimulus_runs"].attrs
        for key in ("latest", "latest_complete", "authoritative_run")
    )
    with pytest.raises(UnifiedH5ContractError, match="new_name"):
        _import_fixture(tmp_path)
    _import_fixture(tmp_path, run_name="retry")
    published = zarr.open_group(str(destination), mode="r", use_consolidated=True)
    assert (
        load_unified_stimulus_candidate(published, run_name="retry").admission[
            "frame_count"
        ]
        == 4
    )
    with pytest.raises(UnifiedH5ContractError):
        load_unified_stimulus_candidate(published, run_name="candidate")


def test_takeover_stops_copy_and_does_not_tombstone_foreign_run(tmp_path, monkeypatch):
    from fisheye.shared.unified_h5 import storage

    original = storage._write_payload
    destination = tmp_path / "candidate.zarr"
    observed = []

    def take_over(array, parts, assert_owner):
        root = zarr.open_group(str(destination), mode="a", use_consolidated=False)
        run = root["analysis/stimulus_runs/candidate"]
        run.attrs["stimulus_publication_owner_uuid"] = "foreign-owner"
        run.attrs["foreign_sentinel"] = "untouched"
        observed.append(dict(run.attrs))
        return original(array, parts, assert_owner)

    monkeypatch.setattr(storage, "_write_payload", take_over)
    with pytest.raises(UnifiedH5ContractError, match="ownership_lost"):
        _import_fixture(tmp_path)
    root = zarr.open_group(str(destination), mode="r", use_consolidated=False)
    assert dict(root["analysis/stimulus_runs/candidate"].attrs) == observed[0]


def test_source_path_replacement_during_copy_refuses_completion(tmp_path, monkeypatch):
    from fisheye.shared.unified_h5 import storage

    original = storage._write_payload
    changed = False

    def replace_source(array, parts, assert_owner):
        nonlocal changed
        if not changed:
            source = tmp_path / "appearance.h5"
            source.rename(tmp_path / "retained-original.h5")
            emit_fixture(tmp_path, "appearance")
            changed = True
        return original(array, parts, assert_owner)

    monkeypatch.setattr(storage, "_write_payload", replace_source)
    with pytest.raises(UnifiedH5ContractError, match="handle_path_mismatch"):
        _import_fixture(tmp_path)
    root = zarr.open_group(
        str(tmp_path / "candidate.zarr"), mode="r", use_consolidated=False
    )
    assert (
        root["analysis/stimulus_runs/candidate"].attrs["palette_run_completion_status"]
        == "failed"
    )


def test_takeover_during_lifecycle_attr_does_not_write_following_attrs(
    tmp_path, monkeypatch
):
    from fisheye.analysis import import_stimulus_to_zarr as owner

    original = owner._write_stimulus_failure_attr
    destination = tmp_path / "candidate.zarr"
    observed = []

    def write_then_takeover(attrs, name, value):
        original(attrs, name, value)
        if name == "palette_run_completion_status" and value == "complete":
            root = zarr.open_group(str(destination), mode="a", use_consolidated=False)
            run = root["analysis/stimulus_runs/candidate"]
            run.attrs["stimulus_publication_owner_uuid"] = "foreign"
            run.attrs["foreign"] = True
            observed.append(dict(run.attrs))

    monkeypatch.setattr(owner, "_write_stimulus_failure_attr", write_then_takeover)
    with pytest.raises(UnifiedH5ContractError, match="ownership_lost"):
        _import_fixture(tmp_path)
    root = zarr.open_group(str(destination), mode="r", use_consolidated=False)
    assert dict(root["analysis/stimulus_runs/candidate"].attrs) == observed[0]
    assert "palette_run_completed_at_utc" not in observed[0]
