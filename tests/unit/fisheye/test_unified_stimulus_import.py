"""Real public importer -> sealed H5 reference -> the one reader; never selected."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from fisheye.analysis.import_stimulus_to_zarr import import_stimulus_to_zarr
from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5 import hdf5_types, reference
from fisheye.shared.unified_h5.reference import open_unified_source
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


def _admitted_tables(source):
    with h5py.File(source, "r") as h5:
        return [
            path
            for path, entry in open_unified_source(
                zarr.open_group(str(source.parent / "candidate.zarr"), mode="r", use_consolidated=True),
                run_name="candidate",
            )._record["datasets"].items()
            if entry["mode"] == "blocks" and h5[path].shape[0] > 0
        ]


@pytest.mark.parametrize("name", ["base", "appearance"])
def test_reference_reads_equal_the_h5_and_never_select(tmp_path, name):
    source, destination = _import_fixture(tmp_path, name)
    root = zarr.open_group(str(destination), mode="r", use_consolidated=True)
    parent = root["analysis/stimulus_runs"]
    assert resolve_latest_complete_run_name(parent) is None
    assert all(key not in parent.attrs for key in ("latest", "latest_complete", "authoritative_run"))
    assert parent["candidate"].attrs["stage_selector_eligible"] is False
    assert "native_h5" not in parent["candidate"]  # nothing is copied
    loaded = open_unified_source(root, run_name="candidate")
    tables = _admitted_tables(source)
    assert "/components/chaser/states" in tables
    with h5py.File(source, "r") as h5:
        for path in tables:
            np.testing.assert_array_equal(loaded.read_table(path), h5[path][()])
        np.testing.assert_array_equal(
            loaded.read_table("/frames/stimulus", 1, 3), h5["/frames/stimulus"][1:3]
        )
    assert loaded.read_json("/protocol/executed/execution_index_json")["status"] == "interrupted"
    assert loaded.typed_attributes("/metadata/session")["session_uuid"]["type"]["class"] == "string"
    loaded.verify()


def test_multi_block_ranges_verify_only_touched_blocks(tmp_path, monkeypatch):
    # One row per block, so every range crosses block boundaries.
    monkeypatch.setattr(hdf5_types, "block_rows", lambda dataset, block_bytes=None: 1)
    monkeypatch.setattr(reference, "block_rows", lambda dataset, block_bytes=None: 1)
    source, destination = _import_fixture(tmp_path)
    loaded = open_unified_source(
        zarr.open_group(str(destination), mode="r", use_consolidated=True), run_name="candidate"
    )
    entry = loaded._record["datasets"]["/frames/stimulus"]
    assert entry["rows_per_block"] == 1
    with h5py.File(source, "r") as h5:
        expected = h5["/frames/stimulus"][()]
    for start in range(len(expected)):
        for stop in range(start, len(expected) + 1):
            np.testing.assert_array_equal(loaded.read_table("/frames/stimulus", start, stop), expected[start:stop])
    corrupted = bytearray(loaded._digests)
    corrupted[(entry["first_block"] + 2) * 32] ^= 1
    loaded._digests = bytes(corrupted)
    np.testing.assert_array_equal(loaded.read_table("/frames/stimulus", 0, 2), expected[0:2])
    with pytest.raises(UnifiedH5ContractError, match="block_digest_mismatch"):
        loaded.read_table("/frames/stimulus", 1, 3)


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


def test_expanded_source_path_is_bound_in_writer_provenance(tmp_path):
    from fisheye.analysis.unified_stimulus_import import import_unified_from_open_h5

    source = emit_fixture(tmp_path, "base")
    receipt = write_receipt(tmp_path, "base")
    destination = tmp_path / "expanded-source.zarr"
    # Exercise the native preflight's path normalization without changing HOME
    # or writing outside pytest's temporary directory. The legacy public path
    # existence check still rejects unexpanded tilde arguments.
    argument = Path("~") / os.path.relpath(source, Path.home())
    assert argument.expanduser().resolve() == source.resolve()
    with h5py.File(source, "r") as h5:
        import_unified_from_open_h5(
            h5,
            source_h5=argument,
            zarr_path=destination,
            run_name="candidate",
            overwrite=False,
            finalization_receipt=receipt,
        )
    root = zarr.open_group(str(destination), mode="r", use_consolidated=True)
    candidate = open_unified_source(root, run_name="candidate")
    artifacts = root["analysis/stimulus_runs/candidate"].attrs["run_provenance"][
        "input_artifacts"
    ]
    assert artifacts == [
        {
            "path": str(source.resolve()),
            "sha256": candidate.admission["source_sha256"],
            "size_bytes": source.stat().st_size,
        }
    ]


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


def _flip_byte_keep_mtime(source, path):
    with h5py.File(source, "r") as h5:
        dataset = h5[path]
        offset = (
            dataset.id.get_offset()
            if dataset.chunks is None
            else dataset.id.get_chunk_info(0).byte_offset
        )
    stat = source.stat()
    with open(source, "r+b") as handle:
        handle.seek(offset)
        value = handle.read(1)
        handle.seek(offset)
        handle.write(bytes([value[0] ^ 1]))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_changed_block_refuses_only_reads_that_touch_it(tmp_path):
    source, destination = _import_fixture(tmp_path)
    _flip_byte_keep_mtime(source, "/components/chaser/states")
    loaded = open_unified_source(
        zarr.open_group(str(destination), mode="r", use_consolidated=True), run_name="candidate"
    )
    loaded.read_table("/frames/stimulus")
    with pytest.raises(UnifiedH5ContractError, match="block_digest_mismatch"):
        loaded.read_table("/components/chaser/states")
    with pytest.raises(UnifiedH5ContractError, match="block_digest_mismatch"):
        loaded.verify()


@pytest.mark.parametrize("change", ["replaced", "missing", "reference", "eligible"])
def test_open_refuses_a_changed_source_or_reference(tmp_path, change):
    source, destination = _import_fixture(tmp_path)
    run = zarr.open_group(str(destination), mode="a", use_consolidated=False)[
        "analysis/stimulus_runs/candidate"
    ]
    if change == "replaced":
        source.write_bytes(source.read_bytes() + b"\0")
    elif change == "missing":
        source.unlink()
    elif change == "reference":
        run[reference.REFERENCE_ARRAY][0] = ord("[")
    else:
        run.attrs["stage_selector_eligible"] = True  # direct only; consolidated is stale
    expected = {
        "replaced": "unified_source_changed",
        "missing": "unified_source_missing",
        "reference": "unified_reference_digest_mismatch",
        "eligible": "unified_reference_metadata_split",
    }[change]
    with pytest.raises(UnifiedH5ContractError, match=expected):
        open_unified_source(
            zarr.open_group(str(destination), mode="r", use_consolidated=True), run_name="candidate"
        )


def test_moving_the_whole_recording_keeps_the_reference_valid(tmp_path):
    recording = tmp_path / "rec"
    source = emit_fixture(recording / "raw", "base")
    receipt = write_receipt(recording / "raw", "base")
    import_stimulus_to_zarr(
        source, recording / "zarr" / "rec_analysis.zarr", run_name="candidate",
        overwrite=False, verbose=False, source_profile="unified_experimental_h5_v1",
        finalization_receipt=receipt,
    )
    moved = recording.rename(tmp_path / "moved")
    loaded = open_unified_source(
        zarr.open_group(str(moved / "zarr" / "rec_analysis.zarr"), mode="r", use_consolidated=True),
        run_name="candidate",
    )
    assert loaded.source_path == (moved / "raw" / "base.h5").resolve()
    loaded.verify()


def test_source_outside_the_recording_is_refused(tmp_path):
    source = emit_fixture(tmp_path / "elsewhere" / "deep", "base")
    destination = tmp_path / "rec" / "zarr" / "rec_analysis.zarr"
    with pytest.raises(UnifiedH5ContractError, match="outside_recording"):
        import_stimulus_to_zarr(
            source, destination, run_name="candidate", overwrite=False, verbose=False,
            source_profile="unified_experimental_h5_v1",
            finalization_receipt=write_receipt(tmp_path / "elsewhere" / "deep", "base"),
        )


def test_admission_scan_hashes_each_table_once(tmp_path, monkeypatch):
    from fisheye.shared.unified_h5 import schema, validate_unified_h5_artifact
    from fisheye.shared.unified_h5.common import admission_scan
    from tests.unit.fisheye.unified_h5_fixtures import receipt_for

    source = emit_fixture(tmp_path, "appearance")
    original, reads = schema.iter_blocks, []

    def counting(dataset, **kwargs):
        reads.append(dataset.name)
        return original(dataset, **kwargs)

    monkeypatch.setattr(schema, "iter_blocks", counting)
    with h5py.File(source, "r") as h5:
        validate_unified_h5_artifact(h5, source_h5=source, finalization_receipt=receipt_for("appearance"))
        unscanned = reads.count("/components/visual_appearance/states")
        reads.clear()
        with admission_scan():
            validate_unified_h5_artifact(h5, source_h5=source, finalization_receipt=receipt_for("appearance"))
        scanned = reads.count("/components/visual_appearance/states")
    assert unscanned > 1
    assert scanned == 1


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
        "seal",
        "completion",
        "consolidation",
        "consolidation_noop",
        "consolidation_persist_then_raise",
    ],
)
def test_owned_failure_tombstone_and_fresh_retry(tmp_path, monkeypatch, failure):
    from fisheye.analysis import import_stimulus_to_zarr as public_module
    from fisheye.analysis import unified_stimulus_import as native_module
    def fail(*args, **kwargs):
        raise RuntimeError("injected " + failure)

    with monkeypatch.context() as patch:
        if failure == "seal":
            patch.setattr(reference, "_write_bytes", fail)
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
        open_unified_source(published, run_name="retry").admission[
            "frame_count"
        ]
        == 4
    )
    with pytest.raises(UnifiedH5ContractError):
        open_unified_source(published, run_name="candidate")


def test_takeover_stops_sealing_and_does_not_tombstone_foreign_run(tmp_path, monkeypatch):
    original = reference._write_bytes
    destination = tmp_path / "candidate.zarr"
    observed = []

    def take_over(group, name, data):
        root = zarr.open_group(str(destination), mode="a", use_consolidated=False)
        run = root["analysis/stimulus_runs/candidate"]
        run.attrs["stimulus_publication_owner_uuid"] = "foreign-owner"
        run.attrs["foreign_sentinel"] = "untouched"
        observed.append(dict(run.attrs))
        return original(group, name, data)

    monkeypatch.setattr(reference, "_write_bytes", take_over)
    with pytest.raises(UnifiedH5ContractError, match="ownership_lost"):
        _import_fixture(tmp_path)
    root = zarr.open_group(str(destination), mode="r", use_consolidated=False)
    assert dict(root["analysis/stimulus_runs/candidate"].attrs) == observed[-1]


def test_source_path_replacement_during_sealing_refuses_completion(tmp_path, monkeypatch):
    original = reference._write_bytes
    changed = False

    def replace_source(group, name, data):
        nonlocal changed
        if not changed:
            source = tmp_path / "appearance.h5"
            source.rename(tmp_path / "retained-original.h5")
            emit_fixture(tmp_path, "appearance")
            changed = True
        return original(group, name, data)

    monkeypatch.setattr(reference, "_write_bytes", replace_source)
    with pytest.raises(UnifiedH5ContractError, match="handle_path_mismatch"):
        _import_fixture(tmp_path)
    root = zarr.open_group(str(tmp_path / "candidate.zarr"), mode="r", use_consolidated=False)
    assert root["analysis/stimulus_runs/candidate"].attrs["palette_run_completion_status"] == "failed"


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
