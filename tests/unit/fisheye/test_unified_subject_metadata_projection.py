"""Unified subject metadata lands where legacy import put it.

Subject metadata and experiment setup are projected from the admitted native
candidate's ``/metadata/subject`` through the same owners as legacy import.
A unified H5 and a legacy H5 carrying the same subject attributes must publish
the same subject record and the same expected subject count.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from fisheye.analysis.import_stimulus_to_zarr import import_stimulus_to_zarr
from fisheye.shared.experiment_setup import resolve_experiment_setup
from fisheye.shared.subject_metadata import resolve_subject_metadata
from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.metadata import string_attributes
from fisheye.shared.unified_h5.storage import MANIFEST_DIGEST_ATTR
from fisheye.utils import import_recording_analysis as mod
from tests.unit.fisheye.test_import_recording_analysis import (
    _acquisition_authority_updates,
    _write_current_manifest,
)
from tests.unit.fisheye.unified_h5_fixtures import (
    emit_fixture,
    synthetic_receipt_for_mutated_test_file,
    write_receipt,
)

# The production GUI's subject keys (Citrus, 2026-09-23), as string attrs, with
# values shaped like a real legacy goodbatbadbat /subject_metadata.
PRODUCTION_SUBJECT = {
    "fish_id": "4bd7a2f6-92b7-42b8-ba28-6e49cb3cc0c9",
    "subject_type": "individual",
    "subject_count": "1",
    "dish_id": "18867_10",
    "cross_id": "18867",
    "source_dish_id": "18867_10",
    "source_cross_id": "18867",
    "fish_count": "10",
    "source_dish_population_count": "10",
    "genotype": "AB [AB IC] SEPT25",
    "line_strain": "AB [AB IC] SEPT25",
    "species": "Danio rerio",
    "sex": "unknown",
    "parents": "8750_M08_B2 [unknown]",
    "date_of_fertilization": "20260804",
    "days_post_fertilization": "8",
    "queried_at_utc": "2026-08-12T21:59:55Z",
}


def _native_candidate(tmp_path: Path, subject: dict[str, str] | None) -> mod.RecordingAnalysisPlan:
    """Import a fixture (optionally with replaced subject attrs) as a native candidate."""

    h5_path = emit_fixture(tmp_path / "source", "base")
    receipt = write_receipt(tmp_path / "source", "base")
    if subject is not None:
        with h5py.File(h5_path, "r+") as h5:
            attrs = h5["/metadata/subject"].attrs
            for key in list(attrs):
                del attrs[key]
            for key, value in subject.items():
                attrs[key] = value
        receipt = tmp_path / "source" / "resealed.receipt.json"
        receipt.write_text(json.dumps(synthetic_receipt_for_mutated_test_file(h5_path, "base")))
    zarr_path = tmp_path / "analysis.zarr"
    import_stimulus_to_zarr(
        h5_path, zarr_path, run_name="candidate", overwrite=False, verbose=False,
        source_profile="unified_experimental_h5_v1", finalization_receipt=receipt,
    )
    return mod.RecordingAnalysisPlan(
        recording_dir=tmp_path, h5_path=h5_path, cam_video=None, zarr_path=zarr_path,
        finalization_receipt_path=receipt,
    )


def _legacy_setup(tmp_path: Path, subject: dict[str, str]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    h5_path = tmp_path / "legacy.h5"
    with h5py.File(h5_path, "w") as h5:
        node = h5.create_group("subject_metadata")
        for key, value in subject.items():
            node.attrs[key] = value
    zarr_path = tmp_path / "legacy.zarr"
    zarr.open_group(str(zarr_path), mode="w")
    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path, h5_path=h5_path, cam_video=None, zarr_path=zarr_path
    )
    mod.import_experiment_setup(plan)
    return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)


def test_unified_and_legacy_publish_the_same_subject_and_setup(tmp_path: Path) -> None:
    plan = _native_candidate(tmp_path / "unified", PRODUCTION_SUBJECT)

    published = mod.project_unified_subject_metadata(plan, "candidate")

    unified_root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=False)
    legacy_root = _legacy_setup(tmp_path / "legacy", PRODUCTION_SUBJECT)
    unified_subject = resolve_subject_metadata(unified_root, allow_legacy=False)
    legacy_subject = resolve_subject_metadata(legacy_root, allow_legacy=False)
    assert unified_subject.metadata == legacy_subject.metadata
    assert unified_subject.subject_ids == legacy_subject.subject_ids == (PRODUCTION_SUBJECT["fish_id"],)
    assert unified_subject.record_sha256 == legacy_subject.record_sha256

    unified_setup = resolve_experiment_setup(unified_root, allow_legacy=False)
    legacy_setup = resolve_experiment_setup(legacy_root, allow_legacy=False)
    assert unified_setup.expected_subject_count == legacy_setup.expected_subject_count == 1
    assert unified_setup.subject_assignment_status == legacy_setup.subject_assignment_status
    assert published["expected_subject_count"] == 1


def test_setup_records_its_verified_native_source(tmp_path: Path) -> None:
    plan = _native_candidate(tmp_path, PRODUCTION_SUBJECT)

    mod.project_unified_subject_metadata(plan, "candidate")

    root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=False)
    source = dict(resolve_experiment_setup(root, allow_legacy=False).source)
    assert source["kind"] == "unified_native_subject_metadata"
    assert source["group_path"] == "/metadata/subject"
    assert source["native_run_path"] == "analysis/stimulus_runs/candidate"
    assert source["native_manifest_sha256"] == root[
        "analysis/stimulus_runs/candidate"
    ].attrs[MANIFEST_DIGEST_ATTR]


def test_missing_subject_count_refuses_rather_than_inventing(tmp_path: Path) -> None:
    # The pinned fixture is sparse: only subject_id, no subject_count.
    plan = _native_candidate(tmp_path, None)

    with pytest.raises(ValueError, match="subject_count"):
        mod.project_unified_subject_metadata(plan, "candidate")


def test_subject_id_is_not_treated_as_fish_id(tmp_path: Path) -> None:
    plan = _native_candidate(tmp_path, {"subject_id": "synthetic-subject", "subject_count": "1"})

    mod.project_unified_subject_metadata(plan, "candidate")

    root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=False)
    assert resolve_subject_metadata(root, allow_legacy=False).subject_ids == ()
    assert resolve_experiment_setup(root, allow_legacy=False).subject_assignment_status == "count_only"


def test_string_attributes_refuse_non_string_values() -> None:
    assert string_attributes(
        {"a": {"shape": [], "type": {"class": "string", "variable_length": True, "character_set": "utf8"},
               "values_hex": ["6869"]}}
    ) == {"a": "hi"}
    assert string_attributes(
        {"b": {"shape": [], "type": {"class": "string", "variable_length": False, "character_set": "ascii",
                                     "size_bytes": 4, "padding": "null_padded"},
               "payload_hex": "68690000"}}
    ) == {"b": "hi"}
    with pytest.raises(UnifiedH5ContractError, match="attribute_not_scalar_string"):
        string_attributes(
            {"n": {"shape": [], "type": {"class": "integer"}, "payload_hex": np.int64(1).tobytes().hex()}}
        )
    with pytest.raises(UnifiedH5ContractError, match="attribute_not_scalar_string"):
        string_attributes(
            {"v": {"shape": [2], "type": {"class": "string", "variable_length": True, "character_set": "utf8"},
                   "values_hex": ["61", "62"]}}
        )


def test_process_import_projects_after_the_native_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording = tmp_path / "rec"
    (recording / "raw").mkdir(parents=True)
    h5_path = emit_fixture(recording / "raw", "base")
    plan = mod.RecordingAnalysisPlan(
        recording_dir=recording, h5_path=h5_path, cam_video=recording / "cams" / "cam.mp4",
        zarr_path=recording / "zarr" / "rec_analysis.zarr",
        finalization_receipt_path=write_receipt(recording / "raw", "base"),
    )
    opts = mod.RecordingImportOptions(
        import_video_metadata=True, video_metadata_overwrite=False, import_stimulus=True,
        stimulus_always=False, stimulus_run_name=None, stimulus_overwrite=False, stimulus_quiet=True,
    )
    _write_current_manifest(plan.recording_dir)
    monkeypatch.setattr(mod, "git_identity", lambda **_kwargs: {"git_sha": "1" * 40, "git_dirty": False})
    monkeypatch.setattr(mod, "apply_video_metadata", lambda _plan, **_kwargs: _acquisition_authority_updates())
    monkeypatch.setattr(mod, "ensure_analysis_archive", lambda _plan: None)
    monkeypatch.setattr(mod, "apply_acquisition_frame_clock", lambda _plan: {})
    monkeypatch.setattr(mod, "stimulus_runs_present", lambda _path: False)
    monkeypatch.setattr(
        mod, "import_experiment_setup",
        lambda _plan: pytest.fail("legacy subject reader must not run on a unified H5"),
    )
    order: list[tuple[str, str]] = []
    monkeypatch.setattr(
        mod, "run_stimulus_import",
        lambda _plan, stim_opts: order.append(("import", stim_opts.stimulus_run_name)) or (True, 0, ["cmd", "x"]),
    )
    monkeypatch.setattr(
        mod, "project_unified_subject_metadata",
        lambda _plan, run_name: order.append(("project", run_name)) or None,
    )

    mod.process_recording_import(plan, opts, logger=None)

    assert [step for step, _ in order] == ["import", "project"]
    assert order[0][1] == order[1][1]
    assert order[0][1].startswith("unified_native_")
