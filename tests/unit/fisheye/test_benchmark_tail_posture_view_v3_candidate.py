from __future__ import annotations

import copy
from pathlib import Path
import shutil

import numpy as np
import pytest
import zarr

from fisheye.analysis import tail_posture_view_runs as tail_writer
from fisheye.analysis.direct_writer_storage import (
    ANALYSIS_STORAGE_PLAN_DIGEST_ATTR,
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    ANALYSIS_STORAGE_PROFILE_ID_ATTR,
)
from fisheye.diagnostics import benchmark_tail_posture_view_v3_candidate as benchmark
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from tests.unit.fisheye.test_subject_shape_coordinate_publication import (  # noqa: F401
    canonical_refined_template,
    canonical_subject_shape_profile_template,
)


SOURCE_RUN = "tail_posture_source"
CANDIDATE_RUN = "tail_posture_candidate"


def _patch_tail_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tail_writer,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "d" * 40,
            "short_hash": "dddddddd",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    original_read_sources = tail_writer._read_sources

    def read_sources_with_one_invalid_row(*args, **kwargs):
        sources, row_count = original_read_sources(*args, **kwargs)
        if row_count < 2:
            raise AssertionError("Tail-posture semantic fixture needs at least two rows.")
        patched = {name: np.asarray(values).copy() for name, values in sources.items()}
        patched["tail_sample_valid"][1] = False
        patched["tail_sample_failure_reason"][1] = "fixture_tail_geometry_invalid"
        return patched, row_count

    monkeypatch.setattr(tail_writer, "_read_sources", read_sources_with_one_invalid_row)
    monkeypatch.setattr(
        tail_writer,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "tail-posture-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )


@pytest.fixture(scope="module")
def tail_posture_pair_archive(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    shape_template = request.getfixturevalue(
        "canonical_subject_shape_profile_template"
    )
    archive = tmp_path_factory.mktemp("tail-posture-read-matrix") / "tail-posture-pair.zarr"
    shutil.copytree(shape_template, archive)
    root = zarr.open_group(archive, mode="r+", use_consolidated=False)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_tail_provenance(monkeypatch)
        source = tail_writer.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_profile_attack",
            run_name=SOURCE_RUN,
            stage_command="tail-posture-source-fixture",
        )
        candidate = tail_writer.write_tail_posture_view_run_group(
            root,
            subject_shape_run="shape_profile_attack",
            run_name=CANDIDATE_RUN,
            stage_command="tail-posture-candidate-fixture",
            storage_profile=PUBLISHED_HTTP_V1,
        )
    assert source["status"] == "updated"
    assert candidate["status"] == "candidate_complete"
    zarr.consolidate_metadata(archive)
    return archive


def _resign(envelope: dict[str, object]) -> None:
    envelope["payload_digest"] = canonical_json_sha256(envelope["payload"])


def test_pair_and_balanced_fresh_process_matrix_are_exact_and_nonpromoting(
    tail_posture_pair_archive: Path,
    tmp_path: Path,
) -> None:
    pair = benchmark.validate_pair(
        tail_posture_pair_archive,
        source_run=SOURCE_RUN,
        candidate_run=CANDIDATE_RUN,
    )
    pair_payload = pair["payload"]
    assert pair_payload["logical_equality"]["all_equal"] is True
    assert pair_payload["logical_equality"]["array_count"] == 10
    assert pair_payload["source"]["selector_eligible"] is True
    assert pair_payload["candidate"]["selector_eligible"] is False
    assert pair_payload["source"]["semantic_validation"]["valid_rows"] == 1
    assert pair_payload["source"]["semantic_validation"]["invalid_rows"] == 1
    assert pair_payload["source"]["semantic_validation"] == pair_payload["candidate"][
        "semantic_validation"
    ]
    assert pair_payload["source"]["scientific_identity"] == pair_payload["candidate"][
        "scientific_identity"
    ]
    assert pair_payload["source"]["storage_plan_sha256"] is None
    assert len(pair_payload["candidate"]["storage_plan_sha256"]) == 64
    assert pair_payload["source"]["adapter"] == benchmark.SOURCE_ADAPTER
    assert pair_payload["candidate"]["adapter"] == benchmark.CANDIDATE_ADAPTER
    assert pair_payload["consumer_boundary"] == benchmark._consumer_boundary()
    assert pair_payload["consumer_boundary"]["standalone_public_payload_reader"] is False
    assert (
        pair_payload["consumer_boundary"]["full_megabouts_input_pack_consumer"]
        == benchmark.BROADER_CONSUMER_STATUS
    )
    assert pair_payload["promotion_authorized"] is False
    assert pair_payload["metadata_equivalence"]["array_count"] == 20

    output = tmp_path / "tail_posture_benchmark_matrix"
    matrix = benchmark.run_benchmark_matrix(
        tail_posture_pair_archive,
        source_run=SOURCE_RUN,
        candidate_run=CANDIDATE_RUN,
        output_root=output,
        repetitions=2,
        seed=7,
        window_rows=1,
        windows_per_array=2,
    )
    payload = matrix["payload"]
    assert payload["fresh_process_count"] == 4
    assert [trial["payload"]["role"] for trial in payload["trials"]] == [
        "source",
        "candidate",
        "candidate",
        "source",
    ]
    assert payload["workload"]["payload"]["operation_count"] == 30
    assert payload["physical_io"] == benchmark._physical_io_null()
    assert payload["promotion_authorized"] is False
    assert payload["selectors_unchanged"] is True
    assert (output / "pair_validation.json").is_file()
    assert (output / "workload.json").is_file()
    assert (output / "matrix.json").is_file()


def test_resigned_evidence_attacks_fail_live_replay(
    tail_posture_pair_archive: Path,
    tmp_path: Path,
) -> None:
    matrix = benchmark.run_benchmark_matrix(
        tail_posture_pair_archive,
        source_run=SOURCE_RUN,
        candidate_run=CANDIDATE_RUN,
        output_root=tmp_path / "tail_posture_benchmark_attacks",
        repetitions=1,
        seed=11,
        window_rows=1,
        windows_per_array=1,
    )
    clean_trial = matrix["payload"]["trials"][0]

    attacks: list[dict[str, object]] = []

    storage = copy.deepcopy(clean_trial)
    storage_pair = storage["payload"]["pair_validation"]
    storage_pair["payload"]["candidate"]["storage_plan_sha256"] = "0" * 64
    _resign(storage_pair)
    _resign(storage)
    attacks.append(storage)

    coordinate = copy.deepcopy(clean_trial)
    coordinate_pair = coordinate["payload"]["pair_validation"]
    coordinate_pair["payload"]["candidate"]["coordinate_manifest_sha256"] = "1" * 64
    _resign(coordinate_pair)
    _resign(coordinate)
    attacks.append(coordinate)

    scientific = copy.deepcopy(clean_trial)
    scientific_pair = scientific["payload"]["pair_validation"]
    for role in ("source", "candidate"):
        scientific_pair["payload"][role]["scientific_identity"]["head_source"] = (
            "invented_head_source"
        )
    _resign(scientific_pair)
    _resign(scientific)
    attacks.append(scientific)

    semantics = copy.deepcopy(clean_trial)
    semantics_pair = semantics["payload"]["pair_validation"]
    for role in ("source", "candidate"):
        semantics_pair["payload"][role]["semantic_validation"]["valid_rows"] = 2
        semantics_pair["payload"][role]["semantic_validation"]["invalid_rows"] = 0
    _resign(semantics_pair)
    _resign(semantics)
    attacks.append(semantics)

    selector = copy.deepcopy(clean_trial)
    selector["payload"]["selectors_before"]["latest"]["value"] = CANDIDATE_RUN
    _resign(selector)
    attacks.append(selector)

    workload = copy.deepcopy(clean_trial)
    workload_envelope = workload["payload"]["workload"]
    workload_envelope["payload"]["seed"] += 1
    _resign(workload_envelope)
    _resign(workload)
    attacks.append(workload)

    payload_digest = copy.deepcopy(clean_trial)
    first_receipt = payload_digest["payload"]["result"]["read_receipts"][0]
    first_receipt["bits_sha256"] = "2" * 64
    _resign(payload_digest)
    attacks.append(payload_digest)

    physical = copy.deepcopy(clean_trial)
    physical["payload"]["physical_io"]["available"] = True
    physical["payload"]["physical_io"]["bytes_transferred"] = 1
    _resign(physical)
    attacks.append(physical)

    consumer = copy.deepcopy(clean_trial)
    consumer["payload"]["consumer_boundary"]["standalone_public_payload_reader"] = True
    _resign(consumer)
    attacks.append(consumer)

    promotion = copy.deepcopy(clean_trial)
    promotion["payload"]["promotion_authorized"] = True
    _resign(promotion)
    attacks.append(promotion)

    for attack in attacks:
        with pytest.raises(ValueError):
            benchmark.validate_trial_evidence(
                attack,
                archive=tail_posture_pair_archive,
                source_run=SOURCE_RUN,
                candidate_run=CANDIDATE_RUN,
            )

    matrix_attacks: list[dict[str, object]] = []

    boolean_repetitions = copy.deepcopy(matrix)
    boolean_repetitions["payload"]["repetitions"] = True
    _resign(boolean_repetitions)
    matrix_attacks.append(boolean_repetitions)

    malformed_completion = copy.deepcopy(matrix)
    malformed_completion["payload"]["completed_at_utc"] = "not-a-timestamp"
    _resign(malformed_completion)
    matrix_attacks.append(malformed_completion)

    non_utc_completion = copy.deepcopy(matrix)
    non_utc_completion["payload"]["completed_at_utc"] = (
        "2026-08-03T12:00:00-04:00"
    )
    _resign(non_utc_completion)
    matrix_attacks.append(non_utc_completion)

    for attack in matrix_attacks:
        with pytest.raises(ValueError):
            benchmark.validate_matrix_evidence(
                attack,
                archive=tail_posture_pair_archive,
                source_run=SOURCE_RUN,
                candidate_run=CANDIDATE_RUN,
            )


@pytest.mark.parametrize(
    "attack",
    (
        "duplicate_instance_key",
        "negative_crop_row",
        "negative_frame",
        "reason_missing_nul",
        "reason_nonzero_padding",
        "valid_reason_not_ok",
        "invalid_reason_ok",
        "valid_float_nan",
        "invalid_float_finite",
        "angle_degree_mismatch",
    ),
)
def test_row_semantic_attacks_fail_closed(
    tail_posture_pair_archive: Path,
    attack: str,
) -> None:
    source_root = zarr.open_group(
        tail_posture_pair_archive,
        mode="r",
        use_consolidated=True,
    )
    source = source_root[f"{benchmark.PARENT_PATH}/{SOURCE_RUN}"]
    scratch = zarr.group()
    scratch.attrs.update(dict(source.attrs))
    for name in source.array_keys():
        scratch.create_array(name, data=np.asarray(source[name][:]))
    dimensions = benchmark._dimensions(scratch)
    assert benchmark._semantic_validation(scratch, dimensions)["invalid_rows"] == 1

    if attack == "duplicate_instance_key":
        values = np.asarray(scratch["instance_key"][:])
        values[1] = values[0]
        scratch["instance_key"][:] = values
    elif attack == "negative_crop_row":
        values = np.asarray(scratch["source_crop_row_ids"][:])
        values[0] = -1
        scratch["source_crop_row_ids"][:] = values
    elif attack == "negative_frame":
        values = np.asarray(scratch["source_acquisition_frame_index"][:])
        values[0] = -1
        scratch["source_acquisition_frame_index"][:] = values
    elif attack == "reason_missing_nul":
        values = np.asarray(scratch["failure_reason_bytes"][:])
        values[0, :] = np.uint8(ord("x"))
        scratch["failure_reason_bytes"][:] = values
    elif attack == "reason_nonzero_padding":
        values = np.asarray(scratch["failure_reason_bytes"][:])
        values[0, -1] = 1
        scratch["failure_reason_bytes"][:] = values
    elif attack == "valid_reason_not_ok":
        values = np.asarray(scratch["failure_reason_bytes"][:])
        values[0] = tail_writer._encode_reasons(["wrong"])[0]
        scratch["failure_reason_bytes"][:] = values
    elif attack == "invalid_reason_ok":
        values = np.asarray(scratch["failure_reason_bytes"][:])
        values[1] = tail_writer._encode_reasons(["ok"])[0]
        scratch["failure_reason_bytes"][:] = values
    elif attack == "valid_float_nan":
        values = np.asarray(scratch["head_xy"][:])
        values[0, 0] = np.nan
        scratch["head_xy"][:] = values
    elif attack == "invalid_float_finite":
        values = np.asarray(scratch["head_xy"][:])
        values[1, 0] = 0.0
        scratch["head_xy"][:] = values
    elif attack == "angle_degree_mismatch":
        values = np.asarray(scratch["tail_angle_deg"][:])
        values[0, 0] += np.float32(1.0)
        scratch["tail_angle_deg"][:] = values
    else:  # pragma: no cover
        raise AssertionError(attack)

    with pytest.raises(ValueError):
        benchmark._semantic_validation(scratch, dimensions)


def test_live_forged_selector_and_rehashed_storage_plan_fail_closed(
    tail_posture_pair_archive: Path,
    tmp_path: Path,
) -> None:
    selector_archive = tmp_path / "selector-attack.zarr"
    shutil.copytree(tail_posture_pair_archive, selector_archive)
    selector_root = zarr.open_group(selector_archive, mode="r+", use_consolidated=False)
    selector_root[benchmark.PARENT_PATH].attrs["latest"] = CANDIDATE_RUN
    zarr.consolidate_metadata(selector_archive)
    with pytest.raises(ValueError, match="all select the frozen source"):
        benchmark.validate_pair(
            selector_archive,
            source_run=SOURCE_RUN,
            candidate_run=CANDIDATE_RUN,
        )

    storage_archive = tmp_path / "storage-attack.zarr"
    shutil.copytree(tail_posture_pair_archive, storage_archive)
    storage_root = zarr.open_group(storage_archive, mode="r+", use_consolidated=False)
    candidate = storage_root[f"{benchmark.PARENT_PATH}/{CANDIDATE_RUN}"]
    receipt = copy.deepcopy(candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR])
    receipt["payload"]["arrays"][0]["plan"]["chunk_shape"][0] += 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt
    candidate.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = receipt["payload_digest"]
    zarr.consolidate_metadata(storage_archive)
    with pytest.raises(ValueError, match="exact schema failed"):
        benchmark.validate_pair(
            storage_archive,
            source_run=SOURCE_RUN,
            candidate_run=CANDIDATE_RUN,
        )

    profile_attr_archive = tmp_path / "profile-attr-attack.zarr"
    shutil.copytree(tail_posture_pair_archive, profile_attr_archive)
    profile_attr_root = zarr.open_group(
        profile_attr_archive,
        mode="r+",
        use_consolidated=False,
    )
    profile_attr_root[
        f"{benchmark.PARENT_PATH}/{CANDIDATE_RUN}"
    ].attrs[ANALYSIS_STORAGE_PROFILE_ID_ATTR] = "invented_profile"
    zarr.consolidate_metadata(profile_attr_archive)
    with pytest.raises(ValueError, match="exact schema failed"):
        benchmark.validate_pair(
            profile_attr_archive,
            source_run=SOURCE_RUN,
            candidate_run=CANDIDATE_RUN,
        )

    profile_receipt_archive = tmp_path / "profile-receipt-attack.zarr"
    shutil.copytree(tail_posture_pair_archive, profile_receipt_archive)
    profile_receipt_root = zarr.open_group(
        profile_receipt_archive,
        mode="r+",
        use_consolidated=False,
    )
    candidate = profile_receipt_root[f"{benchmark.PARENT_PATH}/{CANDIDATE_RUN}"]
    receipt = copy.deepcopy(candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR])
    receipt["payload"]["storage_profile"]["profile_id"] = "invented_profile"
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt
    candidate.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = receipt["payload_digest"]
    zarr.consolidate_metadata(profile_receipt_archive)
    with pytest.raises(ValueError, match="exact schema failed"):
        benchmark.validate_pair(
            profile_receipt_archive,
            source_run=SOURCE_RUN,
            candidate_run=CANDIDATE_RUN,
        )


def test_alias_symlink_and_unsafe_output_paths_are_rejected(
    tail_posture_pair_archive: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="explicit immutable child"):
        benchmark.validate_pair(
            tail_posture_pair_archive,
            source_run="latest",
            candidate_run=CANDIDATE_RUN,
        )
    alias = tmp_path / "tail-posture-alias.zarr"
    alias.symlink_to(tail_posture_pair_archive, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        benchmark.validate_pair(
            alias,
            source_run=SOURCE_RUN,
            candidate_run=CANDIDATE_RUN,
        )
    with pytest.raises(ValueError, match="disjoint"):
        benchmark.run_benchmark_matrix(
            tail_posture_pair_archive,
            source_run=SOURCE_RUN,
            candidate_run=CANDIDATE_RUN,
            output_root=tail_posture_pair_archive / "benchmark-output",
            repetitions=1,
        )
