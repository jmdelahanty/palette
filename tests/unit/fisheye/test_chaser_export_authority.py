from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from fisheye.analytics_exports.chaser_authority import (
    EPOCH_BEHAVIOR_FAMILY,
    ChaserExportAuthorityError,
    build_chaser_export_authority_set,
    build_chaser_export_source_authority,
    load_chaser_export_authority_set,
    validate_chaser_export_authority_set,
    validate_chaser_export_source_authority,
    write_chaser_export_authority_set,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _handle(*, digest_byte: str = "b") -> dict[str, object]:
    return {
        "component_family": EPOCH_BEHAVIOR_FAMILY,
        "component_path": (
            "analysis/chaser_distance_runs/base/epoch_behavior_summary/component"
        ),
        "record_sha256": digest_byte * 64,
    }


def _source(tmp_path: Path, *, name: str = "a") -> dict[str, object]:
    return build_chaser_export_source_authority(
        zarr_path=tmp_path / f"{name}_analysis.zarr",
        recording_id=name,
        base_run_name="base",
        base_publication_seal_sha256="a" * 64,
        component_handles={EPOCH_BEHAVIOR_FAMILY: _handle()},
    )


def _recompute_source(record: dict[str, object]) -> None:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = canonical_json_sha256(body)


def _recompute_root(record: dict[str, object]) -> None:
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = canonical_json_sha256(body)


def test_authority_set_round_trips_canonical_file(tmp_path: Path) -> None:
    authority = build_chaser_export_authority_set(
        [_source(tmp_path, name="b"), _source(tmp_path, name="a")]
    )
    destination = write_chaser_export_authority_set(
        tmp_path / "authority.json",
        authority,
    )
    file_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()

    loaded = load_chaser_export_authority_set(
        destination,
        expected_file_sha256=file_sha256,
    )

    assert [source["recording_id"] for source in loaded.record["sources"]] == [
        "a",
        "b",
    ]
    assert set(loaded.sources_by_path) == {
        str((tmp_path / "a_analysis.zarr").resolve()),
        str((tmp_path / "b_analysis.zarr").resolve()),
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda record: record.update({"latest": "fallback"}), "exactly"),
        (
            lambda record: record.update(
                {"base_run_name": "latest", "base_run_path": "latest"}
            ),
            "never latest",
        ),
        (
            lambda record: record["component_handles"].update(
                {"unknown": _handle()}
            ),
            "unknown component",
        ),
    ],
)
def test_source_authority_rejects_rehashed_semantic_tampering(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    record = copy.deepcopy(_source(tmp_path))
    mutation(record)
    _recompute_source(record)

    with pytest.raises(ChaserExportAuthorityError, match=match):
        validate_chaser_export_source_authority(record)


def test_authority_set_rejects_reordered_duplicate_and_rehashed_sources(
    tmp_path: Path,
) -> None:
    authority = build_chaser_export_authority_set(
        [_source(tmp_path, name="a"), _source(tmp_path, name="b")]
    )

    reordered = copy.deepcopy(authority)
    reordered["sources"] = list(reversed(reordered["sources"]))
    _recompute_root(reordered)
    with pytest.raises(ChaserExportAuthorityError, match="sorted"):
        validate_chaser_export_authority_set(reordered)

    duplicate = copy.deepcopy(authority)
    duplicate["sources"][1] = copy.deepcopy(duplicate["sources"][0])
    _recompute_root(duplicate)
    with pytest.raises(ChaserExportAuthorityError, match="unique"):
        validate_chaser_export_authority_set(duplicate)


def test_authority_file_rejects_wrong_digest_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    authority = build_chaser_export_authority_set([_source(tmp_path)])
    destination = write_chaser_export_authority_set(
        tmp_path / "authority.json",
        authority,
    )
    with pytest.raises(ChaserExportAuthorityError, match="file SHA-256"):
        load_chaser_export_authority_set(
            destination,
            expected_file_sha256="0" * 64,
        )

    destination.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ChaserExportAuthorityError, match="non-finite"):
        load_chaser_export_authority_set(destination)
