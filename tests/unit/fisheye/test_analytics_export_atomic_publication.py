from __future__ import annotations

import json
import os
import concurrent.futures
import hashlib
from pathlib import Path
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.arrow_contracts import arrow_contract_envelope
from fisheye.analytics_exports.contracts import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    RECORDING_SUMMARY_TABLE,
    STATISTICS_TABLE,
    TABLE_CONTRACTS,
    canonicalize_export_row,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import (
    load_export_manifest,
    manifest_selected_part_files,
    sha256_file,
)
from fisheye.analytics_exports.validation import ExportValidationError, validate_export_run
from fisheye.group_analytics_viewer.query import (
    ViewerContext,
    build_health_report,
    parquet_files,
    query_export_summary,
    resolve_statistics_run_id,
)
from fisheye.group_statistics.goodcopbadcop import write_goodcopbadcop_statistics
from fisheye.registry.db import Registry
from fisheye.utils.analytics_export_resolution import resolve_latest_export_table
from fisheye.utils.export_goodcopbadcop_static_montage import (
    _resolve_export_run_id as resolve_montage_export_run_id,
)
from fisheye.utils.export_cross_recording_analytics import (
    SourceExportResult,
    _latest_run,
    export_sources,
)
from fisheye.utils.index_analytics_manifests import index_export_manifest
from fisheye.utils.plot_cross_recording_bout_kinematics import resolve_export_run_id


def _stub_source(path: Path, **_kwargs: object) -> SourceExportResult:
    recording_id = path.stem
    return SourceExportResult(
        zarr_path=str(path),
        recording_id=recording_id,
        rows_by_table={
            RECORDING_SUMMARY_TABLE: [
                canonicalize_export_row(
                    RECORDING_SUMMARY_TABLE,
                    {
                        "recording_id": recording_id,
                        "zarr_path": str(path),
                        "source_lineage_hash": hashlib.sha256(
                            str(path).encode("utf-8")
                        ).hexdigest(),
                        "stimulus_step_count": 0,
                    },
                )
            ]
        },
    )


def _publish(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    sources: list[Path],
    **kwargs: object,
) -> dict[str, object]:
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        _stub_source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    return export_sources(
        sources,
        output_root=root,
        tables=(RECORDING_SUMMARY_TABLE,),
        export_run_id="atomic_test",
        jobs=1,
        **kwargs,
    )


def test_overwrite_commits_one_new_immutable_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    first = _publish(monkeypatch, root, [tmp_path / "a.zarr", tmp_path / "b.zarr"])
    first_generation = first["publication"]["generation_id"]

    second = _publish(monkeypatch, root, [tmp_path / "a.zarr"], overwrite=True)
    selected = manifest_selected_part_files(
        root,
        "atomic_test",
        RECORDING_SUMMARY_TABLE,
    )

    assert second["publication"]["generation_id"] != first_generation
    assert len(selected) == 1
    assert first_generation not in str(selected[0])
    assert validate_export_run(root, "atomic_test")["part_count"] == 1


def test_manifest_commit_failure_preserves_previous_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    first = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    manifest_path = Path(first["manifest_path"])
    old_generation = first["publication"]["generation_id"]
    real_replace = os.replace

    def fail_manifest(source: object, destination: object) -> None:
        if Path(destination) == manifest_path:
            raise OSError("injected manifest commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.os.replace",
        fail_manifest,
    )
    with pytest.raises(OSError, match="manifest commit failure"):
        _publish(monkeypatch, root, [tmp_path / "a.zarr", tmp_path / "b.zarr"], overwrite=True)

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["publication"]["generation_id"] == old_generation
    assert validate_export_run(root, "atomic_test")["status"] == "valid"
    assert not list((root / "v1" / ".staging").glob("*"))


def test_concurrent_first_publication_uses_manifest_compare_and_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        _stub_source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    import fisheye.utils.export_cross_recording_analytics as exporter

    real_validate = exporter.validate_staged_publication
    ready_to_commit = threading.Barrier(2)

    def synchronized_validate(staging_root: Path, payload: object) -> None:
        real_validate(staging_root, payload)
        ready_to_commit.wait(timeout=10)

    monkeypatch.setattr(exporter, "validate_staged_publication", synchronized_validate)

    def publish(source: str) -> dict[str, object]:
        return export_sources(
            [tmp_path / source],
            output_root=root,
            tables=(RECORDING_SUMMARY_TABLE,),
            export_run_id="atomic_race",
            jobs=1,
        )

    outcomes: list[object] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, name) for name in ("a.zarr", "b.zarr")]
        for future in futures:
            try:
                outcomes.append(future.result())
            except Exception as exc:
                outcomes.append(exc)

    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert sum(
        isinstance(item, RuntimeError) and "changed during publication" in str(item)
        for item in outcomes
    ) == 1
    assert validate_export_run(root, "atomic_race")["status"] == "valid"
    generations = list(
        (root / "v1" / ".generations" / "export_run_id=atomic_race").glob(
            "generation=*"
        )
    )
    assert len(generations) == 1


def test_strict_validation_rejects_unlisted_missing_and_tampered_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    manifest = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    selected = manifest_selected_part_files(root, "atomic_test", RECORDING_SUMMARY_TABLE)
    generation_root = root / manifest["publication"]["generation_path"]

    extra = generation_root / "unlisted.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(ExportValidationError, match="outside its exact inventory"):
        validate_export_run(root, "atomic_test")
    extra.unlink()

    part = selected[0]
    original = part.read_bytes()
    part.write_bytes(original + b"tampered")
    with pytest.raises(ExportValidationError, match="digest mismatch"):
        validate_export_run(root, "atomic_test")
    part.write_bytes(original)
    missing = part.with_suffix(".missing")
    part.rename(missing)
    with pytest.raises(ExportValidationError, match="missing part"):
        validate_export_run(root, "atomic_test")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda publication: publication.update({"unexpected": True}), "field set"),
        (lambda publication: publication.update({"state": "running"}), "state"),
        (lambda publication: publication.update({"schema_version": True}), "schema version"),
        (lambda publication: publication.update({"schema_id": 1}), "schema ID"),
        (lambda publication: publication.update({"state": 1}), "state"),
        (lambda publication: publication.update({"generation_id": " test "}), "generation ID"),
        (lambda publication: publication.update({"generation_id": 1}), "generation ID"),
        (lambda publication: publication.update({"generation_path": 1}), "generation path"),
        (
            lambda publication: publication.update(
                {
                    "generation_path": str(publication["generation_path"]).replace(
                        "v1/", "v1//", 1
                    )
                }
            ),
            "generation path/identity",
        ),
        (
            lambda publication: publication["parts_by_table"][
                RECORDING_SUMMARY_TABLE
            ][0].update(
                {
                    "path": str(
                        publication["parts_by_table"][RECORDING_SUMMARY_TABLE][0][
                            "path"
                        ]
                    ).replace("/tables/", "//tables/", 1)
                }
            ),
            "part path",
        ),
        (
            lambda publication: publication["parts_by_table"][RECORDING_SUMMARY_TABLE][0].update(
                {"row_count": True}
            ),
            "row count",
        ),
    ],
)
def test_strict_validation_rejects_malformed_publication_envelopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: object,
    match: str,
) -> None:
    root = tmp_path / "exports"
    manifest = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(payload["publication"])
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match=match):
        validate_export_run(root, "atomic_test")


def test_strict_validation_rejects_non_string_manifest_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    manifest = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["export_run_id"] = 7
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="run ID must be a string"):
        validate_export_run(root, "atomic_test")


def test_strict_validation_rejects_noncanonical_manifest_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    manifest = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = payload["publication"]["parts_by_table"]
    inventory[f" {RECORDING_SUMMARY_TABLE}"] = inventory.pop(
        RECORDING_SUMMARY_TABLE
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="Invalid table name"):
        validate_export_run(root, "atomic_test")


def test_strict_validation_rejects_duplicate_logical_and_inventory_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    manifest = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["part_files_by_table"][RECORDING_SUMMARY_TABLE] *= 2
    payload["publication"]["parts_by_table"][RECORDING_SUMMARY_TABLE] *= 2
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="duplicate publication part path"):
        validate_export_run(root, "atomic_test")


def test_strict_validation_rejects_symlink_part_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    part = manifest_selected_part_files(
        root,
        "atomic_test",
        RECORDING_SUMMARY_TABLE,
    )[0]
    target = part.with_name("physical-target.parquet")
    part.rename(target)
    part.symlink_to(target.name)

    with pytest.raises(ExportValidationError, match="symbolic-link alias"):
        validate_export_run(root, "atomic_test")


def test_strict_validation_rejects_generation_symlink_escape_before_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    manifest = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    generation = root / manifest["publication"]["generation_path"]
    outside = tmp_path / "outside-generation"
    generation.rename(outside)
    generation.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ExportValidationError, match="symbolic-link alias"):
        validate_export_run(root, "atomic_test")


def test_zero_file_publication_still_requires_generation_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_source(path: Path, **_kwargs: object) -> SourceExportResult:
        return SourceExportResult(
            zarr_path=str(path),
            recording_id=path.stem,
            rows_by_table={RECORDING_SUMMARY_TABLE: []},
        )

    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        empty_source,
    )
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.get_git_info",
        lambda _path: {"commit_hash": "test", "is_dirty": False},
    )
    root = tmp_path / "exports"
    manifest = export_sources(
        [tmp_path / "a.zarr"],
        output_root=root,
        tables=(RECORDING_SUMMARY_TABLE,),
        export_run_id="empty_test",
        jobs=1,
    )
    assert validate_export_run(root, "empty_test")["part_count"] == 0
    generation = root / manifest["publication"]["generation_path"]
    generation.rename(generation.with_name("generation=detached"))

    with pytest.raises(ExportValidationError, match="generation directory is missing"):
        validate_export_run(root, "empty_test")


def test_consumers_and_registry_use_only_manifest_selected_absolute_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    manifest = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    selected = manifest_selected_part_files(root, "atomic_test", RECORDING_SUMMARY_TABLE)
    context = ViewerContext(export_root=root, export_run_id="atomic_test")
    assert parquet_files(context, RECORDING_SUMMARY_TABLE) == selected

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        assert index_export_manifest(registry, Path(manifest["manifest_path"])) == "atomic_test"
    finally:
        registry.close()
    resolution = resolve_latest_export_table(
        registry_path=tmp_path / "registry.sqlite",
        table_name=RECORDING_SUMMARY_TABLE,
        export_run_id="atomic_test",
    )
    assert resolution.part_files
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in resolution.part_files)

    external_manifest = tmp_path / "stale-external-manifest.json"
    external_manifest.write_bytes(Path(manifest["manifest_path"]).read_bytes())
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute(
            "UPDATE analytics_exports SET export_manifest_path = ? "
            "WHERE export_run_id = ?",
            (str(external_manifest), "atomic_test"),
        )
        registry.conn.commit()
    finally:
        registry.close()
    with pytest.raises(LookupError, match="canonical output-root path"):
        resolve_latest_export_table(
            registry_path=tmp_path / "registry.sqlite",
            table_name=RECORDING_SUMMARY_TABLE,
            export_run_id="atomic_test",
        )


def test_viewer_reports_manifest_selected_generation_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    selected = manifest_selected_part_files(root, "atomic_test", RECORDING_SUMMARY_TABLE)
    monkeypatch.setattr(
        "fisheye.group_analytics_viewer.query.CHASER_TABLES",
        (RECORDING_SUMMARY_TABLE,),
    )
    monkeypatch.setattr(
        "fisheye.group_analytics_viewer.query.OPTIONAL_CHASER_TABLES",
        (),
    )
    context = ViewerContext(export_root=root, export_run_id="atomic_test")

    health = build_health_report(context)
    summary = query_export_summary(context)

    assert health.details["tables"][RECORDING_SUMMARY_TABLE]["path"] == str(
        selected[0].parent
    )
    assert summary["tables"][0]["table_path"] == str(selected[0].parent)


def test_latest_ignores_lexically_newer_invalid_and_legacy_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    manifest_dir = root / "v1" / "manifests"
    (manifest_dir / "export_run_id=zz_invalid.json").write_text(
        json.dumps({"export_run_id": "zz_invalid"}),
        encoding="utf-8",
    )
    assert resolve_export_run_id(root, "latest") == "atomic_test"
    assert resolve_montage_export_run_id(root, "latest") == "atomic_test"


def test_statistics_latest_skips_newer_legacy_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    base = _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    generation = (
        Path("v1")
        / ".generations"
        / "export_run_id=stats_old"
        / "generation=test"
    )
    stats_part = root / generation / "tables" / STATISTICS_TABLE / "part.parquet"
    stats_part.parent.mkdir(parents=True)
    stats_table = pa.Table.from_pylist(
        [
            canonicalize_export_row(
                STATISTICS_TABLE,
                {
                    "stat_result_id": "stats_old:test",
                    "stats_run_id": "stats_old",
                    "source_export_run_id": "atomic_test",
                    "source_table": RECORDING_SUMMARY_TABLE,
                    "metric_name": "test_metric",
                    "status": "computed",
                },
            )
        ]
    ).replace_schema_metadata(
        {
            b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode(),
            b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode(),
            b"palette.table_contract": json.dumps(
                TABLE_CONTRACTS[STATISTICS_TABLE].to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            b"palette.arrow_schema_mode": b"inferred_v2_compatibility",
        }
    )
    pq.write_table(stats_table, stats_part)
    relative_part = (
        generation / "tables" / STATISTICS_TABLE / stats_part.name
    ).as_posix()
    valid = {
        "export_run_id": "stats_old",
        "schema_id": base["schema_id"],
        "schema_version": base["schema_version"],
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "source_export_run_id": "atomic_test",
        "source_export_manifest_sha256": hashlib.sha256(
            Path(base["manifest_path"]).read_bytes()
        ).hexdigest(),
        "output_tables": [STATISTICS_TABLE],
        "table_contracts": contract_snapshot([STATISTICS_TABLE]),
        "arrow_schema_contracts": arrow_contract_envelope([STATISTICS_TABLE]),
        "row_counts_by_table": {STATISTICS_TABLE: 1},
        "part_files_by_table": {STATISTICS_TABLE: [relative_part]},
        "capabilities": [
            item.capability_id
            for item in resolve_capabilities(
                {STATISTICS_TABLE: tuple(stats_table.schema.names)}
            )
            if item.available
        ],
        "publication": {
            "schema_id": "palette.analytics_export.publication",
            "schema_version": 1,
            "state": "complete",
            "generation_id": "test",
            "generation_path": generation.as_posix(),
            "parts_by_table": {
                STATISTICS_TABLE: [
                    {
                        "path": relative_part,
                        "sha256": sha256_file(stats_part),
                        "size_bytes": stats_part.stat().st_size,
                        "row_count": 1,
                    }
                ]
            },
        },
    }
    manifest_dir = root / "v1" / "manifests"
    (manifest_dir / "export_run_id=stats_old.json").write_text(
        json.dumps(valid),
        encoding="utf-8",
    )
    legacy = {
        "export_run_id": "zz_stats_legacy",
        "schema_id": base["schema_id"],
        "schema_version": base["schema_version"],
        "created_at_utc": "2026-12-01T00:00:00+00:00",
        "source_export_run_id": "atomic_test",
        "output_tables": [STATISTICS_TABLE],
        "row_counts_by_table": {STATISTICS_TABLE: 1},
    }
    (manifest_dir / "export_run_id=zz_stats_legacy.json").write_text(
        json.dumps(legacy),
        encoding="utf-8",
    )

    context = ViewerContext(export_root=root, export_run_id="atomic_test")
    assert resolve_statistics_run_id(context) == "stats_old"

    mismatched = dict(valid)
    mismatched["source_export_manifest_sha256"] = "0" * 64
    (manifest_dir / "export_run_id=stats_old.json").write_text(
        json.dumps(mismatched),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_export_manifest_sha256"):
        resolve_statistics_run_id(
            ViewerContext(
                export_root=root,
                export_run_id="atomic_test",
                stats_run_id="stats_old",
            )
        )


def test_export_run_id_rejects_path_traversal_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        _stub_source,
    )
    with pytest.raises(ValueError, match="Invalid export run ID"):
        export_sources(
            [tmp_path / "a.zarr"],
            output_root=tmp_path / "exports",
            tables=(RECORDING_SUMMARY_TABLE,),
            export_run_id="../escape",
            jobs=1,
        )
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "nested/run", "bad\nid", "café"],
)
def test_manifest_read_helpers_reject_unsafe_run_ids_before_path_access(
    tmp_path: Path,
    run_id: str,
) -> None:
    root = tmp_path / "exports"

    with pytest.raises(ValueError, match="Invalid export run ID"):
        load_export_manifest(root, run_id)
    with pytest.raises(ValueError, match="Invalid export run ID"):
        manifest_selected_part_files(root, run_id, RECORDING_SUMMARY_TABLE)


def test_manifest_namespace_symlink_blocks_all_manifest_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "v1").mkdir(parents=True)
    (root / "v1" / "manifests").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _publish(monkeypatch, root, [tmp_path / "a.zarr"])
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_export_manifest(root, "atomic_test")
    with pytest.raises(ExportValidationError, match="must not be a symlink"):
        validate_export_run(root, "atomic_test")
    with pytest.raises(ValueError, match="must not be a symlink"):
        write_goodcopbadcop_statistics(
            [],
            {},
            export_root=root,
            stats_run_id="stats_test",
        )

    assert not any(outside.iterdir())


@pytest.mark.parametrize("namespace", [".staging", ".generations", ".locks"])
def test_publication_rejects_symlinked_lifecycle_namespaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
) -> None:
    root = tmp_path / "exports"
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "v1" / "manifests").mkdir(parents=True)
    (root / "v1" / namespace).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _publish(monkeypatch, root, [tmp_path / "a.zarr"])

    assert not any(outside.iterdir())


def test_publication_rejects_preexisting_lock_file_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    lock_dir = root / "v1" / ".locks"
    lock_dir.mkdir(parents=True)
    outside_lock = tmp_path / "outside.lock"
    outside_lock.write_bytes(b"unchanged")
    (lock_dir / ".export_run_id=atomic_test.json.lock").symlink_to(outside_lock)

    with pytest.raises(ValueError, match="lock must not be a symlink"):
        _publish(monkeypatch, root, [tmp_path / "a.zarr"])

    assert outside_lock.read_bytes() == b"unchanged"


@pytest.mark.parametrize(
    "run_id",
    [
        "bad:id",
        "bad*id",
        "bad?id",
        'bad"id',
        "bad<id",
        "bad>id",
        "bad|id",
        "bad/id",
        "bad\\id",
        "bad\nid",
        "bad\x00id",
        "café",
        ".hidden",
        "-option",
        "_private",
        "trailing.",
        "trailing-",
        "trailing_",
    ],
)
def test_export_run_id_rejects_nonportable_characters_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    monkeypatch.setattr(
        "fisheye.utils.export_cross_recording_analytics.export_one_zarr",
        _stub_source,
    )
    with pytest.raises(ValueError, match="Invalid export run ID"):
        export_sources(
            [tmp_path / "a.zarr"],
            output_root=tmp_path / "exports",
            tables=(RECORDING_SUMMARY_TABLE,),
            export_run_id=run_id,
            jobs=1,
        )
    assert not (tmp_path / "exports").exists()


class _Group(dict[str, object]):
    def __init__(self, children: dict[str, object] | None = None, **attrs: object) -> None:
        super().__init__(children or {})
        self.attrs = attrs


def test_swim_bout_export_selection_rejects_pointer_mismatch_and_ineligible_newer_child() -> None:
    old = _Group(
        palette_run_completion_status="complete",
        stage_selector_eligible=True,
    )
    newer = _Group(
        palette_run_completion_status="complete",
        stage_selector_eligible=False,
    )
    parent = _Group(
        {"old": old, "zz_newer": newer},
        latest="old",
        latest_complete="old",
    )
    root = _Group({"analysis/swim_bout_runs": parent})
    _group, run_name, error = _latest_run(root, "analysis/swim_bout_runs")
    assert run_name == "old"
    assert error is None

    parent.attrs["latest_complete"] = "zz_newer"
    _group, run_name, error = _latest_run(root, "analysis/swim_bout_runs")
    assert run_name is None
    assert "failed closed" in str(error)

    legacy_parent = _Group({"zz_statusless": _Group()})
    legacy_root = _Group({"analysis/swim_bout_runs": legacy_parent})
    _group, run_name, error = _latest_run(legacy_root, "analysis/swim_bout_runs")
    assert run_name is None
    assert "failed closed" in str(error)
