from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analytics_exports.validated_behavior_cohort import (
    validated_behavior_manifest_path,
)
from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.analytics_exports.validated_behavior_product_catalog import (
    BEHAVIOR_DISTRIBUTION,
    BEHAVIOR_DISTRIBUTION_REPORT,
    ValidatedBehaviorProductCatalogError,
    adopt_validated_behavior_product,
    canonical_validated_behavior_product_dir,
    read_validated_behavior_product_catalog,
    resolve_validated_behavior_product,
    validated_behavior_product_catalog_manifest_path,
)
from fisheye.group_statistics.validated_behavior_distribution_report import (
    render_validated_behavior_distribution_report,
)
from fisheye.group_statistics.validated_behavior_distribution_views import (
    ValidatedBehaviorDistributionViewSource,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    DistributionMetricSpec,
    SCOPE_ORDER,
)
from fisheye.group_statistics.validated_behavior_appearance import APPEARANCE_COLUMNS
from fisheye.group_statistics.validated_behavior_distributions import (
    ValidatedBehaviorDistributionConfig,
    ValidatedBehaviorDistributionResult,
    _SparseAccumulator,
    _cohort_bin_rows,
    _finalize_recording_bins,
    _reduce_metric_values,
    write_validated_behavior_distributions,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
import fisheye.utils.compute_validated_behavior_distributions as distribution_cli
import fisheye.utils.render_validated_behavior_distributions as report_cli

NOW = "2026-09-03T12:00:00Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture_dataset(tmp_path: Path) -> ValidatedBehaviorExportDataset:
    publication = (tmp_path / "cohort-package" / "publication").resolve()
    publication.mkdir(parents=True)
    run_id = "fixture-export-v1"
    body = {
        "schema_id": "palette.analytics.validated_behavior_cohort_export",
        "schema_version": 1,
        "status": "complete_selector_ineligible",
        "export_run_id": run_id,
        "validation_receipt": {"record_sha256": "b" * 64},
    }
    manifest = {**body, "record_sha256": canonical_json_sha256(body)}
    _write_json(validated_behavior_manifest_path(publication, run_id), manifest)
    return ValidatedBehaviorExportDataset(
        root=publication,
        export_run_id=run_id,
        manifest=manifest,
        membership={},
        bundle_set={},
        table_specs={},
        validation_mode="receipt",
    )


def _duration_spec() -> DistributionMetricSpec:
    return DistributionMetricSpec(
        metric_id="fixture.duration_s",
        metric_family="fixture",
        source_surface="bout_observations",
        value_column="duration_s",
        unit="s",
        bin_width=0.1,
        lower_bound=0.0,
        upper_bound=None,
        coverage_policy="zero_anchored_cover_valid_max",
        weighting_ids=("event",),
        group_columns=(),
        validity_policy_id="finite_nonnegative_fixture_v1",
        scope_binding_id="whole_session_and_epoch_fixture_v1",
        interpretation="Fixture duration",
    )


def _appearance(source_digest: str) -> dict[str, object]:
    query = {
        "export_run_id": "fixture-export-v1",
        "export_manifest_record_sha256": source_digest,
        "export_plan_sha256": "c" * 64,
        "table_name": "chaser_occurrences",
        "table_contract_sha256": "d" * 64,
        "grain": "fixture",
        "selected_columns": list(APPEARANCE_COLUMNS),
        "predicate_description": "fixture",
        "analysis_unit_policy_sha256": "e" * 64,
        "capability_policy": "fixture",
        "semantic_metadata": {},
    }
    body = {
        "schema_id": "palette.analytics.validated_behavior.chaser_appearance_dimension",
        "schema_version": 1,
        "method_id": "phase_c_chaser_occurrence_projection_v1",
        "status": "complete",
        "source_table": "chaser_occurrences",
        "join_fields": ["recording_id", "chaser_identity_code"],
        "color_semantics": "experimental_protocol_rgba",
        "role_semantics": "independent_marker_shape_and_text",
        "color_role_independence": True,
        "source_query_identity": query,
        "rows": [
            {
                "recording_id": "recording-a",
                "chaser_identity_code": 1,
                "chaser_index": 0,
                "chaser_identity": "stimulus-exact:chaser_index:0",
                "behavior_role_code": 1,
                "behavior_role": "aggressive",
                "stimulus_run_path": "analysis/stimulus_runs/exact",
                "source_protocol_sha256": "1" * 64,
                "experimental_color_r": 0.0,
                "experimental_color_g": 0.0,
                "experimental_color_b": 1.0,
                "experimental_color_a": 1.0,
                "experimental_color_hex": "#0000ff",
                "experimental_color_css": "rgba(0, 0, 255, 1)",
                "contrast_outline_hex": "#ffffff",
                "plotly_role_symbol": "star",
                "matplotlib_role_marker": "*",
                "appearance_schema_id": (
                    "palette.visualization.chaser_appearance_projection"
                ),
                "appearance_schema_version": 1,
                "appearance_policy_id": (
                    "protocol_rgba_independent_behavior_role_glyph_v1"
                ),
                "appearance_projection_sha256": "2" * 64,
                "occurrence_binding_sha256": "3" * 64,
                "color_semantics": "experimental_protocol_rgba",
                "role_semantics": "independent_marker_shape_and_text",
                "color_role_independence": True,
            }
        ],
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def _write_fixture_distribution(
    target: Path,
    dataset: ValidatedBehaviorExportDataset,
    *,
    run_id: str,
    source_digest: str | None = None,
) -> Path:
    digest = source_digest or dataset.cache_identity
    spec = _duration_spec()
    config = ValidatedBehaviorDistributionConfig(
        distribution_run_id=run_id, metric_specs=(spec,)
    )
    accumulator = _SparseAccumulator()
    values = np.asarray([0.05, 0.15], dtype=np.float64)
    scopes = {
        scope: (
            np.ones(values.shape, dtype=bool)
            if scope == "whole_session"
            else np.zeros(values.shape, dtype=bool)
        )
        for scope in SCOPE_ORDER
    }
    _reduce_metric_values(
        accumulator,
        config=config,
        spec=spec,
        source_export_run_id=dataset.export_run_id,
        source_export_manifest_sha256=digest,
        recording_id="recording-a",
        values=values,
        scope_masks=scopes,
        base_valid=np.ones(values.shape, dtype=bool),
        group_arrays={},
        identity_arrays={"source": np.asarray(["exact", "exact"])},
        time_weights_s=None,
        valid_duration_by_scope={scope: 1.0 for scope in SCOPE_ORDER},
    )
    recipes, support, sparse = _finalize_recording_bins(
        config=config,
        source_export_run_id=dataset.export_run_id,
        source_export_manifest_sha256=digest,
        accumulator=accumulator,
    )
    cohort = _cohort_bin_rows(
        config=config,
        source_export_run_id=dataset.export_run_id,
        source_export_manifest_sha256=digest,
        parent_recording_count=1,
        recipes=recipes,
        support_rows=support,
        sparse_rows=sparse,
    )
    result = ValidatedBehaviorDistributionResult(
        config=config,
        source_export={
            "path": str(dataset.root),
            "export_run_id": dataset.export_run_id,
            "export_manifest_record_sha256": digest,
        },
        cohort_summary={"parent_recording_count": 1},
        source_queries=(),
        epoch_child_receipts=(),
        histogram_recipes=recipes,
        chaser_appearance_dimension=_appearance(digest),
        bout_observations=(),
        inter_bout_interval_observations=(),
        recording_support=support,
        recording_nonzero_bins=sparse,
        cohort_bins=cohort,
    )
    write_validated_behavior_distributions(result, target)
    return target


def test_adoption_copies_exact_product_and_does_not_mutate_export(tmp_path: Path):
    dataset = _fixture_dataset(tmp_path)
    source = _write_fixture_distribution(
        tmp_path / "legacy" / "distribution-a",
        dataset,
        run_id="distribution-a",
    )
    publication_before = {
        path.relative_to(dataset.root): path.read_bytes()
        for path in dataset.root.rglob("*")
        if path.is_file()
    }

    adopted = adopt_validated_behavior_product(
        dataset,
        product_kind=BEHAVIOR_DISTRIBUTION,
        source_product_root=source,
        catalog_generation_id="catalog-a",
        created_at_utc=NOW,
    )

    target = canonical_validated_behavior_product_dir(
        dataset.root, BEHAVIOR_DISTRIBUTION, "distribution-a"
    )
    assert adopted["copied"] is True
    assert Path(str(adopted["product_root"])) == target
    assert source.is_dir()
    assert target.is_dir()
    assert {
        path.relative_to(dataset.root): path.read_bytes()
        for path in dataset.root.rglob("*")
        if path.is_file()
    } == publication_before

    catalog = read_validated_behavior_product_catalog(
        dataset.root, dataset.export_run_id, validate_products=True
    )
    assert len(catalog["products"]) == 1
    assert catalog["safety"]["scientific_authority"] is False
    resolved = resolve_validated_behavior_product(
        dataset.root,
        dataset.export_run_id,
        product_kind=BEHAVIOR_DISTRIBUTION,
    )
    assert resolved.root == target
    assert resolved.manifest_record_sha256 == adopted["product_manifest_record_sha256"]
    assert dataset.products(product_kind=BEHAVIOR_DISTRIBUTION) == (resolved,)
    assert (
        dataset.product(
            product_kind=BEHAVIOR_DISTRIBUTION,
            product_run_id="distribution-a",
        )
        == resolved
    )

    repeated = adopt_validated_behavior_product(
        dataset,
        product_kind=BEHAVIOR_DISTRIBUTION,
        source_product_root=source,
        catalog_generation_id="unused-because-reused",
        created_at_utc=NOW,
    )
    assert repeated["copied"] is False
    assert repeated["catalog_reused"] is True
    assert repeated["catalog_generation_id"] == "catalog-a"


def test_catalog_append_preserves_history_and_ambiguity_fails_closed(
    tmp_path: Path,
):
    dataset = _fixture_dataset(tmp_path)
    first = _write_fixture_distribution(
        tmp_path / "legacy" / "distribution-a",
        dataset,
        run_id="distribution-a",
    )
    second = _write_fixture_distribution(
        tmp_path / "legacy" / "distribution-b",
        dataset,
        run_id="distribution-b",
    )
    adopt_validated_behavior_product(
        dataset,
        product_kind=BEHAVIOR_DISTRIBUTION,
        source_product_root=first,
        catalog_generation_id="catalog-a",
        created_at_utc=NOW,
    )
    adopted_second = adopt_validated_behavior_product(
        dataset,
        product_kind=BEHAVIOR_DISTRIBUTION,
        source_product_root=second,
        catalog_generation_id="catalog-b",
        created_at_utc="2026-09-03T12:01:00Z",
    )

    catalog = read_validated_behavior_product_catalog(
        dataset.root, dataset.export_run_id
    )
    assert len(catalog["products"]) == 2
    assert catalog["previous_catalog"]["catalog_generation_id"] == "catalog-a"
    first_generation = (
        dataset.root.parent
        / "products"
        / "validated_behavior"
        / "v1"
        / "catalog"
        / ".generations"
        / f"export_run_id={dataset.export_run_id}"
        / "generation=catalog-a"
        / "catalog.json"
    )
    assert len(json.loads(first_generation.read_text())["products"]) == 1
    with pytest.raises(
        ValidatedBehaviorProductCatalogError, match="Multiple cataloged"
    ):
        resolve_validated_behavior_product(
            dataset.root,
            dataset.export_run_id,
            product_kind=BEHAVIOR_DISTRIBUTION,
        )
    selected = resolve_validated_behavior_product(
        dataset.root,
        dataset.export_run_id,
        product_kind=BEHAVIOR_DISTRIBUTION,
        product_run_id="distribution-b",
    )
    assert selected.root == Path(str(adopted_second["product_root"]))


def test_adoption_rejects_product_from_another_export_before_copying(
    tmp_path: Path,
):
    dataset = _fixture_dataset(tmp_path)
    source = _write_fixture_distribution(
        tmp_path / "legacy" / "foreign-distribution",
        dataset,
        run_id="foreign-distribution",
        source_digest="f" * 64,
    )
    target = canonical_validated_behavior_product_dir(
        dataset.root, BEHAVIOR_DISTRIBUTION, "foreign-distribution"
    )
    with pytest.raises(
        ValidatedBehaviorProductCatalogError, match="another validated-behavior export"
    ):
        adopt_validated_behavior_product(
            dataset,
            product_kind=BEHAVIOR_DISTRIBUTION,
            source_product_root=source,
            catalog_generation_id="foreign-catalog",
            created_at_utc=NOW,
        )
    assert not target.exists()
    assert not validated_behavior_product_catalog_manifest_path(
        dataset.root, dataset.export_run_id
    ).exists()


def test_canonical_product_namespace_rejects_symbolic_link_escape(tmp_path: Path):
    dataset = _fixture_dataset(tmp_path)
    outside = tmp_path / "outside-products"
    outside.mkdir()
    (dataset.root.parent / "products").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ValidatedBehaviorProductCatalogError, match="symbolic-link alias"
    ):
        canonical_validated_behavior_product_dir(
            dataset.root, BEHAVIOR_DISTRIBUTION, "distribution-a"
        )


def test_missing_catalog_selector_cannot_start_disconnected_history(tmp_path: Path):
    dataset = _fixture_dataset(tmp_path)
    first = _write_fixture_distribution(
        tmp_path / "legacy" / "distribution-a",
        dataset,
        run_id="distribution-a",
    )
    adopt_validated_behavior_product(
        dataset,
        product_kind=BEHAVIOR_DISTRIBUTION,
        source_product_root=first,
        catalog_generation_id="catalog-a",
        created_at_utc=NOW,
    )
    selected_catalog = validated_behavior_product_catalog_manifest_path(
        dataset.root, dataset.export_run_id
    )
    selected_catalog.unlink()
    second = _write_fixture_distribution(
        tmp_path / "legacy" / "distribution-b",
        dataset,
        run_id="distribution-b",
    )
    second_target = canonical_validated_behavior_product_dir(
        dataset.root, BEHAVIOR_DISTRIBUTION, "distribution-b"
    )

    with pytest.raises(
        ValidatedBehaviorProductCatalogError, match="disconnected history"
    ):
        adopt_validated_behavior_product(
            dataset,
            product_kind=BEHAVIOR_DISTRIBUTION,
            source_product_root=second,
            catalog_generation_id="catalog-b",
            created_at_utc=NOW,
        )
    assert not second_target.exists()


def test_catalog_rejects_product_manifest_tampering(tmp_path: Path):
    dataset = _fixture_dataset(tmp_path)
    source = _write_fixture_distribution(
        tmp_path / "legacy" / "distribution-a",
        dataset,
        run_id="distribution-a",
    )
    adopted = adopt_validated_behavior_product(
        dataset,
        product_kind=BEHAVIOR_DISTRIBUTION,
        source_product_root=source,
        catalog_generation_id="catalog-a",
        created_at_utc=NOW,
    )
    manifest_path = Path(str(adopted["product_root"])) / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["status"] = "tampered"
    _write_json(manifest_path, value)

    with pytest.raises(
        ValidatedBehaviorProductCatalogError, match="self digest is stale"
    ):
        read_validated_behavior_product_catalog(dataset.root, dataset.export_run_id)


def test_report_requires_exact_cataloged_colocated_parent(tmp_path: Path):
    dataset = _fixture_dataset(tmp_path)
    external_distribution = _write_fixture_distribution(
        tmp_path / "legacy" / "distribution-a",
        dataset,
        run_id="distribution-a",
    )
    external_report = tmp_path / "legacy" / "report-a"
    render_validated_behavior_distribution_report(
        ValidatedBehaviorDistributionViewSource.open(external_distribution),
        report_run_id="report-a",
        output_dir=external_report,
    )

    with pytest.raises(ValidatedBehaviorProductCatalogError):
        adopt_validated_behavior_product(
            dataset,
            product_kind=BEHAVIOR_DISTRIBUTION_REPORT,
            source_product_root=external_report,
            catalog_generation_id="report-before-parent",
            created_at_utc=NOW,
        )

    distribution = adopt_validated_behavior_product(
        dataset,
        product_kind=BEHAVIOR_DISTRIBUTION,
        source_product_root=external_distribution,
        catalog_generation_id="catalog-distribution",
        created_at_utc=NOW,
    )
    with pytest.raises(
        ValidatedBehaviorProductCatalogError, match="exact co-located catalog parent"
    ):
        adopt_validated_behavior_product(
            dataset,
            product_kind=BEHAVIOR_DISTRIBUTION_REPORT,
            source_product_root=external_report,
            catalog_generation_id="report-with-old-parent-path",
            created_at_utc=NOW,
        )

    canonical_distribution = Path(str(distribution["product_root"]))
    canonical_source_report = tmp_path / "new-report-a"
    render_validated_behavior_distribution_report(
        ValidatedBehaviorDistributionViewSource.open(canonical_distribution),
        report_run_id="report-a",
        output_dir=canonical_source_report,
    )
    adopted_report = adopt_validated_behavior_product(
        dataset,
        product_kind=BEHAVIOR_DISTRIBUTION_REPORT,
        source_product_root=canonical_source_report,
        catalog_generation_id="catalog-report",
        created_at_utc="2026-09-03T12:01:00Z",
    )
    assert adopted_report["copied"] is True
    selected = resolve_validated_behavior_product(
        dataset.root,
        dataset.export_run_id,
        product_kind=BEHAVIOR_DISTRIBUTION_REPORT,
        product_run_id="report-a",
    )
    assert selected.root == Path(str(adopted_report["product_root"]))


def test_distribution_cli_defaults_to_colocated_cataloged_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset = _fixture_dataset(tmp_path)
    result = SimpleNamespace(cohort_summary={"parent_recording_count": 1})
    written: list[Path] = []
    registered: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        distribution_cli,
        "ValidatedBehaviorExportDataset",
        SimpleNamespace(open=lambda *_args, **_kwargs: dataset),
    )
    monkeypatch.setattr(
        distribution_cli,
        "distribution_metric_specs_for_families",
        lambda _families: (_duration_spec(),),
    )
    monkeypatch.setattr(
        distribution_cli,
        "compute_validated_behavior_distributions",
        lambda *_args, **_kwargs: result,
    )

    def fake_write(_result: object, output_dir: Path) -> dict[str, object]:
        written.append(Path(output_dir))
        return {
            "manifest_path": str(Path(output_dir) / "manifest.json"),
            "record_sha256": "1" * 64,
        }

    def fake_register(
        _dataset: object, *, product_kind: str, product_root: Path
    ) -> dict[str, object]:
        registered.append((product_kind, Path(product_root)))
        return {
            "catalog_manifest_path": "/fixture/catalog.json",
            "record_sha256": "2" * 64,
            "catalog_generation_id": "catalog-a",
        }

    monkeypatch.setattr(
        distribution_cli, "write_validated_behavior_distributions", fake_write
    )
    monkeypatch.setattr(
        distribution_cli, "register_validated_behavior_product", fake_register
    )
    assert (
        distribution_cli.main(
            [
                "--export-root",
                str(dataset.root),
                "--source-export-run-id",
                dataset.export_run_id,
                "--distribution-run-id",
                "distribution-default",
                "--apply",
            ]
        )
        == 0
    )
    expected = canonical_validated_behavior_product_dir(
        dataset.root, BEHAVIOR_DISTRIBUTION, "distribution-default"
    )
    assert written == [expected]
    assert registered == [(BEHAVIOR_DISTRIBUTION, expected)]


def test_distribution_report_cli_defaults_to_colocated_cataloged_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset = _fixture_dataset(tmp_path)
    distribution_root = canonical_validated_behavior_product_dir(
        dataset.root, BEHAVIOR_DISTRIBUTION, "distribution-a"
    )
    source = SimpleNamespace(
        root=distribution_root,
        distribution_run_id="distribution-a",
        cache_identity="3" * 64,
        manifest={
            "source_export": {
                "path": str(dataset.root),
                "export_run_id": dataset.export_run_id,
            }
        },
    )
    written: list[Path] = []
    registered: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        report_cli,
        "ValidatedBehaviorDistributionViewSource",
        SimpleNamespace(open=lambda _path: source),
    )
    monkeypatch.setattr(
        report_cli,
        "available_distribution_metrics",
        lambda _source: (
            {
                "metric_id": "fixture.duration_s",
                "weighting_ids": ("event",),
                "interpretation": "Fixture duration",
            },
        ),
    )
    monkeypatch.setattr(
        report_cli,
        "ValidatedBehaviorExportDataset",
        SimpleNamespace(open=lambda *_args, **_kwargs: dataset),
    )
    monkeypatch.setattr(
        report_cli,
        "resolve_validated_behavior_product",
        lambda *_args, **_kwargs: SimpleNamespace(
            root=distribution_root,
            manifest_record_sha256=source.cache_identity,
        ),
    )

    def fake_render(
        _source: object,
        *,
        report_run_id: str,
        output_dir: Path,
        metric_ids: object,
        dpi: int,
        display_range_id: str,
    ) -> dict[str, object]:
        assert report_run_id == "report-default"
        assert metric_ids is None
        assert dpi == 170
        assert display_range_id
        written.append(Path(output_dir))
        return {
            "manifest_path": str(Path(output_dir) / "manifest.json"),
            "record_sha256": "4" * 64,
        }

    def fake_register(
        _dataset: object, *, product_kind: str, product_root: Path
    ) -> dict[str, object]:
        registered.append((product_kind, Path(product_root)))
        return {
            "catalog_manifest_path": "/fixture/catalog.json",
            "record_sha256": "5" * 64,
            "catalog_generation_id": "catalog-report",
        }

    monkeypatch.setattr(
        report_cli, "render_validated_behavior_distribution_report", fake_render
    )
    monkeypatch.setattr(
        report_cli, "register_validated_behavior_product", fake_register
    )
    assert (
        report_cli.main(
            [
                "--distribution-dir",
                str(distribution_root),
                "--report-run-id",
                "report-default",
                "--apply",
            ]
        )
        == 0
    )
    expected = canonical_validated_behavior_product_dir(
        dataset.root, BEHAVIOR_DISTRIBUTION_REPORT, "report-default"
    )
    assert written == [expected]
    assert registered == [(BEHAVIOR_DISTRIBUTION_REPORT, expected)]
