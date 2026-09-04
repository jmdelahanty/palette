"""Co-located, append-only discovery for validated-behavior products.

The validated-behavior cohort export remains an immutable scientific source.
Derived statistics, distributions, and reports live beside that export under a
package-owned product namespace.  A versioned catalog provides the reverse
edge from one exact export manifest to those products without turning path
proximity into scientific authority.

Catalog entries bind exact product manifests.  Family readers remain
responsible for validating product payloads; the catalog never promotes a
product, resolves ``latest``, or changes a source export.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Mapping
import uuid

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .publication import (
    commit_validated_immutable_generation,
    manifest_identity,
    safe_component,
    sha256_file,
)
from .validated_behavior_cohort import validated_behavior_manifest_path
from .validated_behavior_dataset import ValidatedBehaviorExportDataset

CATALOG_SCHEMA_ID = "palette.analytics.validated_behavior.product_catalog"
CATALOG_SCHEMA_VERSION = 1
CATALOG_METHOD_ID = "append_only_exact_manifest_product_discovery_v1"
CATALOG_STATUS = "complete_non_authoritative_discovery"

BEHAVIOR_DISTRIBUTION = "behavior_distribution"
BEHAVIOR_DISTRIBUTION_REPORT = "behavior_distribution_report"
GROUP_STATISTICS = "group_statistics"
GROUP_STATISTICS_REPORT = "group_statistics_report"

CATALOG_SAFETY = MappingProxyType(
    {
        "selector_eligible": False,
        "production_authority": False,
        "scientific_authority": False,
        "registry_update": False,
        "source_export_mutation": False,
        "product_mutation": False,
    }
)

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_CATALOG_FIELDS = {
    "schema_id",
    "schema_version",
    "method_id",
    "status",
    "catalog_generation_id",
    "generation_path",
    "source_export",
    "products",
    "products_sha256",
    "previous_catalog",
    "discovery_semantics",
    "created_at_utc",
    "safety",
    "record_sha256",
}
_SOURCE_EXPORT_FIELDS = {
    "publication_root",
    "export_run_id",
    "manifest_path",
    "manifest_size_bytes",
    "manifest_file_sha256",
    "manifest_record_sha256",
    "validation_receipt_record_sha256",
}
_PRODUCT_ENTRY_FIELDS = {
    "product_kind",
    "product_run_id",
    "product_root",
    "manifest_path",
    "manifest_size_bytes",
    "manifest_file_sha256",
    "manifest_record_sha256",
    "product_schema_id",
    "product_schema_version",
    "product_status",
    "source_export_run_id",
    "source_export_manifest_sha256",
    "source_product",
    "location_policy",
    "entry_sha256",
}
_SOURCE_PRODUCT_FIELDS = {
    "product_kind",
    "product_run_id",
    "manifest_record_sha256",
}
_PREVIOUS_CATALOG_FIELDS = {
    "catalog_generation_id",
    "generation_path",
    "record_sha256",
    "products_sha256",
    "product_count",
}
_DISCOVERY_SEMANTICS = MappingProxyType(
    {
        "catalog_role": "non_authoritative_reverse_discovery_index",
        "scientific_authority": "exact_product_and_source_export_manifests",
        "selection_policy": "exact_product_run_or_unique_compatible_product",
        "ambiguity_policy": "fail_closed",
        "path_discovery": "manifest_only_no_globbing",
        "catalog_update_policy": "append_only_immutable_generations",
    }
)


@dataclass(frozen=True, slots=True)
class _ProductType:
    kind: str
    run_id_field: str
    schema_ids: tuple[str, ...]
    source_kind: str
    source_product_kind: str | None = None


@dataclass(frozen=True, slots=True)
class _ProductInspection:
    kind: str
    run_id: str
    root: Path
    manifest: Mapping[str, Any]
    manifest_path: Path
    manifest_file_sha256: str
    source_export_run_id: str | None
    source_export_manifest_sha256: str
    source_product: Mapping[str, str] | None
    source_product_root: Path | None


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorProductHandle:
    """One manifest-selected product discovered from an exact export."""

    product_kind: str
    product_run_id: str
    root: Path
    manifest_path: Path
    manifest_record_sha256: str
    catalog_record_sha256: str
    entry: Mapping[str, Any]


class ValidatedBehaviorProductCatalogError(ValueError):
    """A product catalog, source binding, or discovery request is unsafe."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorProductCatalogError(message)


def _product_types() -> Mapping[str, _ProductType]:
    # Schema strings are repeated deliberately so importing this lightweight
    # discovery module does not import every statistics/rendering backend.
    return MappingProxyType(
        {
            BEHAVIOR_DISTRIBUTION: _ProductType(
                kind=BEHAVIOR_DISTRIBUTION,
                run_id_field="distribution_run_id",
                schema_ids=("palette.analytics.validated_behavior.distributions",),
                source_kind="source_export",
            ),
            BEHAVIOR_DISTRIBUTION_REPORT: _ProductType(
                kind=BEHAVIOR_DISTRIBUTION_REPORT,
                run_id_field="report_run_id",
                schema_ids=(
                    "palette.analytics.validated_behavior.distributions.static_report",
                ),
                source_kind="source_distribution",
                source_product_kind=BEHAVIOR_DISTRIBUTION,
            ),
            GROUP_STATISTICS: _ProductType(
                kind=GROUP_STATISTICS,
                run_id_field="statistics_run_id",
                schema_ids=("palette.analytics.validated_behavior.group_statistics",),
                source_kind="source_export",
            ),
            GROUP_STATISTICS_REPORT: _ProductType(
                kind=GROUP_STATISTICS_REPORT,
                run_id_field="report_run_id",
                schema_ids=(
                    "palette.analytics.validated_behavior.group_statistics.static_report",
                ),
                source_kind="source_statistics",
                source_product_kind=GROUP_STATISTICS,
            ),
        }
    )


def validated_behavior_product_kinds() -> tuple[str, ...]:
    """Return the closed product-kind roster accepted by the v1 catalog."""

    return tuple(_product_types())


def _product_type(product_kind: object) -> _ProductType:
    kind = str(product_kind)
    result = _product_types().get(kind)
    if result is None:
        _fail(
            f"Unsupported validated-behavior product kind {kind!r}; expected one of "
            f"{validated_behavior_product_kinds()}"
        )
    return result


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest")
    return value


def _timestamp(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one nonempty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidatedBehaviorProductCatalogError(
            f"{field} must be one ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{field} must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        _fail(f"{field} must be expressed in UTC")
    return value


def _canonical_path(path: str | Path, *, field: str, directory: bool) -> Path:
    requested = Path(path).expanduser()
    unresolved = requested if requested.is_absolute() else Path.cwd() / requested
    current = Path(unresolved.anchor)
    for component in unresolved.parts[1:]:
        current /= component
        if current.is_symlink():
            _fail(f"{field} contains a symbolic-link alias: {current}")
    result = unresolved.resolve(strict=False)
    if directory and result.exists() and not result.is_dir():
        _fail(f"{field} is not a directory: {result}")
    return result


def _safe_relative_path(root: Path, value: object, *, field: str) -> Path:
    if type(value) is not str or not value:
        _fail(f"{field} must be one nonempty package-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        _fail(f"{field} must be one canonical package-relative path")
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            _fail(f"{field} contains a symbolic-link alias")
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValidatedBehaviorProductCatalogError(
            f"{field} escapes the dataset package"
        ) from exc
    return resolved


def validated_behavior_package_root(export_root: str | Path) -> Path:
    """Return the package containing the immutable ``publication`` directory."""

    publication = _canonical_path(
        export_root, field="validated-behavior publication root", directory=True
    )
    if publication.name != "publication":
        _fail(
            "Co-located product discovery requires an export root named "
            f"'publication', got: {publication}"
        )
    return publication.parent


def _checked_package_directory(package_root: Path, *parts: str) -> Path:
    package = _canonical_path(
        package_root, field="validated-behavior package root", directory=True
    )
    candidate = package
    for part in parts:
        candidate /= part
        if candidate.is_symlink():
            _fail(f"Product namespace contains a symbolic-link alias: {candidate}")
        if candidate.exists() and not candidate.is_dir():
            _fail(f"Product namespace component is not a directory: {candidate}")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(package)
        except ValueError as exc:
            raise ValidatedBehaviorProductCatalogError(
                f"Product namespace escapes its dataset package: {candidate}"
            ) from exc
        candidate = resolved
    return candidate


def _product_namespace(package_root: Path) -> Path:
    return _checked_package_directory(
        package_root, "products", "validated_behavior", "v1"
    )


def canonical_validated_behavior_product_dir(
    export_root: str | Path,
    product_kind: str,
    product_run_id: str,
) -> Path:
    """Return the canonical immutable location for one derived product."""

    package = validated_behavior_package_root(export_root)
    product_type = _product_type(product_kind)
    run_id = safe_component(product_run_id, label=f"{product_type.kind} run ID")
    return _checked_package_directory(
        package,
        "products",
        "validated_behavior",
        "v1",
        product_type.kind,
        f"run_id={run_id}",
    )


def validated_behavior_product_catalog_manifest_path(
    export_root: str | Path, export_run_id: str
) -> Path:
    package = validated_behavior_package_root(export_root)
    run_id = safe_component(export_run_id, label="export run ID")
    return (
        _checked_package_directory(
            package,
            "products",
            "validated_behavior",
            "v1",
            "catalog",
            "manifests",
        )
        / f"export_run_id={run_id}.json"
    )


def _catalog_generation_relative_path(export_run_id: str, generation_id: str) -> Path:
    run_id = safe_component(export_run_id, label="export run ID")
    generation = safe_component(generation_id, label="catalog generation ID")
    return (
        Path("products")
        / "validated_behavior"
        / "v1"
        / "catalog"
        / ".generations"
        / f"export_run_id={run_id}"
        / f"generation={generation}"
    )


def _catalog_generations_root(package_root: Path, export_run_id: str) -> Path:
    run_id = safe_component(export_run_id, label="export run ID")
    return _checked_package_directory(
        package_root,
        "products",
        "validated_behavior",
        "v1",
        "catalog",
        ".generations",
        f"export_run_id={run_id}",
    )


def _catalog_staging_root(
    package_root: Path, export_run_id: str, generation_id: str
) -> Path:
    run_id = safe_component(export_run_id, label="export run ID")
    generation = safe_component(generation_id, label="catalog generation ID")
    return _checked_package_directory(
        package_root,
        "products",
        "validated_behavior",
        "v1",
        "catalog",
        ".staging",
        f"export_run_id={run_id}-generation={generation}",
    )


def _read_json_object(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{field} is absent or aliased: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorProductCatalogError(
            f"Cannot read {field}: {path}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"{field} must contain one JSON object")
    return value


def _validate_self_digest(value: Mapping[str, Any], *, field: str) -> str:
    body = {key: item for key, item in value.items() if key != "record_sha256"}
    expected = canonical_json_sha256(body)
    observed = _digest(value.get("record_sha256"), field=f"{field}.record_sha256")
    if observed != expected:
        _fail(f"{field} self digest is stale")
    return observed


def _validate_product_payload(product_kind: str, root: Path) -> None:
    if product_kind == BEHAVIOR_DISTRIBUTION:
        from fisheye.group_statistics.validated_behavior_distributions import (
            read_validated_behavior_distributions,
        )

        read_validated_behavior_distributions(root)
    elif product_kind == BEHAVIOR_DISTRIBUTION_REPORT:
        from fisheye.group_statistics.validated_behavior_distribution_report import (
            read_validated_behavior_distribution_report,
        )

        read_validated_behavior_distribution_report(root)
    elif product_kind == GROUP_STATISTICS:
        from fisheye.group_statistics.validated_behavior import (
            read_validated_behavior_group_statistics_sandbox,
        )

        read_validated_behavior_group_statistics_sandbox(root)
    elif product_kind == GROUP_STATISTICS_REPORT:
        from fisheye.group_statistics.validated_behavior_report import (
            read_validated_behavior_statistics_report,
        )

        read_validated_behavior_statistics_report(root)
    else:  # pragma: no cover - closed by _product_type
        _product_type(product_kind)


def _inspect_product(
    product_kind: str,
    product_root: str | Path,
    *,
    validate_payload: bool,
) -> _ProductInspection:
    product_type = _product_type(product_kind)
    root = _canonical_path(product_root, field="product root", directory=True)
    if not root.is_dir():
        _fail(f"Product root does not exist: {root}")
    if validate_payload:
        _validate_product_payload(product_type.kind, root)
    manifest_path = root / "manifest.json"
    manifest = _read_json_object(manifest_path, field="product manifest")
    _validate_self_digest(manifest, field="Product manifest")
    if manifest.get("schema_id") not in product_type.schema_ids:
        _fail(
            f"{product_type.kind} product has unsupported schema: "
            f"{manifest.get('schema_id')!r}"
        )
    version = manifest.get("schema_version")
    if type(version) is not int or version <= 0:
        _fail("Product manifest schema_version must be one positive integer")
    run_id = safe_component(
        manifest.get(product_type.run_id_field),
        label=f"{product_type.kind} run ID",
    )
    source = manifest.get(product_type.source_kind)
    if not isinstance(source, Mapping):
        _fail(f"{product_type.kind} manifest lacks {product_type.source_kind!r}")
    source_product: Mapping[str, str] | None = None
    source_product_root: Path | None = None
    if product_type.source_product_kind is None:
        source_digest = _digest(
            source.get("export_manifest_record_sha256"),
            field="source export manifest digest",
        )
        source_run = safe_component(
            source.get("export_run_id"), label="source export run ID"
        )
    else:
        source_digest = _digest(
            source.get("source_export_manifest_sha256"),
            field="source export manifest digest",
        )
        source_run = None
        parent_type = _product_type(product_type.source_product_kind)
        parent_run_field = (
            "distribution_run_id"
            if parent_type.kind == BEHAVIOR_DISTRIBUTION
            else "statistics_run_id"
        )
        parent_digest_field = (
            "distribution_manifest_sha256"
            if parent_type.kind == BEHAVIOR_DISTRIBUTION
            else "statistics_manifest_sha256"
        )
        source_product = MappingProxyType(
            {
                "product_kind": parent_type.kind,
                "product_run_id": safe_component(
                    source.get(parent_run_field),
                    label=f"source {parent_type.kind} run ID",
                ),
                "manifest_record_sha256": _digest(
                    source.get(parent_digest_field),
                    field=f"source {parent_type.kind} manifest digest",
                ),
            }
        )
        source_product_root = _canonical_path(
            source.get("path"), field="source product root", directory=True
        )
    return _ProductInspection(
        kind=product_type.kind,
        run_id=run_id,
        root=root,
        manifest=MappingProxyType(manifest),
        manifest_path=manifest_path,
        manifest_file_sha256=sha256_file(manifest_path),
        source_export_run_id=source_run,
        source_export_manifest_sha256=source_digest,
        source_product=source_product,
        source_product_root=source_product_root,
    )


def inspect_validated_behavior_product(
    product_kind: str,
    product_root: str | Path,
) -> Mapping[str, object]:
    """Fully validate one supported product and return its exact identity."""

    item = _inspect_product(product_kind, product_root, validate_payload=True)
    return MappingProxyType(
        {
            "product_kind": item.kind,
            "product_run_id": item.run_id,
            "product_root": str(item.root),
            "manifest_path": str(item.manifest_path),
            "manifest_file_sha256": item.manifest_file_sha256,
            "manifest_record_sha256": item.manifest["record_sha256"],
            "source_export_run_id": item.source_export_run_id,
            "source_export_manifest_sha256": (item.source_export_manifest_sha256),
            "source_product": (
                None if item.source_product is None else dict(item.source_product)
            ),
            "source_product_root": (
                None
                if item.source_product_root is None
                else str(item.source_product_root)
            ),
        }
    )


def _source_export_record(
    dataset: ValidatedBehaviorExportDataset,
) -> Mapping[str, object]:
    publication = _canonical_path(
        dataset.root, field="source export publication root", directory=True
    )
    validated_behavior_package_root(publication)
    run_id = safe_component(dataset.export_run_id, label="source export run ID")
    manifest_path = validated_behavior_manifest_path(publication, run_id)
    manifest = _read_json_object(manifest_path, field="source export manifest")
    record_sha256 = _validate_self_digest(manifest, field="Source export manifest")
    if manifest.get("export_run_id") != run_id:
        _fail("Source export manifest carries another export run ID")
    if record_sha256 != dataset.cache_identity or manifest != dict(dataset.manifest):
        _fail("Validated dataset handle differs from its source export manifest")
    validation_receipt = manifest.get("validation_receipt")
    if not isinstance(validation_receipt, Mapping):
        _fail("Source export manifest lacks its validation receipt binding")
    receipt_digest = _digest(
        validation_receipt.get("record_sha256"),
        field="source export validation receipt digest",
    )
    return MappingProxyType(
        {
            "publication_root": str(publication),
            "export_run_id": run_id,
            "manifest_path": str(manifest_path),
            "manifest_size_bytes": manifest_path.stat().st_size,
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_record_sha256": record_sha256,
            "validation_receipt_record_sha256": receipt_digest,
        }
    )


def _entry_from_inspection(
    inspection: _ProductInspection,
    *,
    package_root: Path,
    source_export: Mapping[str, object],
) -> Mapping[str, object]:
    export_run_id = str(source_export["export_run_id"])
    export_digest = str(source_export["manifest_record_sha256"])
    if inspection.source_export_manifest_sha256 != export_digest:
        _fail("Product binds another source export manifest")
    if (
        inspection.source_export_run_id is not None
        and inspection.source_export_run_id != export_run_id
    ):
        _fail("Product binds another source export run ID")
    expected_root = canonical_validated_behavior_product_dir(
        source_export["publication_root"], inspection.kind, inspection.run_id
    )
    if inspection.root != expected_root:
        _fail(
            "Catalog registration accepts only canonical co-located product paths; "
            f"expected {expected_root}, got {inspection.root}"
        )
    root_relative = inspection.root.relative_to(package_root).as_posix()
    manifest_relative = inspection.manifest_path.relative_to(package_root).as_posix()
    body: dict[str, object] = {
        "product_kind": inspection.kind,
        "product_run_id": inspection.run_id,
        "product_root": root_relative,
        "manifest_path": manifest_relative,
        "manifest_size_bytes": inspection.manifest_path.stat().st_size,
        "manifest_file_sha256": inspection.manifest_file_sha256,
        "manifest_record_sha256": inspection.manifest["record_sha256"],
        "product_schema_id": inspection.manifest["schema_id"],
        "product_schema_version": inspection.manifest["schema_version"],
        "product_status": inspection.manifest.get("status"),
        "source_export_run_id": export_run_id,
        "source_export_manifest_sha256": export_digest,
        "source_product": (
            None
            if inspection.source_product is None
            else dict(inspection.source_product)
        ),
        "location_policy": "canonical_package_relative_v1",
    }
    return MappingProxyType({**body, "entry_sha256": canonical_json_sha256(body)})


def _validate_source_export_binding(
    value: object,
    *,
    export_root: Path,
    export_run_id: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_EXPORT_FIELDS:
        _fail("Product catalog source-export binding field set is inexact")
    if value.get("publication_root") != str(export_root):
        _fail("Product catalog binds another publication root")
    if value.get("export_run_id") != export_run_id:
        _fail("Product catalog binds another export run ID")
    manifest_path = validated_behavior_manifest_path(export_root, export_run_id)
    if value.get("manifest_path") != str(manifest_path):
        _fail("Product catalog source manifest path is not canonical")
    manifest = _read_json_object(manifest_path, field="source export manifest")
    record_sha256 = _validate_self_digest(manifest, field="Source export manifest")
    if (
        manifest.get("export_run_id") != export_run_id
        or value.get("manifest_size_bytes") != manifest_path.stat().st_size
        or value.get("manifest_file_sha256") != sha256_file(manifest_path)
        or value.get("manifest_record_sha256") != record_sha256
    ):
        _fail("Product catalog source export manifest binding is stale")
    validation_receipt = manifest.get("validation_receipt")
    if not isinstance(validation_receipt, Mapping) or value.get(
        "validation_receipt_record_sha256"
    ) != validation_receipt.get("record_sha256"):
        _fail("Product catalog source validation-receipt binding is stale")
    _digest(
        value.get("validation_receipt_record_sha256"),
        field="source validation receipt digest",
    )
    return value


def _validate_product_entry(
    value: object,
    *,
    package_root: Path,
    source_export: Mapping[str, Any],
    validate_payload: bool,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PRODUCT_ENTRY_FIELDS:
        _fail("Product catalog entry field set is inexact")
    body = {key: item for key, item in value.items() if key != "entry_sha256"}
    if value.get("entry_sha256") != canonical_json_sha256(body):
        _fail("Product catalog entry digest is stale")
    product_type = _product_type(value.get("product_kind"))
    run_id = safe_component(
        value.get("product_run_id"), label=f"{product_type.kind} run ID"
    )
    if value.get("location_policy") != "canonical_package_relative_v1":
        _fail("Product catalog entry location policy is unsupported")
    expected_root = canonical_validated_behavior_product_dir(
        source_export["publication_root"], product_type.kind, run_id
    )
    root = _safe_relative_path(
        package_root, value.get("product_root"), field="product_root"
    )
    manifest_path = _safe_relative_path(
        package_root, value.get("manifest_path"), field="manifest_path"
    )
    if root != expected_root or manifest_path != root / "manifest.json":
        _fail("Product catalog entry does not use its canonical co-located path")
    manifest = _read_json_object(manifest_path, field="product manifest")
    manifest_digest = _validate_self_digest(manifest, field="Product manifest")
    if (
        value.get("manifest_size_bytes") != manifest_path.stat().st_size
        or value.get("manifest_file_sha256") != sha256_file(manifest_path)
        or value.get("manifest_record_sha256") != manifest_digest
        or value.get("product_schema_id") != manifest.get("schema_id")
        or value.get("product_schema_version") != manifest.get("schema_version")
        or value.get("product_status") != manifest.get("status")
        or manifest.get(product_type.run_id_field) != run_id
    ):
        _fail("Product catalog manifest binding is stale")
    if manifest.get("schema_id") not in product_type.schema_ids:
        _fail("Product catalog entry references an unsupported product schema")
    if (
        value.get("source_export_run_id") != source_export["export_run_id"]
        or value.get("source_export_manifest_sha256")
        != source_export["manifest_record_sha256"]
    ):
        _fail("Product catalog entry binds another source export")
    inspection = _inspect_product(
        product_type.kind, root, validate_payload=validate_payload
    )
    if inspection.source_export_manifest_sha256 != source_export[
        "manifest_record_sha256"
    ] or (
        inspection.source_export_run_id is not None
        and inspection.source_export_run_id != source_export["export_run_id"]
    ):
        _fail("Product payload binds another source export")
    expected_parent = (
        None if inspection.source_product is None else dict(inspection.source_product)
    )
    if value.get("source_product") != expected_parent:
        _fail("Product catalog parent-product binding is stale")
    if expected_parent is not None:
        if set(expected_parent) != _SOURCE_PRODUCT_FIELDS:
            _fail("Product catalog parent-product field set is inexact")
        _product_type(expected_parent["product_kind"])
        safe_component(expected_parent["product_run_id"], label="source product run ID")
        _digest(
            expected_parent["manifest_record_sha256"],
            field="source product manifest digest",
        )
    return value


def _validate_catalog_record(
    value: Mapping[str, Any],
    *,
    package_root: Path,
    export_root: Path,
    export_run_id: str,
    validate_products: bool,
    validate_history: bool,
    visited: set[str] | None = None,
) -> None:
    if set(value) != _CATALOG_FIELDS:
        _fail("Product catalog field set is inexact")
    if (
        value.get("schema_id") != CATALOG_SCHEMA_ID
        or value.get("schema_version") != CATALOG_SCHEMA_VERSION
        or value.get("method_id") != CATALOG_METHOD_ID
        or value.get("status") != CATALOG_STATUS
        or value.get("safety") != dict(CATALOG_SAFETY)
        or value.get("discovery_semantics") != dict(_DISCOVERY_SEMANTICS)
    ):
        _fail("Product catalog schema, method, status, or safety is unsupported")
    record_sha256 = _validate_self_digest(value, field="Product catalog")
    seen = set() if visited is None else visited
    if record_sha256 in seen:
        _fail("Product catalog history contains a cycle")
    seen.add(record_sha256)
    generation_id = safe_component(
        value.get("catalog_generation_id"), label="catalog generation ID"
    )
    expected_generation = _catalog_generation_relative_path(
        export_run_id, generation_id
    ).as_posix()
    if value.get("generation_path") != expected_generation:
        _fail("Product catalog generation path is not canonical")
    source_export = _validate_source_export_binding(
        value.get("source_export"),
        export_root=export_root,
        export_run_id=export_run_id,
    )
    products = value.get("products")
    if not isinstance(products, list):
        _fail("Product catalog products must be one array")
    validated = [
        _validate_product_entry(
            item,
            package_root=package_root,
            source_export=source_export,
            validate_payload=validate_products,
        )
        for item in products
    ]
    keys = [
        (str(item["product_kind"]), str(item["product_run_id"])) for item in validated
    ]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _fail("Product catalog entries must be uniquely key-sorted")
    if value.get("products_sha256") != canonical_json_sha256(products):
        _fail("Product catalog product-roster digest is stale")
    by_key = {key: item for key, item in zip(keys, validated, strict=True)}
    for item in validated:
        parent = item.get("source_product")
        if parent is None:
            continue
        parent_key = (str(parent["product_kind"]), str(parent["product_run_id"]))
        selected_parent = by_key.get(parent_key)
        if (
            selected_parent is None
            or selected_parent["manifest_record_sha256"]
            != parent["manifest_record_sha256"]
        ):
            _fail("Product catalog report lacks its exact parent product")
        child_root = _safe_relative_path(
            package_root, item["product_root"], field="product_root"
        )
        child = _inspect_product(
            str(item["product_kind"]), child_root, validate_payload=False
        )
        selected_parent_root = _safe_relative_path(
            package_root,
            selected_parent["product_root"],
            field="source product root",
        )
        if child.source_product_root != selected_parent_root:
            _fail("Product report source path is not its co-located catalog parent")
    _timestamp(value.get("created_at_utc"), field="created_at_utc")

    previous = value.get("previous_catalog")
    if previous is None:
        return
    if not isinstance(previous, Mapping) or set(previous) != _PREVIOUS_CATALOG_FIELDS:
        _fail("Product catalog previous-generation binding is malformed")
    previous_generation = safe_component(
        previous.get("catalog_generation_id"),
        label="previous catalog generation ID",
    )
    expected_previous_path = _catalog_generation_relative_path(
        export_run_id, previous_generation
    ).as_posix()
    if previous.get("generation_path") != expected_previous_path:
        _fail("Previous product-catalog generation path is not canonical")
    previous_root = _safe_relative_path(
        package_root,
        previous.get("generation_path"),
        field="previous_catalog.generation_path",
    )
    if {path.name for path in previous_root.iterdir()} != {"catalog.json"}:
        _fail("Previous product-catalog generation inventory is not exact")
    previous_path = previous_root / "catalog.json"
    previous_record = _read_json_object(
        previous_path, field="previous product catalog generation"
    )
    if (
        previous.get("record_sha256") != previous_record.get("record_sha256")
        or previous.get("products_sha256") != previous_record.get("products_sha256")
        or previous.get("product_count") != len(previous_record.get("products", []))
    ):
        _fail("Previous product-catalog generation binding is stale")
    previous_products = previous_record.get("products")
    if not isinstance(previous_products, list):
        _fail("Previous product catalog has no product roster")
    current_by_key = {
        (str(item["product_kind"]), str(item["product_run_id"])): item
        for item in products
    }
    for previous_item in previous_products:
        previous_key = (
            str(previous_item["product_kind"]),
            str(previous_item["product_run_id"]),
        )
        if current_by_key.get(previous_key) != previous_item:
            _fail("Product catalog update is not append-only")
    if validate_history:
        _validate_catalog_record(
            previous_record,
            package_root=package_root,
            export_root=export_root,
            export_run_id=export_run_id,
            validate_products=False,
            validate_history=True,
            visited=seen,
        )


def read_validated_behavior_product_catalog(
    export_root: str | Path,
    export_run_id: str,
    *,
    validate_products: bool = False,
) -> Mapping[str, object]:
    """Read one selected catalog generation without globbing product paths."""

    publication = _canonical_path(
        export_root, field="validated-behavior publication root", directory=True
    )
    package = validated_behavior_package_root(publication)
    run_id = safe_component(export_run_id, label="export run ID")
    selected_path = validated_behavior_product_catalog_manifest_path(
        publication, run_id
    )
    selected = _read_json_object(selected_path, field="product catalog manifest")
    _validate_catalog_record(
        selected,
        package_root=package,
        export_root=publication,
        export_run_id=run_id,
        validate_products=validate_products,
        validate_history=True,
    )
    generation_root = _safe_relative_path(
        package,
        selected["generation_path"],
        field="catalog generation path",
    )
    if {path.name for path in generation_root.iterdir()} != {"catalog.json"}:
        _fail("Selected product-catalog generation inventory is not exact")
    generation_record = _read_json_object(
        generation_root / "catalog.json",
        field="selected product catalog generation",
    )
    if generation_record != selected:
        _fail("Selected product-catalog manifest differs from its generation")
    return MappingProxyType(
        {
            **selected,
            "catalog_manifest_path": str(selected_path),
            "catalog_generation_root": str(generation_root),
        }
    )


def _handle_from_entry(
    entry: Mapping[str, Any],
    *,
    package_root: Path,
    catalog_record_sha256: str,
) -> ValidatedBehaviorProductHandle:
    root = _safe_relative_path(
        package_root, entry["product_root"], field="product_root"
    )
    manifest_path = _safe_relative_path(
        package_root, entry["manifest_path"], field="manifest_path"
    )
    return ValidatedBehaviorProductHandle(
        product_kind=str(entry["product_kind"]),
        product_run_id=str(entry["product_run_id"]),
        root=root,
        manifest_path=manifest_path,
        manifest_record_sha256=str(entry["manifest_record_sha256"]),
        catalog_record_sha256=catalog_record_sha256,
        entry=MappingProxyType(dict(entry)),
    )


def list_validated_behavior_products(
    export_root: str | Path,
    export_run_id: str,
    *,
    product_kind: str | None = None,
) -> tuple[ValidatedBehaviorProductHandle, ...]:
    """List only manifest-selected products, optionally restricted by kind."""

    kind = None if product_kind is None else _product_type(product_kind).kind
    catalog = read_validated_behavior_product_catalog(export_root, export_run_id)
    package = validated_behavior_package_root(export_root)
    return tuple(
        _handle_from_entry(
            entry,
            package_root=package,
            catalog_record_sha256=str(catalog["record_sha256"]),
        )
        for entry in catalog["products"]
        if kind is None or entry["product_kind"] == kind
    )


def resolve_validated_behavior_product(
    export_root: str | Path,
    export_run_id: str,
    *,
    product_kind: str,
    product_run_id: str | None = None,
) -> ValidatedBehaviorProductHandle:
    """Resolve an exact product or the sole compatible product; fail on ambiguity."""

    kind = _product_type(product_kind).kind
    run_id = (
        None
        if product_run_id is None
        else safe_component(product_run_id, label=f"{kind} run ID")
    )
    candidates = tuple(
        item
        for item in list_validated_behavior_products(
            export_root, export_run_id, product_kind=kind
        )
        if run_id is None or item.product_run_id == run_id
    )
    if not candidates:
        suffix = "" if run_id is None else f" with run ID {run_id!r}"
        _fail(f"No cataloged {kind!r} product{suffix}")
    if len(candidates) != 1:
        ids = [item.product_run_id for item in candidates]
        _fail(
            f"Multiple cataloged {kind!r} products are compatible; choose one "
            f"exact product run ID from {ids}"
        )
    return candidates[0]


def _previous_catalog_binding(
    catalog: Mapping[str, object],
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "catalog_generation_id": catalog["catalog_generation_id"],
            "generation_path": catalog["generation_path"],
            "record_sha256": catalog["record_sha256"],
            "products_sha256": catalog["products_sha256"],
            "product_count": len(catalog["products"]),
        }
    )


def register_validated_behavior_product(
    dataset: ValidatedBehaviorExportDataset,
    *,
    product_kind: str,
    product_root: str | Path,
    generation_id: str | None = None,
    created_at_utc: str | None = None,
) -> Mapping[str, object]:
    """Append one fully validated co-located product to the export catalog."""

    source_export = _source_export_record(dataset)
    publication = Path(str(source_export["publication_root"]))
    package = validated_behavior_package_root(publication)
    inspection = _inspect_product(product_kind, product_root, validate_payload=True)
    entry = _entry_from_inspection(
        inspection, package_root=package, source_export=source_export
    )
    selected_path = validated_behavior_product_catalog_manifest_path(
        publication, dataset.export_run_id
    )
    baseline = manifest_identity(selected_path)
    if selected_path.exists():
        current = read_validated_behavior_product_catalog(
            publication, dataset.export_run_id
        )
        if manifest_identity(selected_path) != baseline:
            _fail("Product catalog changed during registration preflight")
        current_products = [dict(item) for item in current["products"]]
        previous_catalog: Mapping[str, object] | None = _previous_catalog_binding(
            current
        )
    else:
        retained_generations = _catalog_generations_root(package, dataset.export_run_id)
        if retained_generations.exists() and any(retained_generations.iterdir()):
            _fail(
                "Product catalog selector is missing while immutable generations "
                "remain; refusing to start a disconnected history"
            )
        current = None
        current_products = []
        previous_catalog = None

    key = (entry["product_kind"], entry["product_run_id"])
    existing = next(
        (
            item
            for item in current_products
            if (item["product_kind"], item["product_run_id"]) == key
        ),
        None,
    )
    if existing is not None:
        if existing != dict(entry):
            _fail("Product catalog key already binds different immutable evidence")
        assert current is not None
        return MappingProxyType({**current, "reused": True})

    products = sorted(
        [*current_products, dict(entry)],
        key=lambda item: (item["product_kind"], item["product_run_id"]),
    )
    by_key = {(item["product_kind"], item["product_run_id"]): item for item in products}
    parent = entry["source_product"]
    if parent is not None:
        selected_parent = by_key.get((parent["product_kind"], parent["product_run_id"]))
        if (
            selected_parent is None
            or selected_parent["manifest_record_sha256"]
            != parent["manifest_record_sha256"]
        ):
            _fail("Register the report's exact source product before the report")

    generation = safe_component(
        generation_id or uuid.uuid4().hex, label="catalog generation ID"
    )
    generation_relative = _catalog_generation_relative_path(
        dataset.export_run_id, generation
    )
    final_generation = package / generation_relative
    stage = _catalog_staging_root(package, dataset.export_run_id, generation)
    if stage.exists() or final_generation.exists():
        raise FileExistsError("Product-catalog staging or generation already exists")
    created = _timestamp(
        created_at_utc or datetime.now(timezone.utc).isoformat(),
        field="created_at_utc",
    )
    body: dict[str, object] = {
        "schema_id": CATALOG_SCHEMA_ID,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "method_id": CATALOG_METHOD_ID,
        "status": CATALOG_STATUS,
        "catalog_generation_id": generation,
        "generation_path": generation_relative.as_posix(),
        "source_export": dict(source_export),
        "products": products,
        "products_sha256": canonical_json_sha256(products),
        "previous_catalog": (
            None if previous_catalog is None else dict(previous_catalog)
        ),
        "discovery_semantics": dict(_DISCOVERY_SEMANTICS),
        "created_at_utc": created,
        "safety": dict(CATALOG_SAFETY),
    }
    catalog = {**body, "record_sha256": canonical_json_sha256(body)}
    stage.mkdir(parents=True, exist_ok=False)
    try:
        write_json_atomic(stage / "catalog.json", catalog, overwrite=False)

        def validate_staging() -> None:
            staged = _read_json_object(
                stage / "catalog.json", field="staged product catalog"
            )
            if staged != catalog or {path.name for path in stage.iterdir()} != {
                "catalog.json"
            }:
                _fail("Staged product-catalog generation changed before commit")
            _validate_catalog_record(
                staged,
                package_root=package,
                export_root=publication,
                export_run_id=dataset.export_run_id,
                validate_products=True,
                validate_history=True,
            )

        commit_validated_immutable_generation(
            package,
            stage,
            final_generation,
            selected_path,
            catalog,
            baseline_manifest_identity=baseline,
            lock_directory=_product_namespace(package) / "catalog" / ".locks",
            validate_staging=validate_staging,
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    selected = read_validated_behavior_product_catalog(
        publication, dataset.export_run_id, validate_products=True
    )
    return MappingProxyType({**selected, "reused": False})


def _assert_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        _fail(f"Product source is a symbolic-link alias: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail(f"Product source contains a symbolic-link alias: {path}")


def adopt_validated_behavior_product(
    dataset: ValidatedBehaviorExportDataset,
    *,
    product_kind: str,
    source_product_root: str | Path,
    catalog_generation_id: str | None = None,
    created_at_utc: str | None = None,
) -> Mapping[str, object]:
    """Copy exact legacy bytes into the canonical package and catalog them.

    The source is fully validated before copying and the copied destination is
    fully validated again before it becomes visible.  No source file is moved
    or modified, and no scientific data are recomputed.
    """

    source = _inspect_product(product_kind, source_product_root, validate_payload=True)
    source_export = _source_export_record(dataset)
    if source.source_export_manifest_sha256 != source_export[
        "manifest_record_sha256"
    ] or (
        source.source_export_run_id is not None
        and source.source_export_run_id != dataset.export_run_id
    ):
        _fail("Product source binds another validated-behavior export")
    package = validated_behavior_package_root(dataset.root)
    selected_catalog = validated_behavior_product_catalog_manifest_path(
        dataset.root, dataset.export_run_id
    )
    if selected_catalog.exists():
        read_validated_behavior_product_catalog(dataset.root, dataset.export_run_id)
    else:
        retained_generations = _catalog_generations_root(package, dataset.export_run_id)
        if retained_generations.exists() and any(retained_generations.iterdir()):
            _fail(
                "Product catalog selector is missing while immutable generations "
                "remain; refusing adoption into a disconnected history"
            )
    if source.source_product is not None:
        parent = resolve_validated_behavior_product(
            dataset.root,
            dataset.export_run_id,
            product_kind=str(source.source_product["product_kind"]),
            product_run_id=str(source.source_product["product_run_id"]),
        )
        if (
            parent.manifest_record_sha256
            != source.source_product["manifest_record_sha256"]
            or parent.root != source.source_product_root
        ):
            _fail("Product report does not bind its exact co-located catalog parent")
    target = canonical_validated_behavior_product_dir(
        dataset.root, source.kind, source.run_id
    )
    copied = False
    if source.root != target:
        _assert_no_symlinks(source.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = _inspect_product(source.kind, target, validate_payload=True)
            if existing.manifest["record_sha256"] != source.manifest["record_sha256"]:
                _fail("Canonical product path already contains different evidence")
        else:
            temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copytree(source.root, temporary, symlinks=False)
                copied_product = _inspect_product(
                    source.kind, temporary, validate_payload=True
                )
                if (
                    copied_product.manifest["record_sha256"]
                    != source.manifest["record_sha256"]
                    or copied_product.manifest_file_sha256
                    != source.manifest_file_sha256
                ):
                    _fail("Copied product differs from its validated source")
                os.replace(temporary, target)
                copied = True
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    catalog = register_validated_behavior_product(
        dataset,
        product_kind=source.kind,
        product_root=target,
        generation_id=catalog_generation_id,
        created_at_utc=created_at_utc,
    )
    return MappingProxyType(
        {
            "product_kind": source.kind,
            "product_run_id": source.run_id,
            "source_product_root": str(source.root),
            "product_root": str(target),
            "product_manifest_record_sha256": source.manifest["record_sha256"],
            "copied": copied,
            "catalog_reused": catalog["reused"],
            "catalog_generation_id": catalog["catalog_generation_id"],
            "catalog_record_sha256": catalog["record_sha256"],
            "catalog_manifest_path": catalog["catalog_manifest_path"],
        }
    )


__all__ = [
    "BEHAVIOR_DISTRIBUTION",
    "BEHAVIOR_DISTRIBUTION_REPORT",
    "CATALOG_METHOD_ID",
    "CATALOG_SCHEMA_ID",
    "CATALOG_SCHEMA_VERSION",
    "CATALOG_STATUS",
    "GROUP_STATISTICS",
    "GROUP_STATISTICS_REPORT",
    "ValidatedBehaviorProductCatalogError",
    "ValidatedBehaviorProductHandle",
    "adopt_validated_behavior_product",
    "canonical_validated_behavior_product_dir",
    "inspect_validated_behavior_product",
    "list_validated_behavior_products",
    "read_validated_behavior_product_catalog",
    "register_validated_behavior_product",
    "resolve_validated_behavior_product",
    "validated_behavior_package_root",
    "validated_behavior_product_catalog_manifest_path",
    "validated_behavior_product_kinds",
]
