"""Manifest-selected lazy access to validated-behavior cohort publications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from .validated_behavior_cohort import (
    read_validated_behavior_export_manifest,
    selected_table_parts,
    validated_behavior_manifest_path,
)
from .validated_behavior_contracts import ValidatedBehaviorTableSpec


@dataclass(frozen=True)
class ValidatedBehaviorTable:
    """One exact logical table backed only by manifest-selected Parquet parts."""

    dataset: "ValidatedBehaviorExportDataset"
    name: str
    spec: ValidatedBehaviorTableSpec
    part_paths: tuple[Path, ...]

    @property
    def contract(self) -> Mapping[str, object]:
        return self.spec.contract.to_dict()

    @property
    def grain(self) -> str:
        return self.spec.grain

    @property
    def primary_key(self) -> tuple[str, ...]:
        return self.spec.contract.primary_key

    @property
    def foreign_keys(
        self,
    ) -> tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...]:
        return self.spec.foreign_keys

    @property
    def capability_policy(self) -> str:
        return self.spec.capability_policy

    def scan(
        self,
        *,
        columns: Sequence[str] | None = None,
        predicate: Any | None = None,
    ) -> Any:
        """Return a true Polars LazyFrame with projection/predicate pushdown."""

        import polars as pl

        known = {item.name for item in self.spec.contract.fields}
        selected = None if columns is None else tuple(columns)
        if selected is not None:
            unknown = sorted(set(selected) - known)
            if unknown:
                raise KeyError(f"{self.name}: unknown columns: {unknown}")
            if len(set(selected)) != len(selected):
                raise ValueError(f"{self.name}: selected columns must be unique")
        lazy = pl.scan_parquet([str(path) for path in self.part_paths])
        if predicate is not None:
            if not isinstance(predicate, pl.Expr):
                raise TypeError("predicate must be one Polars expression")
            lazy = lazy.filter(predicate)
        if selected is not None:
            lazy = lazy.select(*selected)
        return lazy

    def collect_bounded(
        self,
        *,
        max_rows: int,
        columns: Sequence[str] | None = None,
        predicate: Any | None = None,
    ) -> Any:
        """Collect at most ``max_rows`` after pushdown, never an implicit whole table."""

        if type(max_rows) is not int or max_rows <= 0:
            raise ValueError("max_rows must be one positive integer")
        return self.scan(columns=columns, predicate=predicate).limit(max_rows).collect()

    def iter_batches(
        self,
        *,
        columns: Sequence[str] | None = None,
        predicate: Any | None = None,
        batch_size: int = 65536,
        max_rows: int | None = None,
    ) -> Iterator[Any]:
        """Yield bounded Arrow record batches from the selected part roster."""

        import pyarrow.dataset as ds

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be one positive integer")
        if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
            raise ValueError("max_rows must be null or one positive integer")
        known = {item.name for item in self.spec.contract.fields}
        selected = None if columns is None else tuple(columns)
        if selected is not None:
            unknown = sorted(set(selected) - known)
            if unknown:
                raise KeyError(f"{self.name}: unknown columns: {unknown}")
        dataset = ds.dataset([str(path) for path in self.part_paths], format="parquet")
        scanner = dataset.scanner(
            columns=None if selected is None else list(selected),
            filter=predicate,
            batch_size=batch_size,
        )
        remaining = max_rows
        for batch in scanner.to_batches():
            if remaining is None:
                yield batch
                continue
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                yield batch.slice(0, remaining)
                break
            yield batch
            remaining -= batch.num_rows

    def query_identity(
        self,
        *,
        columns: Sequence[str] | None = None,
        predicate_description: str | None = None,
    ) -> dict[str, object]:
        """Return the immutable provenance prefix for a consumer query receipt."""

        return {
            "export_run_id": self.dataset.export_run_id,
            "export_manifest_record_sha256": self.dataset.manifest["record_sha256"],
            "export_plan_sha256": self.dataset.manifest["export_plan"]["plan_sha256"],
            "table_name": self.name,
            "table_contract_sha256": self.spec.contract.payload_sha256,
            "grain": self.spec.grain,
            "selected_columns": (
                [item.name for item in self.spec.contract.fields]
                if columns is None
                else list(columns)
            ),
            "predicate_description": predicate_description,
            "analysis_unit_policy_sha256": self.dataset.manifest[
                "analysis_unit_policy"
            ]["sha256"],
            "capability_policy": self.spec.capability_policy,
        }


@dataclass(frozen=True)
class ValidatedBehaviorExportDataset:
    """Read-only handle over one exact validated-behavior export generation."""

    root: Path
    export_run_id: str
    manifest: Mapping[str, Any]
    membership: Mapping[str, Any]
    bundle_set: Mapping[str, Any]
    table_specs: Mapping[str, ValidatedBehaviorTableSpec]
    validation_mode: str

    @classmethod
    def open(
        cls,
        root: str | Path,
        export_run_id: str,
        *,
        validate: bool = True,
        full_part_hashes: bool = False,
        table_specs: Mapping[str, ValidatedBehaviorTableSpec] | None = None,
    ) -> "ValidatedBehaviorExportDataset":
        """Open one selected generation; never resolve ``latest`` or glob parts."""

        if not validate:
            raise ValueError(
                "ValidatedBehaviorExportDataset does not expose an unvalidated open"
            )
        mode = "full" if full_part_hashes else "receipt"
        resolved = Path(root).expanduser().resolve()
        if table_specs is None:
            from .validated_behavior_profiles import (
                profile_id_from_record,
                resolve_validated_behavior_profile,
            )

            manifest_path = validated_behavior_manifest_path(resolved, export_run_id)
            profile = resolve_validated_behavior_profile(
                profile_id_from_record(manifest_path, record_kind="export manifest")
            )
            selected_specs = profile.table_specs
        else:
            selected_specs = table_specs
        manifest, membership, bundle_set = read_validated_behavior_export_manifest(
            resolved,
            export_run_id,
            table_specs=selected_specs,
            validate_parts=mode,
        )
        return cls(
            root=resolved,
            export_run_id=export_run_id,
            manifest=manifest,
            membership=membership,
            bundle_set=bundle_set,
            table_specs=MappingProxyType(dict(selected_specs)),
            validation_mode=mode,
        )

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["table_names"])

    @property
    def cache_identity(self) -> str:
        """Cache identity changes whenever the exact selected manifest changes."""

        return str(self.manifest["record_sha256"])

    def table(self, name: str) -> ValidatedBehaviorTable:
        if name not in self.table_specs or name not in self.table_names:
            raise KeyError(f"Unknown validated-behavior table: {name}")
        return ValidatedBehaviorTable(
            dataset=self,
            name=name,
            spec=self.table_specs[name],
            part_paths=selected_table_parts(self.root, self.manifest, name),
        )


__all__ = ["ValidatedBehaviorExportDataset", "ValidatedBehaviorTable"]
