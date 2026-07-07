"""Backfill detect-review authoritative pointers from legacy review pointers.

The tool is intentionally store-list driven: it never discovers real stores or
registry rows on its own. Dry-run is the default; ``--execute`` only applies to
stores explicitly named on the command line or in ``--store-list``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.diagnostics.detect_review_pointer_census import (
    LEGACY_DETECT_REVIEW_AUTHORITY_ATTR,
    ParentCensus,
    StoreCensus,
    scan_store,
)
from fisheye.shared.zarr_run_completion import AUTHORITATIVE_RUN_ATTR


MODULE_NAME = "fisheye.utils.backfill_detect_review_authoritative_run"
REPORT_SCHEMA_ID = "palette.detect_review_authoritative_backfill.v1"


@dataclass(frozen=True)
class ParentMutation:
    parent_name: str
    legacy_attr: str
    legacy_value: str
    set_attr: str
    old_value: str | None
    new_value: str


@dataclass(frozen=True)
class StoreBackfillPlan:
    zarr_path: str
    bucket: str
    reason: str
    mutations: tuple[ParentMutation, ...]
    skipped: bool
    skip_reason: str | None = None


@dataclass(frozen=True)
class StoreBackfillResult:
    plan: StoreBackfillPlan
    executed: bool
    applied_mutations: int
    error: str | None = None


@dataclass(frozen=True)
class BackfillReport:
    schema_id: str
    dry_run: bool
    stores: tuple[StoreBackfillResult, ...]

    @property
    def planned_mutation_count(self) -> int:
        return sum(len(result.plan.mutations) for result in self.stores)

    @property
    def applied_mutation_count(self) -> int:
        return sum(result.applied_mutations for result in self.stores)


def _legacy_value(parent: ParentCensus) -> str | None:
    value = parent.attrs.get(LEGACY_DETECT_REVIEW_AUTHORITY_ATTR)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def plan_store_backfill(zarr_path: Path) -> StoreBackfillPlan:
    """Return the dry-run plan for one explicit store path."""

    store = scan_store(zarr_path, ["explicit"])
    return plan_scanned_store_backfill(store)


def plan_scanned_store_backfill(store: StoreCensus) -> StoreBackfillPlan:
    if store.bucket != "BACKFILLABLE":
        return StoreBackfillPlan(
            zarr_path=store.zarr_path,
            bucket=store.bucket,
            reason=store.reason,
            mutations=(),
            skipped=True,
            skip_reason=f"store bucket is {store.bucket}; only BACKFILLABLE stores are mutated",
        )

    mutations: list[ParentMutation] = []
    for parent in store.parents:
        if not parent.parent_exists or parent.bucket != "BACKFILLABLE":
            continue
        legacy = _legacy_value(parent)
        if legacy is None:
            continue
        mutations.append(
            ParentMutation(
                parent_name=parent.parent_name,
                legacy_attr=LEGACY_DETECT_REVIEW_AUTHORITY_ATTR,
                legacy_value=legacy,
                set_attr=AUTHORITATIVE_RUN_ATTR,
                old_value=parent.attrs.get(AUTHORITATIVE_RUN_ATTR),
                new_value=legacy,
            )
        )

    if not mutations:
        return StoreBackfillPlan(
            zarr_path=store.zarr_path,
            bucket=store.bucket,
            reason=store.reason,
            mutations=(),
            skipped=True,
            skip_reason="store classified BACKFILLABLE but no parent mutation was identified",
        )

    return StoreBackfillPlan(
        zarr_path=store.zarr_path,
        bucket=store.bucket,
        reason=store.reason,
        mutations=tuple(mutations),
        skipped=False,
    )


def _load_group_metadata(group_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    zarr_json = group_path / "zarr.json"
    if zarr_json.exists():
        payload = json.loads(zarr_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"{zarr_json} did not contain a JSON object")
        attrs = payload.get("attributes")
        if attrs is None:
            attrs = {}
            payload["attributes"] = attrs
        if not isinstance(attrs, dict):
            raise RuntimeError(f"{zarr_json} attributes were not a JSON object")
        return zarr_json, payload, attrs, "zarr.json"

    zattrs = group_path / ".zattrs"
    if zattrs.exists():
        payload = json.loads(zattrs.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"{zattrs} did not contain a JSON object")
        return zattrs, payload, payload, ".zattrs"

    raise RuntimeError(f"{group_path} has no zarr.json or .zattrs metadata file")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _apply_parent_mutation(zarr_path: Path, mutation: ParentMutation) -> None:
    metadata_path, payload, attrs, _metadata_format = _load_group_metadata(zarr_path / mutation.parent_name)
    current = attrs.get(mutation.set_attr)
    if current not in (None, mutation.old_value):
        raise RuntimeError(
            f"{zarr_path}/{mutation.parent_name} {mutation.set_attr} changed during planning: "
            f"expected {mutation.old_value!r}, found {current!r}"
        )
    attrs[mutation.set_attr] = mutation.new_value
    _write_json_atomic(metadata_path, payload)


def backfill_stores(store_paths: Sequence[Path], *, execute: bool = False) -> BackfillReport:
    results: list[StoreBackfillResult] = []
    for zarr_path in store_paths:
        plan = plan_store_backfill(zarr_path)
        if not execute or plan.skipped:
            results.append(
                StoreBackfillResult(
                    plan=plan,
                    executed=False,
                    applied_mutations=0,
                )
            )
            continue

        applied = 0
        try:
            for mutation in plan.mutations:
                _apply_parent_mutation(Path(plan.zarr_path), mutation)
                applied += 1
        except Exception as exc:
            results.append(
                StoreBackfillResult(
                    plan=plan,
                    executed=True,
                    applied_mutations=applied,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            StoreBackfillResult(
                plan=plan,
                executed=True,
                applied_mutations=applied,
            )
        )

    return BackfillReport(
        schema_id=REPORT_SCHEMA_ID,
        dry_run=not execute,
        stores=tuple(results),
    )


def _read_store_list(path: Path) -> list[Path]:
    stores: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stores.append(Path(stripped).expanduser())
    return stores


def _collect_store_paths(args: argparse.Namespace) -> list[Path]:
    stores = [Path(value).expanduser() for value in args.stores]
    for store_list in args.store_list or ():
        stores.extend(_read_store_list(Path(store_list).expanduser()))
    return stores


def _json_ready(report: BackfillReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["summary"] = {
        "store_count": len(report.stores),
        "planned_mutations": report.planned_mutation_count,
        "applied_mutations": report.applied_mutation_count,
        "error_count": sum(1 for result in report.stores if result.error),
        "skipped_count": sum(1 for result in report.stores if result.plan.skipped),
    }
    return payload


def render_text_report(report: BackfillReport) -> str:
    mode = "DRY-RUN" if report.dry_run else "EXECUTE"
    lines = [
        f"{mode} {MODULE_NAME}",
        (
            "summary: "
            f"stores={len(report.stores)} "
            f"planned_mutations={report.planned_mutation_count} "
            f"applied_mutations={report.applied_mutation_count}"
        ),
    ]
    for result in report.stores:
        plan = result.plan
        lines.append(f"{plan.zarr_path}: {plan.bucket} - {plan.reason}")
        if plan.skipped:
            lines.append(f"  skip: {plan.skip_reason}")
        for mutation in plan.mutations:
            verb = "would set" if report.dry_run else "set"
            lines.append(
                f"  {verb} {mutation.parent_name}.{mutation.set_attr}: "
                f"{mutation.old_value!r} -> {mutation.new_value!r} "
                f"(from {mutation.legacy_attr})"
            )
        if result.error:
            lines.append(f"  error: {result.error}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill authoritative_run from detect_review_status_latest for explicitly "
            "listed BACKFILLABLE refined-detect stores. Dry-run is the default."
        )
    )
    parser.add_argument("stores", nargs="*", help="Explicit zarr store paths to inspect/backfill.")
    parser.add_argument(
        "--store-list",
        action="append",
        default=[],
        help="Text file containing one explicit zarr store path per line; comments and blanks ignored.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write authoritative_run for BACKFILLABLE stores. Requires at least one explicit store.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for clarity; dry-run is already the default unless --execute is supplied.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stores = _collect_store_paths(args)
    if args.execute and args.dry_run:
        parser.error("--dry-run and --execute are mutually exclusive")
    if args.execute and not stores:
        parser.error("--execute requires at least one explicit store path or --store-list entry")

    report = backfill_stores(stores, execute=bool(args.execute))
    if args.json:
        print(json.dumps(_json_ready(report), indent=2, sort_keys=True))
    else:
        print(render_text_report(report), end="")

    if any(result.error for result in report.stores):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
