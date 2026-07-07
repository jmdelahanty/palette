#!/usr/bin/env python3
"""Read-only census for legacy detect-review run pointers."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.registry.db import RegistryPaths
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs
from fisheye.shared.zarr_run_completion import (
    AUTHORITATIVE_RUN_ATTR,
    is_run_complete_in_parent,
)


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")
REFINED_DETECT_PARENT_NAMES = ("refined_detect_runs", "refined_runs")
LEGACY_DETECT_REVIEW_AUTHORITY_ATTR = "detect_review_status_latest"
BATCH_POINTER_ORDER = ("latest", "latest_materialized", LEGACY_DETECT_REVIEW_AUTHORITY_ATTR)
TASK_GENERATION_POINTER_ORDER = (LEGACY_DETECT_REVIEW_AUTHORITY_ATTR, "latest")
POINTER_ATTRS = (
    AUTHORITATIVE_RUN_ATTR,
    LEGACY_DETECT_REVIEW_AUTHORITY_ATTR,
    "latest",
    "latest_materialized",
)


@dataclass(frozen=True)
class NamedRunCensus:
    attr: str
    value: str | None
    exists: bool
    complete: bool
    error: str | None = None


@dataclass(frozen=True)
class ParentCensus:
    zarr_path: str
    parent_name: str
    parent_exists: bool
    attrs: dict[str, str | None]
    named_runs: tuple[NamedRunCensus, ...]
    legacy_fallback_would_fire: bool
    batch_winner_with_legacy: str | None
    batch_winner_without_legacy: str | None
    batch_winner_would_change: bool
    task_generation_winner_with_legacy: str | None
    task_generation_winner_without_legacy: str | None
    task_generation_winner_would_change: bool
    bucket: str
    reason: str
    error: str | None = None


@dataclass(frozen=True)
class StoreCensus:
    zarr_path: str
    source_modalities: tuple[str, ...]
    parents: tuple[ParentCensus, ...]
    bucket: str
    reason: str
    unreadable_error: str | None = None


@dataclass(frozen=True)
class EnumerationCensus:
    filesystem_count: int
    registry_count: int
    both_count: int
    filesystem_only: tuple[str, ...]
    registry_only: tuple[str, ...]
    registry_error: str | None = None


@dataclass(frozen=True)
class DetectReviewPointerCensus:
    recordings_root: str
    registry_path: str
    enumeration: EnumerationCensus
    stores: tuple[StoreCensus, ...]

    @property
    def bucket_counts(self) -> dict[str, int]:
        counts = {"SAFE": 0, "BACKFILLABLE": 0, "AMBIGUOUS": 0}
        for store in self.stores:
            counts[store.bucket] = counts.get(store.bucket, 0) + 1
        return counts


class MetadataGroup:
    """Small metadata-file-backed group facade for completion helpers."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self._path = path
        self._attrs: dict[str, Any] | None = None

    @property
    def attrs(self) -> dict[str, Any]:
        if self._attrs is None:
            self._attrs = read_zarr_attrs(self._path)
        return self._attrs

    def child_path(self, name: str) -> Path:
        child = self._path
        for part in [piece for piece in str(name).split("/") if piece]:
            child = child / part
        return child

    def child_group(self, name: str) -> "MetadataGroup":
        return MetadataGroup(self.child_path(name))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.child_path(name).is_dir()

    def __getitem__(self, name: str) -> "MetadataGroup":
        if name not in self:
            raise KeyError(name)
        return self.child_group(name)

    def get(self, name: str) -> "MetadataGroup | None":
        return self.child_group(name) if name in self else None

    def keys(self) -> list[str]:
        return self.group_keys()

    def group_keys(self) -> list[str]:
        if not self._path.is_dir():
            return []
        return sorted(path.name for path in self._path.iterdir() if path.is_dir())


def _normalize_text(value: Any) -> str | None:
    normalized = normalize_attr(value)
    if normalized is None:
        return None
    return normalized or None


def read_zarr_attrs(group_path: Path) -> dict[str, Any]:
    zarr_json = group_path / "zarr.json"
    if zarr_json.exists():
        payload = json.loads(zarr_json.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"{zarr_json} did not contain a JSON object")
        attrs = payload.get("attributes")
        return dict(attrs) if isinstance(attrs, Mapping) else {}

    zattrs = group_path / ".zattrs"
    if zattrs.exists():
        payload = json.loads(zattrs.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"{zattrs} did not contain a JSON object")
        return dict(payload)

    return {}


def _resolve_recordings_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return DEFAULT_RECORDINGS_ROOT


def enumerate_filesystem_zarrs(recordings_root: Path) -> tuple[Path, ...]:
    return tuple(
        Path(path)
        for path in iter_filesystem_zarrs(
            [recordings_root],
            recursive=False,
            pattern_policy="recording",
            require_zarr_root=False,
        )
    )


def enumerate_registry_zarrs(registry_path: Path) -> tuple[Path, ...]:
    uri = f"file:{registry_path.expanduser().resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON;")
        rows = conn.execute(
            """
            SELECT DISTINCT zarr_path
            FROM datasets
            WHERE zarr_path IS NOT NULL
              AND TRIM(zarr_path) != ''
              AND (status IS NULL OR status != 'missing')
            ORDER BY zarr_path;
            """
        ).fetchall()
    finally:
        conn.close()
    return tuple(Path(str(row["zarr_path"])).expanduser() for row in rows)


def _path_key(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path.expanduser())


def _dedupe_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    by_key: dict[str, Path] = {}
    for path in paths:
        by_key.setdefault(_path_key(path), path.expanduser())
    return tuple(by_key[key] for key in sorted(by_key))


def _named_run_census(parent: MetadataGroup, attr: str, value: str | None) -> NamedRunCensus:
    if value is None:
        return NamedRunCensus(attr=attr, value=None, exists=False, complete=False)
    try:
        exists = value in parent
    except Exception as exc:
        return NamedRunCensus(attr=attr, value=value, exists=False, complete=False, error=str(exc))
    if not exists:
        return NamedRunCensus(attr=attr, value=value, exists=False, complete=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            complete = is_run_complete_in_parent(parent, parent[value], legacy_default=None)
    except Exception as exc:
        return NamedRunCensus(attr=attr, value=value, exists=True, complete=False, error=str(exc))
    return NamedRunCensus(attr=attr, value=value, exists=True, complete=bool(complete))


def _run_status_by_attr(named_runs: Sequence[NamedRunCensus]) -> dict[str, NamedRunCensus]:
    return {run.attr: run for run in named_runs}


def _first_existing_attr_winner(
    parent: MetadataGroup,
    attrs: Mapping[str, str | None],
    order: Sequence[str],
) -> str | None:
    for key in order:
        value = attrs.get(key)
        if value and value in parent:
            return value
    children = parent.group_keys()
    return children[-1] if children else None


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _has_review_status(parent: MetadataGroup, run_name: str) -> bool:
    if run_name not in parent:
        return False
    try:
        status = parent.child_group(run_name).attrs.get("detect_review_status")
    except Exception:
        return False
    return isinstance(status, Mapping)


def _task_generation_winner(
    parent: MetadataGroup,
    attrs: Mapping[str, str | None],
    order: Sequence[str],
) -> str | None:
    candidate_names = [attrs[key] for key in order if attrs.get(key)]
    if not candidate_names:
        candidate_names = parent.group_keys()
    candidates = _dedupe_preserving_order(str(name) for name in candidate_names if name)
    for run_name in candidates:
        if _has_review_status(parent, run_name):
            return run_name
    return candidates[0] if candidates else None


def _classify_parent(attrs: Mapping[str, str | None], named_runs: Sequence[NamedRunCensus]) -> tuple[str, str]:
    legacy = _run_status_by_attr(named_runs)[LEGACY_DETECT_REVIEW_AUTHORITY_ATTR]
    if attrs.get(AUTHORITATIVE_RUN_ATTR):
        return "SAFE", "authoritative_run is present"
    if legacy.value is None:
        return "SAFE", "legacy detect_review_status_latest attr is absent"
    if not legacy.exists:
        return "AMBIGUOUS", f"legacy attr names missing run {legacy.value!r}"
    if not legacy.complete:
        return "AMBIGUOUS", f"legacy attr names incomplete run {legacy.value!r}"

    conflicts: list[str] = []
    for key in ("latest", "latest_materialized"):
        candidate = _run_status_by_attr(named_runs)[key]
        if candidate.value and candidate.exists and candidate.complete and candidate.value != legacy.value:
            conflicts.append(f"{key}={candidate.value!r} disagrees with legacy={legacy.value!r}")
    if conflicts:
        return "AMBIGUOUS", "; ".join(conflicts)
    return "BACKFILLABLE", f"authoritative_run missing; legacy complete run {legacy.value!r} is mechanical"


def scan_parent(zarr_path: Path, parent_name: str) -> ParentCensus:
    parent_path = zarr_path / parent_name
    if not parent_path.is_dir():
        return ParentCensus(
            zarr_path=str(zarr_path),
            parent_name=parent_name,
            parent_exists=False,
            attrs={key: None for key in POINTER_ATTRS},
            named_runs=(),
            legacy_fallback_would_fire=False,
            batch_winner_with_legacy=None,
            batch_winner_without_legacy=None,
            batch_winner_would_change=False,
            task_generation_winner_with_legacy=None,
            task_generation_winner_without_legacy=None,
            task_generation_winner_would_change=False,
            bucket="SAFE",
            reason=f"{parent_name} parent absent",
        )
    parent = MetadataGroup(parent_path)
    try:
        parent_attrs = parent.attrs
        attrs = {key: _normalize_text(parent_attrs.get(key)) for key in POINTER_ATTRS}
        named_runs = tuple(_named_run_census(parent, key, value) for key, value in attrs.items())
        by_attr = _run_status_by_attr(named_runs)
        legacy = by_attr[LEGACY_DETECT_REVIEW_AUTHORITY_ATTR]
        legacy_fallback_would_fire = (
            attrs[AUTHORITATIVE_RUN_ATTR] is None
            and legacy.value is not None
            and legacy.exists
            and legacy.complete
        )
        batch_with = _first_existing_attr_winner(parent, attrs, BATCH_POINTER_ORDER)
        batch_without = _first_existing_attr_winner(
            parent,
            attrs,
            tuple(key for key in BATCH_POINTER_ORDER if key != LEGACY_DETECT_REVIEW_AUTHORITY_ATTR),
        )
        task_with = _task_generation_winner(parent, attrs, TASK_GENERATION_POINTER_ORDER)
        task_without = _task_generation_winner(
            parent,
            attrs,
            tuple(key for key in TASK_GENERATION_POINTER_ORDER if key != LEGACY_DETECT_REVIEW_AUTHORITY_ATTR),
        )
        bucket, reason = _classify_parent(attrs, named_runs)
    except Exception as exc:
        return ParentCensus(
            zarr_path=str(zarr_path),
            parent_name=parent_name,
            parent_exists=True,
            attrs={key: None for key in POINTER_ATTRS},
            named_runs=(),
            legacy_fallback_would_fire=False,
            batch_winner_with_legacy=None,
            batch_winner_without_legacy=None,
            batch_winner_would_change=False,
            task_generation_winner_with_legacy=None,
            task_generation_winner_without_legacy=None,
            task_generation_winner_would_change=False,
            bucket="AMBIGUOUS",
            reason="parent metadata unreadable",
            error=f"{type(exc).__name__}: {exc}",
        )

    return ParentCensus(
        zarr_path=str(zarr_path),
        parent_name=parent_name,
        parent_exists=True,
        attrs=attrs,
        named_runs=named_runs,
        legacy_fallback_would_fire=legacy_fallback_would_fire,
        batch_winner_with_legacy=batch_with,
        batch_winner_without_legacy=batch_without,
        batch_winner_would_change=batch_with != batch_without,
        task_generation_winner_with_legacy=task_with,
        task_generation_winner_without_legacy=task_without,
        task_generation_winner_would_change=task_with != task_without,
        bucket=bucket,
        reason=reason,
    )


def _combine_parent_buckets(parents: Sequence[ParentCensus]) -> tuple[str, str]:
    existing = [parent for parent in parents if parent.parent_exists]
    if not existing:
        return "SAFE", "no refined-detect parent present"
    ambiguous = [parent for parent in existing if parent.bucket == "AMBIGUOUS"]
    if ambiguous:
        return "AMBIGUOUS", "; ".join(f"{parent.parent_name}: {parent.reason}" for parent in ambiguous)
    backfillable = [parent for parent in existing if parent.bucket == "BACKFILLABLE"]
    if backfillable:
        return "BACKFILLABLE", "; ".join(f"{parent.parent_name}: {parent.reason}" for parent in backfillable)
    return "SAFE", "; ".join(f"{parent.parent_name}: {parent.reason}" for parent in existing)


def scan_store(zarr_path: Path, source_modalities: Sequence[str]) -> StoreCensus:
    try:
        if not zarr_path.is_dir():
            raise RuntimeError("store path is not a readable directory")
        parents = tuple(scan_parent(zarr_path, parent_name) for parent_name in REFINED_DETECT_PARENT_NAMES)
        unreadable = next((parent.error for parent in parents if parent.error), None)
        bucket, reason = _combine_parent_buckets(parents)
        return StoreCensus(
            zarr_path=str(zarr_path),
            source_modalities=tuple(sorted(source_modalities)),
            parents=parents,
            bucket=bucket,
            reason=reason,
            unreadable_error=unreadable,
        )
    except Exception as exc:
        return StoreCensus(
            zarr_path=str(zarr_path),
            source_modalities=tuple(sorted(source_modalities)),
            parents=(),
            bucket="AMBIGUOUS",
            reason="store unreadable",
            unreadable_error=f"{type(exc).__name__}: {exc}",
        )


def run_census(*, recordings_root: Path, registry_path: Path) -> DetectReviewPointerCensus:
    filesystem_paths = _dedupe_paths(enumerate_filesystem_zarrs(recordings_root))
    registry_error: str | None = None
    try:
        registry_paths = _dedupe_paths(enumerate_registry_zarrs(registry_path))
    except Exception as exc:
        registry_paths = ()
        registry_error = f"{type(exc).__name__}: {exc}"

    fs_keys = {_path_key(path): path for path in filesystem_paths}
    registry_keys = {_path_key(path): path for path in registry_paths}
    all_keys = sorted(set(fs_keys) | set(registry_keys))
    stores = []
    for key in all_keys:
        path = fs_keys.get(key) or registry_keys[key]
        modalities = []
        if key in fs_keys:
            modalities.append("filesystem")
        if key in registry_keys:
            modalities.append("registry")
        stores.append(scan_store(path, modalities))

    enumeration = EnumerationCensus(
        filesystem_count=len(filesystem_paths),
        registry_count=len(registry_paths),
        both_count=len(set(fs_keys) & set(registry_keys)),
        filesystem_only=tuple(str(fs_keys[key]) for key in sorted(set(fs_keys) - set(registry_keys))),
        registry_only=tuple(str(registry_keys[key]) for key in sorted(set(registry_keys) - set(fs_keys))),
        registry_error=registry_error,
    )
    return DetectReviewPointerCensus(
        recordings_root=str(recordings_root),
        registry_path=str(registry_path),
        enumeration=enumeration,
        stores=tuple(stores),
    )


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def _summarize_counts(census: DetectReviewPointerCensus) -> dict[str, int]:
    return {
        "stores_scanned": len(census.stores),
        "unreadable": sum(1 for store in census.stores if store.unreadable_error),
        "fallback_would_fire": sum(
            1
            for store in census.stores
            if any(parent.legacy_fallback_would_fire for parent in store.parents)
        ),
        "winner_would_change": sum(
            1
            for store in census.stores
            if any(
                parent.batch_winner_would_change or parent.task_generation_winner_would_change
                for parent in store.parents
            )
        ),
        "clean_safe": census.bucket_counts.get("SAFE", 0),
    }


def render_markdown(census: DetectReviewPointerCensus) -> str:
    counts = _summarize_counts(census)
    bucket_counts = census.bucket_counts
    lines = [
        "# Detect-review pointer census - 2026-07-06",
        "",
        "Read-only census of legacy `detect_review_status_latest` pointers in refined-detect parents.",
        "",
        "## Inputs",
        "",
        f"- Recordings root: `{census.recordings_root}`",
        f"- Registry: `{census.registry_path}`",
        "",
        "## Enumeration",
        "",
        "| modality | count |",
        "| --- | ---: |",
        f"| filesystem | {census.enumeration.filesystem_count} |",
        f"| registry | {census.enumeration.registry_count} |",
        f"| both | {census.enumeration.both_count} |",
        f"| filesystem only | {len(census.enumeration.filesystem_only)} |",
        f"| registry only | {len(census.enumeration.registry_only)} |",
        "",
    ]
    if census.enumeration.registry_error:
        lines.extend(["Registry enumeration error:", "", f"```text\n{census.enumeration.registry_error}\n```", ""])
    lines.extend(
        [
            "### Filesystem-only stores",
            "",
            *[f"- `{path}`" for path in census.enumeration.filesystem_only[:200]],
        ]
    )
    if len(census.enumeration.filesystem_only) > 200:
        lines.append(f"- ... {len(census.enumeration.filesystem_only) - 200} more")
    lines.extend(["", "### Registry-only stores", ""])
    lines.extend(f"- `{path}`" for path in census.enumeration.registry_only[:200])
    if len(census.enumeration.registry_only) > 200:
        lines.append(f"- ... {len(census.enumeration.registry_only) - 200} more")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| metric | count |",
            "| --- | ---: |",
            f"| stores scanned | {counts['stores_scanned']} |",
            f"| unreadable stores/parents | {counts['unreadable']} |",
            f"| fallback would fire | {counts['fallback_would_fire']} |",
            f"| winner would change | {counts['winner_would_change']} |",
            f"| clean SAFE stores | {counts['clean_safe']} |",
            "",
            "## Buckets",
            "",
            "| bucket | count |",
            "| --- | ---: |",
            f"| SAFE | {bucket_counts.get('SAFE', 0)} |",
            f"| BACKFILLABLE | {bucket_counts.get('BACKFILLABLE', 0)} |",
            f"| AMBIGUOUS | {bucket_counts.get('AMBIGUOUS', 0)} |",
            "",
            "## Pointer-impact rows",
            "",
            "| bucket | store | parent | authoritative_run | detect_review_status_latest | latest | latest_materialized | fallback | batch winner with/without | task winner with/without | reason |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    impact_rows = [
        (store, parent)
        for store in census.stores
        for parent in store.parents
        if parent.parent_exists
        and (
            parent.bucket != "SAFE"
            or parent.legacy_fallback_would_fire
            or parent.batch_winner_would_change
            or parent.task_generation_winner_would_change
        )
    ]
    for store, parent in impact_rows:
        attrs = parent.attrs
        lines.append(
            "| "
            + " | ".join(
                [
                    parent.bucket,
                    f"`{store.zarr_path}`",
                    parent.parent_name,
                    attrs.get(AUTHORITATIVE_RUN_ATTR) or "",
                    attrs.get(LEGACY_DETECT_REVIEW_AUTHORITY_ATTR) or "",
                    attrs.get("latest") or "",
                    attrs.get("latest_materialized") or "",
                    _format_bool(parent.legacy_fallback_would_fire),
                    f"{parent.batch_winner_with_legacy or ''} / {parent.batch_winner_without_legacy or ''}",
                    f"{parent.task_generation_winner_with_legacy or ''} / {parent.task_generation_winner_without_legacy or ''}",
                    parent.reason.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    if not impact_rows:
        lines.append("| SAFE |  |  |  |  |  |  | no |  |  | no fallback or winner-change rows |")

    unreadable = [store for store in census.stores if store.unreadable_error]
    lines.extend(["", "## Unreadable Rows", ""])
    if unreadable:
        lines.extend(["| store | error |", "| --- | --- |"])
        for store in unreadable:
            lines.append(f"| `{store.zarr_path}` | `{store.unreadable_error}` |")
    else:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## Decision Memo",
            "",
            "_Fill in after writer-side re-grep and validation._",
            "",
        ]
    )
    return "\n".join(lines)


def _json_ready(census: DetectReviewPointerCensus) -> dict[str, Any]:
    return asdict(census)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=None,
        help="Recording root to scan; defaults to PALETTE_RECORDINGS_ROOT or /nvme1/recordings.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry SQLite path; defaults to RegistryPaths.from_env(Path.cwd()).path.",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    recordings_root = _resolve_recordings_root(args.recordings_root)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    census = run_census(recordings_root=recordings_root, registry_path=registry_path)
    if args.format == "json":
        rendered = json.dumps(_json_ready(census), indent=2, sort_keys=True)
    else:
        rendered = render_markdown(census)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
