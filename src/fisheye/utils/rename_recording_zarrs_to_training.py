#!/usr/bin/env python3
"""Safely rename legacy recording Zarr files to *_training.zarr.

Default behavior is dry-run. Use --apply to perform renames.

Safety guards:
- Only considers files under */zarr/*.zarr.
- Skips files already ending with _training.zarr or _analysis.zarr.
- Skips if target *_training.zarr already exists.
- Skips if directory contains multiple legacy .zarr files (ambiguous mapping),
  unless --allow-multiple-legacy is provided.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    target: Path
    status: str
    reason: Optional[str] = None


def _iter_candidate_zarrs(roots: List[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")
            yield from root.glob("*.zarr")


def _is_under_zarr_dir(path: Path) -> bool:
    return path.parent.name == "zarr"


def _is_legacy_name(path: Path) -> bool:
    return not (path.name.endswith("_training.zarr") or path.name.endswith("_analysis.zarr"))


def _build_plan(
    roots: List[Path],
    *,
    recursive: bool,
    allow_multiple_legacy: bool,
) -> List[RenamePlan]:
    candidates = sorted({p.resolve() for p in _iter_candidate_zarrs(roots, recursive)})
    by_parent: dict[Path, List[Path]] = {}
    for path in candidates:
        if not _is_under_zarr_dir(path):
            continue
        by_parent.setdefault(path.parent, []).append(path)

    plans: List[RenamePlan] = []
    for parent, zarrs in sorted(by_parent.items(), key=lambda item: str(item[0])):
        legacy = [p for p in zarrs if _is_legacy_name(p)]
        if not legacy:
            continue
        if len(legacy) > 1 and not allow_multiple_legacy:
            for src in legacy:
                target = src.with_name(f"{src.stem}_training.zarr")
                plans.append(
                    RenamePlan(
                        source=src,
                        target=target,
                        status="skip",
                        reason="multiple legacy zarr files in same zarr/ directory",
                    )
                )
            continue

        for src in legacy:
            target = src.with_name(f"{src.stem}_training.zarr")
            if target.exists():
                plans.append(
                    RenamePlan(
                        source=src,
                        target=target,
                        status="skip",
                        reason="target already exists",
                    )
                )
                continue
            plans.append(RenamePlan(source=src, target=target, status="rename"))
    return plans


def _apply_renames(plans: List[RenamePlan]) -> int:
    renamed = 0
    for plan in plans:
        if plan.status != "rename":
            continue
        plan.source.rename(plan.target)
        renamed += 1
    return renamed


def _print_summary(plans: List[RenamePlan], *, applied: bool, limit: int) -> None:
    rename_rows = [p for p in plans if p.status == "rename"]
    skip_rows = [p for p in plans if p.status == "skip"]

    print("Recording Zarr Rename Plan")
    print(f"- planned renames: {len(rename_rows)}")
    print(f"- skipped: {len(skip_rows)}")
    print(f"- mode: {'apply' if applied else 'dry-run'}")

    if rename_rows:
        print("\nRenames:")
        for plan in rename_rows[:limit]:
            print(f"  - {plan.source}")
            print(f"    -> {plan.target}")
        if len(rename_rows) > limit:
            print(f"  ... ({len(rename_rows) - limit} more)")

    if skip_rows:
        print("\nSkipped:")
        for plan in skip_rows[:limit]:
            reason = plan.reason or "unspecified"
            print(f"  - {plan.source} [{reason}]")
        if len(skip_rows) > limit:
            print(f"  ... ({len(skip_rows) - limit} more)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots (default: /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively find .zarr under provided roots.")
    parser.add_argument("--apply", action="store_true", help="Perform renames. Default is dry-run.")
    parser.add_argument(
        "--allow-multiple-legacy",
        action="store_true",
        help="Allow renaming when more than one legacy .zarr exists in a single zarr/ directory.",
    )
    parser.add_argument("--list-limit", type=int, default=50, help="Max rows to print per section.")
    args = parser.parse_args(argv)

    roots = args.paths or [Path("/nvme1/recordings")]
    plans = _build_plan(
        roots,
        recursive=bool(args.recursive),
        allow_multiple_legacy=bool(args.allow_multiple_legacy),
    )

    renamed = 0
    if args.apply:
        renamed = _apply_renames(plans)

    _print_summary(plans, applied=bool(args.apply), limit=max(1, int(args.list_limit)))
    if args.apply:
        print(f"\nApplied renames: {renamed}")
        print("Next steps:")
        print("  1) scripts/py -m fisheye.utils.registry_rescan --registry /nvme1/palette_registry.sqlite /nvme1/recordings --recursive")
        print("  2) scripts/py -m fisheye.registry.maintenance --registry /nvme1/palette_registry.sqlite --reconcile-registry")
        print("  3) scripts/py -m fisheye.registry.maintenance --registry /nvme1/palette_registry.sqlite --check-integrity --list-limit 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

