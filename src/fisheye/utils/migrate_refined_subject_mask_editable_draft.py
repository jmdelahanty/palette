"""Safely demote one unapproved canonical refined-mask run for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import zarr

from fisheye.shared.refined_subject_mask_mutation import (
    migrate_unapproved_refined_subject_mask_to_editable_draft,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, inspect and report only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = zarr.open_group(
        str(args.zarr_path.expanduser()),
        mode="a" if args.apply else "r",
        use_consolidated=False,
    )
    if not args.apply:
        parent = root["refined_subject_masks_runs"]
        run = parent[str(args.run)]
        payload = {
            "status": "dry_run",
            "run": str(args.run),
            "coordinate_contract": run.attrs.get("coordinate_contract"),
            "stage_selector_eligible": run.attrs.get("stage_selector_eligible"),
            "review_status": run.attrs.get("refined_subject_mask_review_status"),
            "parent_selectors": {
                name: parent.attrs.get(name)
                for name in (
                    "latest",
                    "latest_complete",
                    "latest_pending",
                    "authoritative_run",
                )
            },
            "scientific_arrays_rewritten": False,
        }
    else:
        payload = {
            "status": "applied",
            **migrate_unapproved_refined_subject_mask_to_editable_draft(
                root,
                str(args.run),
            ),
        }
        zarr.consolidate_metadata(str(args.zarr_path.expanduser()))
        payload["root_metadata_reconsolidated"] = True
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
