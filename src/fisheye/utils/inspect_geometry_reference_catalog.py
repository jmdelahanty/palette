"""Build or revalidate a read-only index of producer-approved geometry references.

All output goes to stdout. This command never publishes candidates, moves a
selector, writes approvals, or changes its source recordings. Source-list JSON
contains an explicit GeometryReferenceSource object per recording, including
nullable applicable_at_utc; no path/camera/registration discovery is implicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fisheye.shared.recording_geometry import RecordingGeometryError
from fisheye.analysis_workflows.materializers.arena_geometry_reference_catalog import (
    GeometryReferenceKey,
    build_geometry_reference_catalog,
    geometry_reference_source_from_record,
    resolve_geometry_reference,
    validate_geometry_reference_catalog,
)


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise RecordingGeometryError(f"Non-finite JSON value: {value}.")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise RecordingGeometryError(f"Duplicate JSON field: {key}.")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_keys,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="Revalidate sources and print the index.")
    build.add_argument("--source-list", type=Path, required=True)
    validate = actions.add_parser("validate", help="Reopen all indexed sources.")
    validate.add_argument("--catalog", type=Path, required=True)
    resolve = actions.add_parser(
        "resolve", help="Resolve an exact reference/source pair."
    )
    resolve.add_argument("--catalog", type=Path, required=True)
    resolve.add_argument("--key-json", type=Path, required=True)
    resolve.add_argument("--source-zarr", type=Path, required=True)
    resolve.add_argument("--applicable-at-utc")
    args = parser.parse_args(argv)
    if args.action == "build":
        inputs = _read_json(args.source_list)
        if not isinstance(inputs, list):
            raise RecordingGeometryError("Source-list JSON must be an array.")
        result = build_geometry_reference_catalog(
            [geometry_reference_source_from_record(value) for value in inputs]
        )
    elif args.action == "validate":
        catalog = validate_geometry_reference_catalog(_read_json(args.catalog))
        result = {
            "status": "validated_index_only",
            "catalog_sha256": catalog["catalog_sha256"],
            "reference_count": len(catalog["entries"]),
            "source_count": sum(len(entry["sources"]) for entry in catalog["entries"]),
            "authority_activated": False,
        }
    else:
        result = resolve_geometry_reference(
            _read_json(args.catalog),
            key=GeometryReferenceKey(**_read_json(args.key_json)),
            source_zarr=args.source_zarr,
            applicable_at_utc=args.applicable_at_utc,
        )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
