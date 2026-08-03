"""Build one exact runner receipt for published chaser components.

The receipt is generated only after every requested workflow step succeeds.  It
reopens the authoritative archive read-only, validates every named immutable
component against the exact chaser-distance base, and records the portable
dependency handle that a later workflow node must use.  It never changes a
selector or registry record.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.analysis.chaser_component_publication import (
    build_chaser_component_handle,
    canonical_component_json_bytes,
    component_record_sha256,
    validate_chaser_component_handle,
)
from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr_io import open_zarr_root

CHASER_COMPONENT_RUNNER_RECEIPT_SCHEMA_ID = "palette.chaser_component_runner_receipt"
CHASER_COMPONENT_RUNNER_RECEIPT_SCHEMA_VERSION = 1
_RECEIPT_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "status",
        "zarr_path",
        "requested_chaser_distance_run",
        "resolved_base_run_name",
        "resolved_base_run_path",
        "base_read_authority_sha256",
        "requested_component_count",
        "components",
        "record_sha256",
    }
)
_COMPONENT_FIELDS = frozenset(
    {
        "component_family",
        "component_name",
        "component_path",
        "component_manifest_sha256",
        "dependency_handle",
        "authority_mode",
        "selector_eligible",
        "validation",
    }
)
_VALIDATION_FIELDS = frozenset({"valid", "payload_array_count", "payload_group_count"})


def _controlled_name(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text:
        raise ValueError(f"{label} must be one controlled name.")
    return text


def parse_component_request(value: str) -> tuple[str, str]:
    """Parse one exact ``FAMILY=NAME`` request."""

    family, separator, name = str(value).partition("=")
    if not separator:
        raise ValueError("--component must use FAMILY=NAME.")
    return (
        _controlled_name(family, label="component family"),
        _controlled_name(name, label="component name"),
    )


def build_chaser_component_runner_receipt(
    zarr_path: Path | str,
    *,
    chaser_distance_run: str,
    component_requests: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Validate all requested outputs and return a self-digested receipt."""

    path = Path(zarr_path).expanduser().resolve()
    requested_run = str(chaser_distance_run or "").strip()
    if not requested_run:
        raise ValueError("chaser_distance_run is required.")
    normalized = tuple(
        (
            _controlled_name(family, label="component family"),
            _controlled_name(name, label="component name"),
        )
        for family, name in component_requests
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Component runner receipt contains a duplicate request.")

    root = open_zarr_root(path, mode="r")
    snapshot = load_chaser_distance_run(root, run_name=requested_run)
    components: list[dict[str, Any]] = []
    for family, name in normalized:
        relative_path = f"{family}/{name}"
        component_path = f"{snapshot.run_path}/{relative_path}"
        try:
            component = root[component_path]
        except Exception as exc:
            raise ValueError(
                f"Requested chaser component is unavailable: {component_path!r}."
            ) from exc
        handle = build_chaser_component_handle(
            component,
            snapshot=snapshot,
            relative_path=relative_path,
        )
        validate_chaser_component_handle(handle, snapshot=snapshot)
        if component.attrs.get("stage_selector_eligible") is not False:
            raise ValueError(
                "Runner receipts authorize only explicit selector-ineligible "
                f"component dependencies: {component_path!r}."
            )
        manifest = component.attrs["chaser_component_publication_manifest"]
        components.append(
            {
                "component_family": family,
                "component_name": name,
                "component_path": component_path,
                "component_manifest_sha256": handle["component_manifest_sha256"],
                "dependency_handle": copy.deepcopy(handle),
                "authority_mode": "explicit_dependency_handle",
                "selector_eligible": False,
                "validation": {
                    "valid": True,
                    "payload_array_count": len(manifest["payload"]["arrays"]),
                    "payload_group_count": len(manifest["payload"]["groups"]),
                },
            }
        )

    body = {
        "schema_id": CHASER_COMPONENT_RUNNER_RECEIPT_SCHEMA_ID,
        "schema_version": CHASER_COMPONENT_RUNNER_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "zarr_path": str(path),
        "requested_chaser_distance_run": requested_run,
        "resolved_base_run_name": snapshot.run_name,
        "resolved_base_run_path": snapshot.run_path,
        "base_read_authority_sha256": component_record_sha256(
            snapshot.authority_record()
        ),
        "requested_component_count": len(normalized),
        "components": components,
    }
    return {**body, "record_sha256": component_record_sha256(body)}


def validate_chaser_component_runner_receipt(
    receipt: Any,
    *,
    expected_zarr_path: Path | str | None = None,
) -> Mapping[str, Any]:
    """Reopen and recompute one complete runner receipt fail closed."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("Chaser component runner receipt fields are not exact.")
    if (
        receipt["schema_id"] != CHASER_COMPONENT_RUNNER_RECEIPT_SCHEMA_ID
        or receipt["schema_version"] != CHASER_COMPONENT_RUNNER_RECEIPT_SCHEMA_VERSION
        or receipt["status"] != "complete"
    ):
        raise ValueError("Chaser component runner receipt identity is unsupported.")
    body = {key: receipt[key] for key in _RECEIPT_FIELDS if key != "record_sha256"}
    digest = str(receipt["record_sha256"] or "")
    if component_record_sha256(body) != digest:
        raise ValueError("Chaser component runner receipt digest does not match.")
    path = Path(str(receipt["zarr_path"])).expanduser().resolve()
    if (
        expected_zarr_path is not None
        and path != Path(expected_zarr_path).expanduser().resolve()
    ):
        raise ValueError("Chaser component runner receipt names a different archive.")
    components = receipt["components"]
    if not isinstance(components, list) or receipt["requested_component_count"] != len(
        components
    ):
        raise ValueError("Chaser component runner receipt count is inconsistent.")
    requests: list[tuple[str, str]] = []
    for row in components:
        if not isinstance(row, Mapping) or set(row) != _COMPONENT_FIELDS:
            raise ValueError("Chaser component runner row fields are not exact.")
        validation = row["validation"]
        if (
            not isinstance(validation, Mapping)
            or set(validation) != _VALIDATION_FIELDS
            or validation["valid"] is not True
        ):
            raise ValueError("Chaser component runner validation is not exact.")
        requests.append(
            (
                _controlled_name(row["component_family"], label="component family"),
                _controlled_name(row["component_name"], label="component name"),
            )
        )
    recomputed = build_chaser_component_runner_receipt(
        path,
        chaser_distance_run=str(receipt["requested_chaser_distance_run"]),
        component_requests=requests,
    )
    if canonical_component_json_bytes(recomputed) != canonical_component_json_bytes(
        receipt
    ):
        raise ValueError("Chaser component runner receipt or payload changed.")
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--chaser-distance-run", required=True)
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="FAMILY=NAME",
        help="Exact component output to validate; may be repeated.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = build_chaser_component_runner_receipt(
        args.zarr_path,
        chaser_distance_run=args.chaser_distance_run,
        component_requests=[parse_component_request(value) for value in args.component],
    )
    write_json_atomic(args.output_json, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHASER_COMPONENT_RUNNER_RECEIPT_SCHEMA_ID",
    "CHASER_COMPONENT_RUNNER_RECEIPT_SCHEMA_VERSION",
    "build_chaser_component_runner_receipt",
    "parse_component_request",
    "validate_chaser_component_runner_receipt",
]
