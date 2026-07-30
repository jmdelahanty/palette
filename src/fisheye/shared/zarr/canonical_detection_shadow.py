"""Standalone selector-ineligible canonical raw-detection shadow publisher."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.canonical_detection_benchmark_input import (
    CanonicalDetectionBenchmarkInput,
    load_detection_benchmark_input,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    build_canonical_detection_run_manifest,
    build_coordinate_canonical_detection_run_manifest,
    build_legacy_detection_source_evidence,
    refined_source_identity_from_canonical_manifest,
    validate_canonical_detection_publication,
    validate_legacy_detection_source_evidence,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_schema import CanonicalDetectionDimensions
from fisheye.shared.zarr.detection_storage import (
    CanonicalDetectionStoragePlanSet,
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)


DEFAULT_CANONICAL_DETECTION_SHADOW_ROOT = Path(
    "/tmp/palette-canonical-detection-shadows"
)
CANONICAL_DETECTION_SHADOW_RECEIPT_SCHEMA_ID = (
    "palette.canonical_detection.shadow_publication"
)
CANONICAL_DETECTION_SHADOW_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CanonicalDetectionShadowPublication:
    """Validated canonical source artifact and its open array evidence."""

    output_path: Path
    run_id: str
    dimensions: CanonicalDetectionDimensions
    plans: CanonicalDetectionStoragePlanSet
    manifest: Mapping[str, object]
    arrays: Mapping[str, Any]
    receipt: Mapping[str, object]

    def refined_source_identity(self):
        return refined_source_identity_from_canonical_manifest(self.manifest)


def validate_canonical_detection_shadow_publication(
    publication: CanonicalDetectionShadowPublication,
) -> tuple[str, ...]:
    """Reopen persisted metadata and re-run the complete canonical gate."""

    if publication.plans.dimensions != publication.dimensions:
        return ("canonical shadow plan dimensions mismatch",)
    try:
        direct, consolidated = _metadata_maps(
            publication.output_path,
            run_id=publication.run_id,
            plans=publication.plans,
        )
    except (OSError, TypeError, ValueError) as exc:
        return (f"canonical shadow metadata reopen failed: {exc}",)
    errors = list(
        validate_canonical_detection_publication(
            publication.manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=publication.arrays,
        )
    )
    payload = publication.manifest.get("payload")
    source_evidence = (
        payload.get("source_evidence") if isinstance(payload, Mapping) else None
    )
    if not isinstance(source_evidence, Mapping):
        errors.append("canonical shadow source evidence is missing")
        return tuple(dict.fromkeys(errors))
    source_errors = validate_legacy_detection_source_evidence(source_evidence)
    errors.extend(source_errors)
    if not source_errors:
        try:
            source_path = Path(str(source_evidence["source_group_path"]))
            source_group = zarr.open_group(
                str(source_path),
                mode="r",
                use_consolidated=False,
            )
            observed_source_evidence = build_legacy_detection_source_evidence(
                source_group,
                source_group_path=source_path,
                source_run_id=str(source_evidence["source_run_id"]),
                recording_identity=str(source_evidence["recording_identity"]),
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"canonical legacy source evidence reopen failed: {exc}")
        else:
            if observed_source_evidence != dict(source_evidence):
                errors.append("canonical legacy source evidence changed on disk")
    return tuple(dict.fromkeys(errors))


def require_safe_canonical_detection_shadow_destination(
    destination: Path,
    *,
    shadow_root: Path = DEFAULT_CANONICAL_DETECTION_SHADOW_ROOT,
) -> Path:
    path = destination.expanduser().resolve()
    root = shadow_root.expanduser().resolve()
    root_is_safe = root.is_relative_to(Path("/tmp").resolve()) or (
        ".palette_benchmarks" in root.parts
    )
    if not root_is_safe:
        raise ValueError(
            "Canonical shadow roots must be below /tmp or .palette_benchmarks."
        )
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Canonical shadow destination must be a child of {root}.")
    if path.suffix != ".zarr":
        raise ValueError("Canonical shadow destination must use a .zarr suffix.")
    if path.exists():
        raise FileExistsError(f"Canonical shadow destination exists: {path}")
    if any(part.endswith("_analysis.zarr") for part in path.parts[:-1]):
        raise ValueError("Canonical shadow cannot be nested in a recording archive.")
    return path


def _write_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: Any,
) -> None:
    if plan.chunk_shape is None:
        raise ValueError("Canonical detection arrays cannot be scalars.")
    unit_rows = int(
        plan.shard_shape[0] if plan.shard_shape is not None else plan.chunk_shape[0]
    )
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), unit_rows):
        stop = min(start + unit_rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _read_historical_zarr_envelope(path: Path) -> dict[str, Any]:
    """Read an enclosing legacy archive declaration permissively.

    Historical root attributes can contain unrelated bare ``NaN`` or
    ``Infinity`` values.  The enclosing root is therefore discovery-only; the
    exact selected canonical run declarations are independently required to be
    strict finite JSON below.
    """

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _metadata_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: CanonicalDetectionStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    relative_paths = ("", "instances", *(entry.rule.path for entry in plans.entries))
    run_prefix = f"detect_runs/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = output_path / run_prefix
        if relative:
            metadata_path = metadata_path / relative
        direct[relative] = _read_strict_json(metadata_path / "zarr.json")
    archive_root = _read_historical_zarr_envelope(output_path / "zarr.json")
    envelope = archive_root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping):
        raise ValueError("Canonical shadow lacks root consolidated metadata.")
    if envelope.get("kind") != "inline" or envelope.get("must_understand") is not False:
        raise ValueError("Canonical consolidated metadata envelope is invalid.")
    flattened = envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Canonical consolidated metadata map is missing.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_prefix if not relative else f"{run_prefix}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Canonical consolidated metadata lacks {full_path!r}.")
        try:
            canonical_json_bytes(declaration)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Canonical consolidated metadata at {full_path!r} is not "
                f"strict finite JSON: {exc}"
            ) from exc
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def canonical_detection_metadata_declaration_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: CanonicalDetectionStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return exact direct/consolidated declarations for one canonical v1 run."""

    return _metadata_maps(output_path, run_id=run_id, plans=plans)


def publish_legacy_canonical_detection_shadow(
    *,
    source_group_path: Path,
    source_evidence_group_path: Path | None = None,
    recording_identity: str,
    source_run_id: str,
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_CANONICAL_DETECTION_SHADOW_ROOT,
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    coordinate_catalog: bool = False,
    preserve_source_instance_keys: bool = False,
) -> CanonicalDetectionShadowPublication:
    """Convert one complete legacy run into a validated canonical shadow."""

    if type(coordinate_catalog) is not bool:
        raise TypeError("coordinate_catalog must be an exact bool.")
    if type(preserve_source_instance_keys) is not bool:
        raise TypeError("preserve_source_instance_keys must be an exact bool.")

    output_path = require_safe_canonical_detection_shadow_destination(
        destination,
        shadow_root=shadow_root,
    )
    source_path = source_group_path.expanduser().resolve()
    source_group = zarr.open_group(
        str(source_path),
        mode="r",
        use_consolidated=False,
    )
    benchmark_input = load_detection_benchmark_input(
        source_path,
        recording_identity=str(recording_identity),
        frame_limit=None,
    )
    if preserve_source_instance_keys:
        if "instance_key" not in source_group:
            raise ValueError("Canonical source lacks persisted instance_key values.")
        arrays = dict(benchmark_input.arrays)
        arrays["instances/instance_key"] = np.ascontiguousarray(
            np.asarray(source_group["instance_key"][:])
        )
        source_identity = dict(benchmark_input.source_identity)
        conversion = dict(source_identity.get("conversion") or {})
        conversion["instance_key"] = "preserved_from_source_explicit_policy"
        source_identity["conversion"] = conversion
        benchmark_input = CanonicalDetectionBenchmarkInput(
            dimensions=benchmark_input.dimensions,
            arrays=arrays,
            source_identity=source_identity,
        )
    evidence_path = (
        source_path
        if source_evidence_group_path is None
        else source_evidence_group_path.expanduser().resolve()
    )
    evidence_group = (
        source_group
        if evidence_path == source_path
        else zarr.open_group(
            str(evidence_path),
            mode="r",
            use_consolidated=False,
        )
    )
    source_evidence = build_legacy_detection_source_evidence(
        evidence_group,
        source_group_path=evidence_path,
        source_run_id=str(source_run_id),
        recording_identity=str(recording_identity),
    )
    if evidence_path != source_path:
        staged_evidence = build_legacy_detection_source_evidence(
            source_group,
            source_group_path=source_path,
            source_run_id=str(source_run_id),
            recording_identity=str(recording_identity),
        )
        for field in (
            "source_group_metadata_sha256",
            "source_arrays_digest",
            "source_arrays",
        ):
            if staged_evidence[field] != source_evidence[field]:
                raise ValueError(
                    "Staged canonical source differs from its authoritative "
                    f"evidence at {field!r}."
                )
    plans = plan_canonical_detection_storage(
        benchmark_input.dimensions,
        profile=profile,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "schema_id": CANONICAL_DETECTION_SHADOW_RECEIPT_SCHEMA_ID,
            "schema_version": CANONICAL_DETECTION_SHADOW_RECEIPT_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
        }
    )
    family = root.create_group("detect_runs")
    family.attrs.update(
        {
            "benchmark_only": True,
            "selector_eligible": False,
            "selection_contract": "none_shadow_direct_path_only",
        }
    )
    run = family.create_group(str(run_id))
    run.attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "shadow_only": True,
            "logical_schema": CANONICAL_DETECTION_SCHEMA_V1.as_manifest(
                dimensions=benchmark_input.dimensions
            ),
            "storage_plan": plans.as_manifest(),
            "source_evidence": source_evidence,
        }
    )
    instances = run.create_group("instances")
    destination_arrays: dict[str, Any] = {}
    writes: list[dict[str, object]] = []
    try:
        binding_by_path = {
            binding.path: binding for binding in CANONICAL_DETECTION_SCHEMA_V1.bindings
        }
        for entry in plans.entries:
            path = entry.rule.path
            leaf = path.split("/", 1)[1]
            values = np.asarray(benchmark_input.arrays[path])
            binding = binding_by_path[path]
            contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            array = create_array_from_plan(
                instances,
                name=leaf,
                contract=contract,
                plan=entry.plan,
                fill_value=0,
                attributes={"shadow_only": True, "selector_eligible": False},
            )
            _write_by_physical_units(array, values, plan=entry.plan)
            destination_arrays[path] = array
            writes.append(
                {
                    "path": path,
                    "logical_shape": list(values.shape),
                    "logical_dtype": str(values.dtype),
                    "chunk_shape": list(entry.plan.chunk_shape or ()),
                    "shard_shape": (
                        None
                        if entry.plan.shard_shape is None
                        else list(entry.plan.shard_shape)
                    ),
                    "write_ownership": entry.plan.write_ownership,
                }
            )

        CANONICAL_DETECTION_SCHEMA_V1.require(
            destination_arrays,
            dimensions=benchmark_input.dimensions,
        )
        source_hashes = {
            path: sha256_array(np.asarray(values))
            for path, values in benchmark_input.arrays.items()
        }
        destination_hashes = {
            path: sha256_array(np.asarray(array[...]))
            for path, array in destination_arrays.items()
        }
        if source_hashes != destination_hashes:
            raise RuntimeError("Canonical shadow decoded values differ from source.")

        run.attrs["status"] = "complete"
        first_consolidation = consolidate_metadata_capture_expected_warnings(
            output_path
        )
        direct, consolidated = _metadata_maps(
            output_path,
            run_id=str(run_id),
            plans=plans,
        )
        manifest_builder = (
            build_coordinate_canonical_detection_run_manifest
            if coordinate_catalog
            else build_canonical_detection_run_manifest
        )
        manifest_kwargs: dict[str, Any] = {}
        if coordinate_catalog:
            manifest_kwargs["source_evidence_kind"] = "legacy_conversion"
        manifest = manifest_builder(
            run_id=str(run_id),
            dimensions=benchmark_input.dimensions,
            storage_plan=plans,
            arrays=destination_arrays,
            source_evidence=source_evidence,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            selector_eligible=False,
            **manifest_kwargs,
        )
        run.attrs["run_manifest"] = manifest
        second_consolidation = consolidate_metadata_capture_expected_warnings(
            output_path
        )
        direct, consolidated = _metadata_maps(
            output_path,
            run_id=str(run_id),
            plans=plans,
        )
        errors = validate_canonical_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=destination_arrays,
        )
        if errors:
            raise RuntimeError(
                "Canonical shadow publication validation failed: " + "; ".join(errors)
            )
        receipt: dict[str, object] = {
            "schema_id": CANONICAL_DETECTION_SHADOW_RECEIPT_SCHEMA_ID,
            "schema_version": CANONICAL_DETECTION_SHADOW_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "output_path": str(output_path),
            "run_id": str(run_id),
            "storage_profile_id": plans.profile.profile_id,
            "run_manifest_digest": manifest["payload_digest"],
            "logical_content_digest": manifest["payload"]["logical_content"]["digest"],
            "logical_hashes": destination_hashes,
            "instance_key_policy": (
                "preserved_from_source"
                if preserve_source_instance_keys
                else "minted_from_canonical_values"
            ),
            "writes": writes,
            "consolidation": {
                "before_manifest": first_consolidation,
                "after_manifest": second_consolidation,
            },
            "publication_seconds": float(time.perf_counter() - started),
            "production_state_changes": [],
        }
        with (output_path / "shadow_publication_receipt.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                receipt,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        return CanonicalDetectionShadowPublication(
            output_path=output_path,
            run_id=str(run_id),
            dimensions=benchmark_input.dimensions,
            plans=plans,
            manifest=manifest,
            arrays=destination_arrays,
            receipt=receipt,
        )
    except Exception as exc:
        run.attrs["status"] = "failed"
        run.attrs["stage_selector_eligible"] = False
        run.attrs["shadow_failure"] = str(exc)
        raise


__all__ = [
    "CANONICAL_DETECTION_SHADOW_RECEIPT_SCHEMA_ID",
    "DEFAULT_CANONICAL_DETECTION_SHADOW_ROOT",
    "CanonicalDetectionShadowPublication",
    "canonical_detection_metadata_declaration_maps",
    "publish_legacy_canonical_detection_shadow",
    "require_safe_canonical_detection_shadow_destination",
    "validate_canonical_detection_shadow_publication",
]
