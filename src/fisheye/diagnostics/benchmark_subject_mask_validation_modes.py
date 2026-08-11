"""Compare current subject-mask validation modes on one immutable fixture.

The driver copies its compact reference stores to node-local scratch, builds
one exhaustive source-validation receipt, then publishes the same raw and
refined logical arrays twice:

* ``reference_full_v1`` performs the deliberately expensive independent
  reference validation; and
* ``production_streaming_v1`` consumes the source receipt, hashes values while
  writing complete physical units, and performs only bounded reopen checks.

The resulting stores are benchmark-only and selector-ineligible.  This module
never writes to the reference fixture, a registry, or a production selector.
The production-streaming arm accepts an explicit physical-unit worker count so
fresh scratch runs can compare the serial and bounded-parallel write policies.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_runtime import (
    local_environment_manifest,
    peak_rss_bytes,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_METADATA_DIGEST_SCOPE,
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    SubjectMaskCorePublication,
    SubjectMaskCoreValidationMode,
    publish_selector_ineligible_subject_mask_core_snapshot,
    subject_mask_core_metadata_declaration_maps,
    validate_persisted_subject_mask_core_publication,
    validate_subject_mask_core_run_manifest,
)
from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    RawSubjectMaskSchema,
    RefinedSubjectMaskCoreSchema,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    build_reference_subject_mask_validation_receipt,
)

CANARY_SCHEMA_ID = "palette.subject_mask.validation_mode_canary"
CANARY_SCHEMA_VERSION = 2
REFERENCE_HANDOFF_SCHEMA_ID = "palette.subject_mask_cache_pipeline_benchmark"
REFERENCE_HANDOFF_SCHEMA_VERSION = 1
LEGACY_REFERENCE_METADATA_DIGEST_SCOPE = (
    "exact_run_group_and_array_declarations_redacting_only_run_manifest"
)
_SELECTOR_NAMES = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "approved_run",
)
_REMOTE_SCRATCH_PREFIXES = (Path("/groups"), Path("/nrs"), Path("/Volumes"))


@dataclass(frozen=True)
class _Case:
    name: str
    kind: str
    family: str
    source_store: Path
    source_run_id: str
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema
    arrays: Mapping[str, Any]
    source_manifest: Mapping[str, Any]
    dimensions: SubjectMaskDimensions
    components: SubjectMaskComponentRegistry
    threshold: float | None
    declared_logical_content_digest: str

    @property
    def source_run_path(self) -> str:
        return f"{self.family}/{self.source_run_id}"


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_new_node_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if any(
        resolved == prefix or prefix in resolved.parents
        for prefix in _REMOTE_SCRATCH_PREFIXES
    ):
        raise ValueError(f"Scratch must be node-local, got {resolved}.")
    if resolved.exists():
        raise FileExistsError(f"Scratch path already exists: {resolved}")
    resolved.mkdir(parents=True)
    return resolved


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _io_counters() -> dict[str, int] | None:
    path = Path("/proc/self/io")
    if not path.is_file():
        return None
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        if separator and value.strip().isdigit():
            result[name.strip()] = int(value.strip())
    return result


def _io_delta(
    before: Mapping[str, int] | None, after: Mapping[str, int] | None
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in sorted(set(before) | set(after))
    }


def _timed(function: Any) -> tuple[Any, dict[str, object]]:
    before_io = _io_counters()
    before_rss = peak_rss_bytes()
    started = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - started
    after_rss = peak_rss_bytes()
    return value, {
        "seconds": elapsed,
        "peak_rss_before_bytes": before_rss,
        "peak_rss_after_bytes": after_rss,
        "peak_rss_growth_bytes": max(0, after_rss - before_rss),
        "process_io_delta": _io_delta(before_io, _io_counters()),
    }


def _copytree(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    _, timing = _timed(lambda: shutil.copytree(source, destination))
    timing["source"] = str(source)
    timing["destination"] = str(destination)
    timing["storage"] = storage_stats(destination)
    return timing


def _validate_handoff(path: Path) -> dict[str, Any]:
    document = _strict_json(path)
    if set(document) != {"payload", "payload_digest"}:
        raise ValueError("Reference handoff envelope fields are not exact.")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Reference handoff payload is absent.")
    if document.get("payload_digest") != canonical_json_sha256(payload):
        raise ValueError("Reference handoff payload digest differs.")
    if (
        payload.get("schema_id") != REFERENCE_HANDOFF_SCHEMA_ID
        or payload.get("schema_version") != REFERENCE_HANDOFF_SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("selector_eligible") is not False
        or payload.get("registry_registered") is not False
        or payload.get("production_state_changes") != []
    ):
        raise ValueError("Reference handoff is not a completed isolated fixture.")
    outputs = payload.get("outputs")
    inputs = payload.get("inputs")
    if not isinstance(outputs, dict) or not isinstance(inputs, dict):
        raise ValueError("Reference handoff inputs or outputs are absent.")
    for name in ("raw", "refined", "quality"):
        item = outputs.get(name)
        if not isinstance(item, dict) or not str(item.get("run_id") or ""):
            raise ValueError(f"Reference handoff lacks its {name} output.")
    return document


def _manifest_dimensions(value: Mapping[str, Any]) -> SubjectMaskDimensions:
    return SubjectMaskDimensions(
        n_frames=int(value["n_frames"]),
        n_rois=int(value["n_rois"]),
        n_channels=int(value["n_channels"]),
        roi_height=int(value["roi_height"]),
        roi_width=int(value["roi_width"]),
    )


def _load_case(
    *,
    name: str,
    store: Path,
    run_id: str,
    declared_logical_content_digest: str,
) -> _Case:
    if name == "raw":
        kind = "raw_probability_uint8"
        family = "subject_mask_runs"
        schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema = (
            RAW_SUBJECT_MASK_UINT8_SCHEMA_V1
        )
    elif name == "refined":
        kind = "refined_dense_core"
        family = "refined_subject_masks_runs"
        schema = REFINED_SUBJECT_MASK_CORE_SCHEMA_V1
    else:
        raise ValueError(f"Unsupported canary case: {name}")
    run = zarr.open_group(
        str(store / family / run_id), mode="r", use_consolidated=False
    )
    manifest = run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Reference {name} run_manifest is absent.")
    manifest = deepcopy(dict(manifest))
    if manifest.get("payload_digest") != canonical_json_sha256(manifest.get("payload")):
        raise ValueError(f"Reference {name} run_manifest digest differs.")
    payload = manifest.get("payload")
    logical = payload.get("logical_content") if isinstance(payload, Mapping) else None
    document = logical.get("document") if isinstance(logical, Mapping) else None
    arrays_document = document.get("arrays") if isinstance(document, Mapping) else None
    if not isinstance(arrays_document, Mapping) or not arrays_document:
        raise ValueError(f"Reference {name} logical array inventory is absent.")
    logical_digest = logical.get("digest")
    if (
        logical.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or logical_digest != canonical_json_sha256(document)
        or logical_digest != str(declared_logical_content_digest)
    ):
        raise ValueError(f"Reference {name} logical-content binding differs.")
    dimensions_value = document.get("dimensions")
    components_value = document.get("components")
    if not isinstance(dimensions_value, Mapping) or not isinstance(
        components_value, Mapping
    ):
        raise ValueError(f"Reference {name} dimensions/components are absent.")
    labels = components_value.get("labels")
    if not isinstance(labels, list):
        raise ValueError(f"Reference {name} component labels are absent.")
    dimensions = _manifest_dimensions(dimensions_value)
    components = SubjectMaskComponentRegistry(tuple(str(label) for label in labels))
    paths = tuple(str(path) for path in arrays_document)
    arrays = {path: run[path] for path in paths}
    threshold = None
    if isinstance(schema, RawSubjectMaskSchema):
        logical_schema = payload.get("logical_schema")
        threshold_value = (
            logical_schema.get("threshold")
            if isinstance(logical_schema, Mapping)
            else None
        )
        if type(threshold_value) not in (int, float):
            raise ValueError("Reference raw threshold is absent.")
        threshold = float(threshold_value)
    return _Case(
        name=name,
        kind=kind,
        family=family,
        source_store=store,
        source_run_id=run_id,
        schema=schema,
        arrays=arrays,
        source_manifest=manifest,
        dimensions=dimensions,
        components=components,
        threshold=threshold,
        declared_logical_content_digest=str(declared_logical_content_digest),
    )


def _normalized_reference_metadata(
    declarations: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for path in sorted(declarations):
        value = metadata_without_empty_group_consolidation(
            declarations[path], path=path
        )
        value = dict(value)
        attributes = value.get("attributes")
        if isinstance(attributes, Mapping):
            attrs = dict(attributes)
            attrs.pop(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE, None)
            value["attributes"] = attrs
        result[path] = value
    return result


def _validate_reference_publication(case: _Case) -> dict[str, object]:
    """Validate current references and the one frozen pre-redaction revision."""

    manifest = deepcopy(dict(case.source_manifest))
    errors = validate_subject_mask_core_run_manifest(manifest)
    compatibility = "current_exact_v2"
    direct, consolidated = subject_mask_core_metadata_declaration_maps(
        case.source_store,
        family=case.family,
        run_id=case.source_run_id,
        manifest=manifest,
    )
    direct_document = _normalized_reference_metadata(direct)
    consolidated_document = _normalized_reference_metadata(consolidated)
    if direct_document != consolidated_document:
        raise ValueError(f"Reference {case.name} direct/consolidated metadata differ.")
    publication = manifest["payload"].get("publication")
    if not isinstance(publication, Mapping):
        raise ValueError(f"Reference {case.name} publication envelope is absent.")
    scope = publication.get("metadata_digest_scope")
    if errors:
        if tuple(errors) != ("subject-mask core publication is not exact",):
            raise ValueError(
                f"Reference {case.name} manifest is invalid: {'; '.join(errors)}"
            )
        if scope != LEGACY_REFERENCE_METADATA_DIGEST_SCOPE:
            raise ValueError(f"Reference {case.name} compatibility scope is unknown.")
        observed = canonical_json_sha256(direct_document)
        if observed != publication.get("metadata_digest"):
            raise ValueError(f"Reference {case.name} metadata digest differs.")
        currentized = deepcopy(manifest)
        currentized["payload"]["publication"][
            "metadata_digest_scope"
        ] = SUBJECT_MASK_CORE_METADATA_DIGEST_SCOPE
        currentized["payload_digest"] = canonical_json_sha256(currentized["payload"])
        currentized_errors = validate_subject_mask_core_run_manifest(currentized)
        if currentized_errors:
            raise ValueError(
                f"Reference {case.name} differs beyond its frozen metadata scope: "
                + "; ".join(currentized_errors)
            )
        compatibility = "frozen_v2_pre_transport_attr_redaction"
    elif scope != SUBJECT_MASK_CORE_METADATA_DIGEST_SCOPE:
        raise ValueError(f"Reference {case.name} current metadata scope differs.")
    write_receipt = manifest["payload"].get("write_receipt")
    if (
        not isinstance(write_receipt, Mapping)
        or write_receipt.get("validation_mode")
        != SubjectMaskCoreValidationMode.REFERENCE_FULL.value
    ):
        raise ValueError(f"Reference {case.name} is not reference-full evidence.")
    run = zarr.open_group(
        str(case.source_store / case.family / case.source_run_id),
        mode="r",
        use_consolidated=False,
    )
    if run.attrs.get("stage_selector_eligible") is not False:
        raise ValueError(f"Reference {case.name} is selector-eligible.")
    return {
        "compatibility": compatibility,
        "manifest_payload_digest": manifest["payload_digest"],
        "logical_content_digest": case.declared_logical_content_digest,
        "metadata_digest": publication["metadata_digest"],
        "array_count": len(case.arrays),
    }


def _source_crop_arrays(crop_store: Path, crop_run_id: str) -> dict[str, Any]:
    crop = zarr.open_group(
        str(crop_store / "crop_runs" / crop_run_id),
        mode="r",
        use_consolidated=False,
    )
    paths = ("instance_key", "source_acquisition_frame_index", "source_crop_xywh")
    missing = [path for path in paths if path not in crop]
    if missing:
        raise ValueError(f"Staged crop-v2 evidence lacks {missing!r}.")
    return {path: crop[path] for path in paths}


def _selector_isolation(publication: SubjectMaskCorePublication) -> dict[str, object]:
    root = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=False
    )
    parent = root[publication.family]
    run = parent[publication.run_id]
    present = [name for name in _SELECTOR_NAMES if name in parent.attrs]
    if present:
        raise ValueError(f"Canary parent unexpectedly has selectors: {present!r}.")
    if (
        root.attrs.get("benchmark_only") is not True
        or root.attrs.get("selector_eligible") is not False
        or root.attrs.get("registry_registered") is not False
        or parent.attrs.get("selector_eligible") is not False
        or run.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Canary selector/registry isolation differs.")
    return {
        "benchmark_only": True,
        "selector_eligible": False,
        "registry_registered": False,
        "parent_selectors_present": [],
    }


def _pair_equality(
    *,
    case: _Case,
    reference: SubjectMaskCorePublication,
    streaming: SubjectMaskCorePublication,
) -> dict[str, object]:
    reference_payload = reference.manifest["payload"]
    streaming_payload = streaming.manifest["payload"]
    for name in ("logical_schema", "storage_plan", "logical_content"):
        if reference_payload[name] != streaming_payload[name]:
            raise ValueError(f"{case.name} paired {name} differs.")
    digest = reference_payload["logical_content"]["digest"]
    if digest != case.declared_logical_content_digest:
        raise ValueError(f"{case.name} current output differs from frozen reference.")
    reference_direct, reference_consolidated = (
        subject_mask_core_metadata_declaration_maps(
            reference.output_path,
            family=reference.family,
            run_id=reference.run_id,
            manifest=reference.manifest,
        )
    )
    streaming_direct, streaming_consolidated = (
        subject_mask_core_metadata_declaration_maps(
            streaming.output_path,
            family=streaming.family,
            run_id=streaming.run_id,
            manifest=streaming.manifest,
        )
    )
    if _normalized_reference_metadata(
        reference_direct
    ) != _normalized_reference_metadata(reference_consolidated):
        raise ValueError(f"{case.name} reference direct/consolidated metadata differ.")
    if _normalized_reference_metadata(
        streaming_direct
    ) != _normalized_reference_metadata(streaming_consolidated):
        raise ValueError(f"{case.name} streaming direct/consolidated metadata differ.")
    paths = tuple(reference_payload["logical_content"]["document"]["arrays"])
    differing_array_metadata = [
        path for path in paths if reference_direct[path] != streaming_direct[path]
    ]
    if differing_array_metadata:
        raise ValueError(
            f"{case.name} paired array declarations differ: "
            f"{differing_array_metadata!r}."
        )
    for publication in (reference, streaming):
        persisted_errors = validate_persisted_subject_mask_core_publication(
            publication.output_path,
            family=publication.family,
            run_id=publication.run_id,
        )
        if persisted_errors:
            raise ValueError(
                f"{case.name} {publication.validation_mode.value} persisted gate "
                f"failed: {'; '.join(persisted_errors)}"
            )
    receipt_binding = streaming_payload["source"].get("validation_receipt")
    if not isinstance(receipt_binding, Mapping):
        raise ValueError(f"{case.name} streaming receipt binding is absent.")
    return {
        "exact_logical_content": True,
        "logical_content_digest": digest,
        "exact_logical_schema": True,
        "exact_storage_plan": True,
        "exact_array_metadata": True,
        "direct_consolidated_equivalence": True,
        "streaming_receipt_payload_digest": receipt_binding["payload_digest"],
    }


def _publish_pair(
    *,
    case: _Case,
    source_crop_arrays: Mapping[str, Any],
    output_root: Path,
    progress: dict[str, Any],
    progress_path: Path,
    physical_unit_workers: int,
) -> dict[str, object]:
    def checkpoint(phase: str) -> None:
        progress["phase"] = phase
        progress["updated_at_utc"] = utc_now()
        _write_json(progress_path, progress)
        print(f"[{case.name}] {phase}", flush=True)

    checkpoint("build_reference_source_validation_receipt")
    receipt, receipt_timing = _timed(
        lambda: build_reference_subject_mask_validation_receipt(
            kind=case.kind,
            source_run_path=case.source_run_path,
            source_manifest=case.source_manifest,
            schema=case.schema,
            arrays=case.arrays,
            dimensions=case.dimensions,
            components=case.components,
            threshold=case.threshold,
            source_crop_arrays=source_crop_arrays,
        )
    )
    checkpoint("publish_reference_full")
    reference, reference_timing = _timed(
        lambda: publish_selector_ineligible_subject_mask_core_snapshot(
            case.arrays,
            source_crop_arrays=source_crop_arrays,
            source_manifest=case.source_manifest,
            n_frames=case.dimensions.n_frames,
            components=case.components,
            destination=output_root / f"{case.name}_reference_full.zarr",
            run_id=f"{case.name}_reference_full_v1",
            kind=case.kind,
            source_run_path=case.source_run_path,
            threshold=case.threshold if case.threshold is not None else 0.5,
            created_by=CANARY_SCHEMA_ID,
            validation_mode=SubjectMaskCoreValidationMode.REFERENCE_FULL,
        )
    )
    checkpoint("publish_production_streaming")
    streaming, streaming_timing = _timed(
        lambda: publish_selector_ineligible_subject_mask_core_snapshot(
            case.arrays,
            source_crop_arrays=source_crop_arrays,
            source_manifest=case.source_manifest,
            n_frames=case.dimensions.n_frames,
            components=case.components,
            destination=output_root / f"{case.name}_production_streaming.zarr",
            run_id=f"{case.name}_production_streaming_v1",
            kind=case.kind,
            source_run_path=case.source_run_path,
            threshold=case.threshold if case.threshold is not None else 0.5,
            created_by=CANARY_SCHEMA_ID,
            validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
            source_validation_receipt=receipt,
            physical_unit_workers=physical_unit_workers,
        )
    )
    checkpoint("compare_pair")
    equality = _pair_equality(case=case, reference=reference, streaming=streaming)
    return {
        "dimensions": case.dimensions.as_manifest(),
        "components": case.components.as_manifest(),
        "source_validation_receipt": {
            "payload_digest": receipt["payload_digest"],
            "timing": receipt_timing,
        },
        "reference_full": {
            "path": str(reference.output_path),
            "run_id": reference.run_id,
            "timing": reference_timing,
            "internal_phase_seconds": dict(reference.phase_seconds),
            "storage": storage_stats(reference.output_path),
            "selector_isolation": _selector_isolation(reference),
        },
        "production_streaming": {
            "path": str(streaming.output_path),
            "run_id": streaming.run_id,
            "timing": streaming_timing,
            "internal_phase_seconds": dict(streaming.phase_seconds),
            "storage": storage_stats(streaming.output_path),
            "selector_isolation": _selector_isolation(streaming),
        },
        "equality": equality,
    }


def run_canary(
    *,
    reference_root: Path,
    scratch_root: Path,
    physical_unit_workers: int = 1,
) -> dict[str, object]:
    if type(physical_unit_workers) is not int or physical_unit_workers <= 0:
        raise ValueError("physical_unit_workers must be a positive integer.")
    reference = reference_root.expanduser().resolve()
    handoff_path = reference / "handoff_manifest.json"
    handoff = _validate_handoff(handoff_path)
    payload = handoff["payload"]
    scratch = _require_new_node_local_scratch(scratch_root)
    progress_path = scratch / "progress.json"
    progress: dict[str, Any] = {
        "schema_id": CANARY_SCHEMA_ID,
        "schema_version": CANARY_SCHEMA_VERSION,
        "status": "running",
        "phase": "stage_sources",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }
    _write_json(progress_path, progress)
    started = time.perf_counter()
    try:
        print("[canary] staging immutable sources to node-local scratch", flush=True)
        staged = scratch / "staged"
        staged.mkdir()
        stage_receipts = {
            "raw": _copytree(reference / "raw.zarr", staged / "raw.zarr"),
            "refined": _copytree(reference / "refined.zarr", staged / "refined.zarr"),
        }
        source_crop_path = Path(str(payload["inputs"]["source_crop_zarr"]))
        stage_receipts["crop"] = _copytree(source_crop_path, staged / "crop.zarr")
        shutil.copy2(handoff_path, staged / "handoff_manifest.json")
        if _sha256_file(staged / "handoff_manifest.json") != _sha256_file(handoff_path):
            raise RuntimeError("Staged handoff manifest hash differs.")
        outputs = payload["outputs"]
        raw = _load_case(
            name="raw",
            store=staged / "raw.zarr",
            run_id=str(outputs["raw"]["run_id"]),
            declared_logical_content_digest=str(
                outputs["raw"]["logical_content_digest"]
            ),
        )
        refined = _load_case(
            name="refined",
            store=staged / "refined.zarr",
            run_id=str(outputs["refined"]["run_id"]),
            declared_logical_content_digest=str(
                outputs["refined"]["logical_content_digest"]
            ),
        )
        if raw.dimensions.n_rois != refined.dimensions.n_rois or (
            raw.dimensions.n_frames != refined.dimensions.n_frames
        ):
            raise ValueError("Reference raw/refined recording dimensions differ.")
        crop_arrays = _source_crop_arrays(
            staged / "crop.zarr", str(payload["inputs"]["crop_run"])
        )
        progress["phase"] = "validate_reference_publications"
        progress["updated_at_utc"] = utc_now()
        _write_json(progress_path, progress)
        reference_validation = {
            case.name: _validate_reference_publication(case) for case in (raw, refined)
        }
        output_root = scratch / "outputs"
        output_root.mkdir()
        cases: dict[str, object] = {}
        for case in (raw, refined):
            cases[case.name] = _publish_pair(
                case=case,
                source_crop_arrays=crop_arrays,
                output_root=output_root,
                progress=progress,
                progress_path=progress_path,
                physical_unit_workers=physical_unit_workers,
            )
        result_payload: dict[str, object] = {
            "schema_id": CANARY_SCHEMA_ID,
            "schema_version": CANARY_SCHEMA_VERSION,
            "status": "complete",
            "result": "pass",
            "started_at_utc": progress["started_at_utc"],
            "completed_at_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "palette_commit": _git_commit(),
            "worktree_dirty": bool(
                subprocess.run(
                    ["git", "status", "--short"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
            "environment": {
                **local_environment_manifest(),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
            "execution": {
                "physical_unit_workers_requested": int(physical_unit_workers),
                "parallel_write_policy": (
                    "single_writer_v1_future_workers_require_disjoint_whole_shards"
                    if int(physical_unit_workers) == 1
                    else "bounded_threaded_disjoint_whole_physical_row_bands_v1"
                ),
            },
            "reference": {
                "root": str(reference),
                "handoff_manifest_sha256": _sha256_file(handoff_path),
                "handoff_payload_digest": handoff["payload_digest"],
                "source_palette_commit": payload["inputs"].get(
                    "resume_source_palette_commit"
                ),
                "validation": reference_validation,
                "stage_receipts": stage_receipts,
            },
            "scratch_root": str(scratch),
            "cases": cases,
            "quality_surface": {
                "paired_validation_mode": False,
                "reason": "quality publication has no alternate full/streaming mode",
                "reference_run_id": outputs["quality"]["run_id"],
                "logical_content_digest": outputs["quality"]["logical_content_digest"],
            },
            "production_state_changes": [],
        }
        result = {
            "payload": result_payload,
            "payload_digest": canonical_json_sha256(result_payload),
        }
        canonical_json_bytes(result)
        _write_json(scratch / "result.json", result)
        progress.update(
            {
                "status": "complete",
                "phase": "complete",
                "updated_at_utc": utc_now(),
                "result_payload_digest": result["payload_digest"],
            }
        )
        _write_json(progress_path, progress)
        return result
    except Exception as exc:
        progress.update(
            {
                "status": "failed",
                "phase": "failed",
                "updated_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(progress_path, progress)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Completed immutable cache-pipeline handoff directory.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        required=True,
        help="New node-local directory for staged inputs and paired outputs.",
    )
    parser.add_argument(
        "--physical-unit-workers",
        type=int,
        default=1,
        help=(
            "Bounded production-streaming physical row-band writer threads. "
            "Use separate immutable scratch roots for a 1/2/4 comparison."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_canary(
        reference_root=args.reference_root,
        scratch_root=args.scratch_root,
        physical_unit_workers=args.physical_unit_workers,
    )
    print(
        json.dumps(
            {
                "status": result["payload"]["status"],
                "result": result["payload"]["result"],
                "payload_digest": result["payload_digest"],
                "result_path": str(
                    args.scratch_root.expanduser().resolve() / "result.json"
                ),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
