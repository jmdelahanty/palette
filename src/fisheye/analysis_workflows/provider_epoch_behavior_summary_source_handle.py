"""Strict handles for immutable protocol-semantic epoch behavior summaries."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    read_exact_immutable_child_validation_receipt,
)
from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    METHOD_ID,
    PARENT_PATH,
    SCHEMA_ID,
    SEMANTIC_EPOCH_BINDING_MODE,
    SEMANTIC_METHOD_VERSION,
    SEMANTIC_SCHEMA_VERSION,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.columnar import read_columnar_array_as_declared
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)

VERIFICATION_MODE = "receipt_bound_targeted_array_rehash_v1"
_HANDLE_SEAL = object()
_FORBIDDEN_NAMES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "selected",
        "authoritative",
        "authoritative_run",
        "default",
    }
)


class ProviderEpochBehaviorSummarySourceError(ValueError):
    """Raised when a semantic epoch summary is not exact immutable evidence."""


def _fail(message: str) -> None:
    raise ProviderEpochBehaviorSummarySourceError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}.")
    return value


def _run_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or value.casefold() in _FORBIDDEN_NAMES
        or "/" in value
        or "\\" in value
    ):
        _fail("run_name must identify one exact immutable child.")
    return value


def _archive(value: str | Path) -> Path:
    result = Path(value).expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {result}")
    return result


def _declarations(manifest: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    raw = manifest.get("array_declarations")
    if not isinstance(raw, list) or not raw:
        _fail("Semantic epoch summary manifest lacks array declarations.")
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        declaration = _mapping(item, field="array declaration")
        path = declaration.get("path")
        if (
            type(path) is not str
            or not path
            or path != path.strip("/")
            or path in result
            or set(declaration) != {"path", "dtype", "shape", "content_sha256"}
        ):
            _fail("Semantic epoch summary array declarations are inexact.")
        _digest(declaration.get("content_sha256"), field=f"{path} content digest")
        shape = declaration.get("shape")
        if not isinstance(shape, list) or any(
            type(item) is not int or item < 0 for item in shape
        ):
            _fail(f"{path} shape declaration is invalid.")
        if type(declaration.get("dtype")) is not str:
            _fail(f"{path} dtype declaration is invalid.")
        result[path] = declaration
    return MappingProxyType(result)


def _validate_manifest(
    attrs: Mapping[str, Any],
    *,
    run_path: str,
    run_name: str,
    expected_recording_id: str | None,
    expected_semantic_selection: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    manifest = _mapping(attrs.get(MANIFEST_ATTR), field="summary manifest")
    manifest_digest = _digest(
        attrs.get(MANIFEST_DIGEST_ATTR), field="summary manifest digest"
    )
    if canonical_json_sha256(_plain(manifest)) != manifest_digest:
        _fail("Semantic epoch summary manifest digest is stale.")
    scientific = _mapping(
        manifest.get("scientific_schema"), field="summary scientific schema"
    )
    parameters = _mapping(manifest.get("parameters"), field="summary parameters")
    sources = _mapping(manifest.get("sources"), field="summary sources")
    if (
        scientific.get("schema_id") != SCHEMA_ID
        or scientific.get("schema_version") != SEMANTIC_SCHEMA_VERSION
        or manifest.get("method_id") != METHOD_ID
        or manifest.get("method_version") != SEMANTIC_METHOD_VERSION
        or manifest.get("epoch_binding_mode") != SEMANTIC_EPOCH_BINDING_MODE
        or manifest.get("run_path") != run_path
        or attrs.get("run_path") != run_path
        or attrs.get("schema_id") != SCHEMA_ID
        or attrs.get("schema_version") != SEMANTIC_SCHEMA_VERSION
        or attrs.get("method_version") != SEMANTIC_METHOD_VERSION
        or attrs.get("epoch_binding_mode") != SEMANTIC_EPOCH_BINDING_MODE
        or attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get(RUN_NAME_ATTR) != run_name
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("production_authority") is not False
        or attrs.get("registry_update") is not False
        or attrs.get("selection") != "none"
        or manifest.get("selector_eligible") is not False
        or manifest.get("production_authority") is not False
        or manifest.get("registry_update") is not False
        or manifest.get("selection") != "none"
    ):
        _fail("Semantic epoch summary identity or safety state is invalid.")
    recording_id = manifest.get("recording_id")
    if type(recording_id) is not str or not recording_id:
        _fail("Semantic epoch summary recording identity is absent.")
    if attrs.get("recording_id") != recording_id or (
        expected_recording_id is not None and recording_id != expected_recording_id
    ):
        _fail("Semantic epoch summary belongs to another recording.")
    speed_level = parameters.get("physical_speed_level")
    if speed_level not in {"filtered", "smoothed", "averaged"}:
        _fail("Exact semantic epoch summaries prohibit raw or unknown speed levels.")
    if parameters.get("spatial_metrics") != (
        "omitted_requires_separately_selected_position_provider"
    ):
        _fail("Semantic epoch summary spatial omission policy is invalid.")
    if parameters.get("rate_denominator") != "valid_tracked_duration_s":
        _fail("Semantic epoch summary rate denominator is invalid.")
    if attrs.get("source_refs") != sources or attrs.get("parameters") != parameters:
        _fail("Semantic epoch summary manifest differs from its persisted bindings.")
    if attrs.get("source_refs_sha256") != canonical_json_sha256(_plain(sources)):
        _fail("Semantic epoch summary source binding digest is stale.")
    offer = _mapping(attrs.get("analysis_offer"), field="analysis offer")
    readiness = _mapping(offer.get("readiness"), field="analysis readiness")
    if (
        attrs.get("analysis_offer_sha256") != canonical_json_sha256(_plain(offer))
        or manifest.get("analysis_offer_sha256") != attrs.get("analysis_offer_sha256")
        or readiness.get("scientific") != "ready"
        or offer.get("selector_eligible") is not False
    ):
        _fail("Semantic epoch summary analysis offer is not scientifically ready.")
    semantic = _mapping(
        sources.get("protocol_semantic_selection"),
        field="protocol-semantic selection binding",
    )
    if (
        sources.get("epoch_binding_mode") != SEMANTIC_EPOCH_BINDING_MODE
        or tuple(semantic.get("roles", ())) != CHASER_WINDOW_ROLES
        or semantic.get("selector_eligible") is not False
        or semantic.get("production_authority") is not False
    ):
        _fail("Semantic epoch summary lacks the exact chaser role authority.")
    if expected_semantic_selection is not None:
        for field_name in ("run_path", "manifest_sha256"):
            if semantic.get(field_name) != expected_semantic_selection.get(field_name):
                _fail("Semantic epoch summary binds another semantic selection.")
    dimensions = _mapping(manifest.get("dimensions"), field="summary dimensions")
    if _integer(dimensions.get("n_epoch_rows"), field="summary epoch-row count") != len(
        CHASER_WINDOW_ROLES
    ):
        _fail("Semantic epoch summary dimensions are invalid.")
    for name in (
        "n_bout_rows",
        "n_bout_histogram_rows",
        "n_inter_bout_interval_histogram_rows",
    ):
        _integer(dimensions.get(name), field=f"summary {name}")
    payload = manifest.get("payload_digest")
    unsigned = {
        key: _plain(value) for key, value in manifest.items() if key != "payload_digest"
    }
    if _digest(payload, field="summary payload digest") != canonical_json_sha256(
        unsigned
    ):
        _fail("Semantic epoch summary payload digest is stale.")
    return _freeze(_plain(manifest)), _declarations(manifest)


@dataclass(frozen=True, slots=True, init=False)
class ProviderEpochBehaviorSummarySourceHandle:
    analysis_zarr: Path
    run_name: str
    run_path: str
    recording_id: str
    manifest: Mapping[str, Any] = field(repr=False)
    arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    verification_mode: str
    verified_array_paths: tuple[str, ...]
    receipt_digest: str | None
    metadata_evidence: Mapping[str, Any] = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _seal: object | None = None, **values: Any) -> None:
        if _seal is not _HANDLE_SEAL:
            raise TypeError(
                "Semantic epoch summary handles require their strict loader."
            )
        for name, value in values.items():
            if name in {"manifest", "metadata_evidence"}:
                value = _freeze(_plain(value))
            elif name == "arrays":
                copied: dict[str, np.ndarray] = {}
                for path, raw in value.items():
                    item = np.array(raw, copy=True, order="C")
                    item.setflags(write=False)
                    copied[path] = item
                value = MappingProxyType(copied)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _HANDLE_SEAL)

    @property
    def manifest_sha256(self) -> str:
        return canonical_json_sha256(_plain(self.manifest))

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])

    @property
    def deep_audited(self) -> bool:
        return self.verification_mode == "deep_audit"

    @property
    def verified_array_names(self) -> tuple[str, ...]:
        return self.verified_array_paths

    @property
    def semantic_selection(self) -> Mapping[str, Any]:
        return self.manifest["sources"]["protocol_semantic_selection"]

    def array(self, path: str) -> np.ndarray:
        try:
            return self.arrays[path]
        except KeyError as exc:
            raise KeyError(f"Unknown semantic epoch summary array {path!r}.") from exc

    def require_verified_arrays(self, paths: Collection[str]) -> None:
        required = {str(path) for path in paths}
        missing = required.difference(self.arrays)
        unverified = required.difference(self.verified_array_paths)
        if (
            missing
            or unverified
            or self.verification_mode
            not in {
                "deep_audit",
                VERIFICATION_MODE,
            }
        ):
            _fail(
                "Semantic epoch summary arrays lack accepted verification: "
                + ", ".join(sorted(missing or unverified or required))
            )


def load_provider_epoch_behavior_summary_source_handle(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
    expected_semantic_selection: Mapping[str, Any] | None = None,
    use_consolidated: bool = True,
    deep_audit: bool = False,
    direct_validation_receipt: str | Path | None = None,
    required_array_paths: Collection[str] | None = None,
) -> ProviderEpochBehaviorSummarySourceHandle:
    """Load one named schema-v2 summary without selector or legacy fallback."""

    if deep_audit and required_array_paths is not None:
        _fail("Deep audit cannot be combined with a targeted array roster.")
    if required_array_paths is not None and direct_validation_receipt is None:
        _fail("Targeted summary loading requires an exact validation receipt.")
    archive = _archive(analysis_zarr)
    name = _run_name(run_name)
    run_path = f"{PARENT_PATH}/{name}"
    if direct_validation_receipt is None:
        metadata = validate_direct_consolidated_subtree(
            archive, subtree_path=run_path
        ).to_json()
        root = open_zarr_root(archive, mode="r", use_consolidated=use_consolidated)
        run = root[run_path]
        receipt_digest = None
    else:
        receipt_path = Path(direct_validation_receipt).expanduser().resolve()
        receipt = read_exact_immutable_child_validation_receipt(
            receipt_path,
            expected_analysis_zarr=archive,
            expected_run_path=run_path,
            expected_recording_id=expected_recording_id,
            expected_manifest_attr=MANIFEST_ATTR,
            expected_manifest_digest_attr=MANIFEST_DIGEST_ATTR,
        )
        metadata = {
            "verification_mode": VERIFICATION_MODE,
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt["record_sha256"],
            "direct_metadata_inventory_sha256": receipt["direct_metadata_inventory"][
                "inventory_sha256"
            ],
            "archive_root_consolidated_metadata_reparse": False,
        }
        receipt_digest = str(receipt["record_sha256"])
        run = open_zarr_root(archive / run_path, mode="r", use_consolidated=False)
    attrs = dict(getattr(run, "attrs", {}))
    manifest, declarations = _validate_manifest(
        attrs,
        run_path=run_path,
        run_name=name,
        expected_recording_id=expected_recording_id,
        expected_semantic_selection=expected_semantic_selection,
    )
    if deep_audit:
        requested = set(declarations)
    elif required_array_paths is not None:
        requested = {str(path) for path in required_array_paths}
        unknown = requested.difference(declarations)
        if unknown:
            _fail(
                "Requested semantic epoch arrays are undeclared: "
                + ", ".join(sorted(unknown))
            )
    else:
        requested = set()
    arrays: dict[str, np.ndarray] = {}
    for path in sorted(requested):
        declaration = declarations[path]
        try:
            values = read_columnar_array_as_declared(
                run[path],
                expected_dtype=str(declaration["dtype"]),
                expected_shape=tuple(int(value) for value in declaration["shape"]),
            )
        except Exception as exc:
            raise ProviderEpochBehaviorSummarySourceError(
                f"Cannot read semantic epoch summary array {path!r}: {exc}"
            ) from exc
        if array_values_sha256(values) != declaration.get("content_sha256"):
            _fail(f"Semantic epoch summary array {path!r} changed.")
        arrays[path] = values
    return ProviderEpochBehaviorSummarySourceHandle(
        analysis_zarr=archive,
        run_name=name,
        run_path=run_path,
        recording_id=str(manifest["recording_id"]),
        manifest=manifest,
        arrays=arrays,
        verification_mode=(
            "deep_audit"
            if deep_audit
            else (
                VERIFICATION_MODE
                if required_array_paths is not None
                else "metadata_only"
            )
        ),
        verified_array_paths=tuple(sorted(arrays)),
        receipt_digest=receipt_digest,
        metadata_evidence=metadata,
        _seal=_HANDLE_SEAL,
    )


def validate_provider_epoch_behavior_summary_metadata(
    attrs: Mapping[str, Any],
    *,
    run_path: str,
    run_name: str,
    expected_recording_id: str | None = None,
    expected_semantic_selection: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate metadata-only discovery evidence and return its exact binding."""

    manifest, declarations = _validate_manifest(
        attrs,
        run_path=run_path,
        run_name=run_name,
        expected_recording_id=expected_recording_id,
        expected_semantic_selection=expected_semantic_selection,
    )
    sources = manifest["sources"]
    return _freeze(
        {
            "run_path": run_path,
            "manifest_sha256": canonical_json_sha256(_plain(manifest)),
            "payload_digest": manifest["payload_digest"],
            "source_protocol_semantic_selection": sources[
                "protocol_semantic_selection"
            ],
            "source_provider_motion": sources["provider_motion"],
            "source_swim_bouts": sources["swim_bouts"],
            "parameters": manifest["parameters"],
            "dimensions": manifest["dimensions"],
            "array_declaration_count": len(declarations),
        }
    )


__all__ = [
    "ProviderEpochBehaviorSummarySourceError",
    "ProviderEpochBehaviorSummarySourceHandle",
    "VERIFICATION_MODE",
    "load_provider_epoch_behavior_summary_source_handle",
    "validate_provider_epoch_behavior_summary_metadata",
]
