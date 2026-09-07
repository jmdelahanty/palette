"""Row adapters for the collision-safe core-plus-chaser export profile.

The adapters reuse the existing five-grain and chaser table projections.  A
small routing context presents each projector only the source bundle it owns,
then rewrites generic row provenance to the enclosing composite bundle.  No
scientific value or authority identity is translated here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from fisheye.analysis_workflows.core_behavior_cohort_adapter import (
    CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
    CORE_BEHAVIOR_BUNDLE_METHOD_ID,
    CORE_BEHAVIOR_BUNDLE_STATUS,
)
from fisheye.analysis_workflows.core_chaser_composite_bundle import (
    CORE_CHASER_BUNDLE_ADAPTER_ID,
    read_core_chaser_composite_bundle,
)
from fisheye.analysis_workflows.validated_behavior_source_admission import (
    CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLES,
)
from fisheye.analysis_workflows.validated_behavior_cohort_adapters import sha256_file

from .validated_behavior_adapters import (
    _project_semantic_epoch_rows,
    build_phase_c_compact_row_extractors,
)
from .validated_behavior_cohort import ValidatedBehaviorBatchSource
from .validated_behavior_core_behavior_adapters import (
    build_core_behavior_row_extractors,
)
from .validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CORE_BEHAVIOR_TABLE_SPECS,
)
from .validated_behavior_core_chaser_contracts import (
    CORE_CHASER_EXTENSION_TABLE_SPECS,
)
from .validated_behavior_phase_b_adapters import (
    build_phase_b_dense_row_extractors,
)


class CoreChaserExportAdapterError(ValueError):
    """A composite bundle changed or cannot be routed without reinterpretation."""


def _fail(message: str) -> None:
    raise CoreChaserExportAdapterError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


class _CompositeRoutingContext:
    def __init__(
        self,
        plan: Mapping[str, Any],
        membership_member: Mapping[str, Any],
        bundle_member: Mapping[str, Any],
    ) -> None:
        if bundle_member.get("bundle_state") != "complete":
            _fail("Composite extraction requires one complete bundle member.")
        bundle_binding = _mapping(bundle_member.get("bundle"), field="bundle binding")
        if bundle_binding.get("adapter_id") != CORE_CHASER_BUNDLE_ADAPTER_ID:
            _fail("Composite extractor received another bundle adapter.")
        path = Path(str(bundle_binding.get("path"))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != bundle_binding.get("file_sha256"):
            _fail("Composite bundle file is absent or changed after admission.")
        bundle = read_core_chaser_composite_bundle(
            path,
            expected_analysis_zarr=membership_member["analysis_zarr"],
            expected_recording_id=membership_member["recording_id"],
            validate_current_sources=False,
        )
        if bundle["record_sha256"] != bundle_binding.get("record_sha256"):
            _fail("Composite bundle record differs from its bundle-set member.")
        self.plan = plan
        self.membership_member = membership_member
        self.bundle_member = bundle_member
        self.bundle_binding = bundle_binding
        self.bundle = bundle
        self._semantic_epoch_source: Mapping[str, Any] | None = None

    def core_plan(self) -> dict[str, Any]:
        plan = _plain(self.plan)
        profile = _mapping(plan.get("export_profile"), field="export profile")
        plan["export_profile"] = {
            **_plain(profile),
            "profile_id": CORE_BEHAVIOR_EXPORT_PROFILE_ID,
        }
        return plan

    def core_bundle_member(self) -> dict[str, Any]:
        receipts = {
            item["role"]: item for item in self.bundle["source_admission_receipts"]
        }
        core_receipts = [
            receipts[role]
            for role in CORE_BEHAVIOR_EXECUTION_ADMISSION_ROLES
            if role in receipts
        ]
        if len(core_receipts) != 1:
            _fail("Composite bundle lacks one installed core execution receipt.")
        report = core_receipts[0]
        capabilities = {
            key: _plain(self.bundle["capabilities"][key])
            for key in CORE_BEHAVIOR_CAPABILITY_KEYS
        }
        return {
            "recording_id": self.bundle_member["recording_id"],
            "analysis_zarr": self.bundle_member["analysis_zarr"],
            "bundle_state": "complete",
            "reason_code": None,
            "bundle": {
                "adapter_id": CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
                "path": report["path"],
                "file_sha256": report["file_sha256"],
                "record_sha256": report["record_sha256"],
                "schema_id": report["schema_id"],
                "schema_version": report["schema_version"],
                "method_id": CORE_BEHAVIOR_BUNDLE_METHOD_ID,
                "status": CORE_BEHAVIOR_BUNDLE_STATUS,
                "receipt_bindings": [report],
                "binding_inventory_sha256": self.bundle_binding[
                    "binding_inventory_sha256"
                ],
            },
            "capabilities": capabilities,
            "member_sha256": self.bundle_member["member_sha256"],
        }

    def chaser_bundle_member(self) -> dict[str, Any]:
        internal = _mapping(
            self.bundle.get("internal_capabilities"),
            field="internal capabilities",
        )
        capabilities: dict[str, Any] = {}
        for key, raw in internal.items():
            item = _mapping(raw, field=f"internal capability {key}")
            binding = (
                None
                if item.get("state") != "complete"
                else {
                    "scope": item["binding_scope"],
                    "key": item["binding_key"],
                }
            )
            capabilities[str(key)] = {
                "state": item["state"],
                "reason_code": item["reason_code"],
                "detail": item["detail"],
                "binding": binding,
            }
        return {
            "recording_id": self.bundle_member["recording_id"],
            "analysis_zarr": self.bundle_member["analysis_zarr"],
            "bundle_state": "complete",
            "reason_code": None,
            "bundle": _plain(self.bundle_binding),
            "capabilities": capabilities,
            "member_sha256": self.bundle_member["member_sha256"],
        }

    def semantic_epoch_source_binding(self) -> Mapping[str, Any]:
        """Project one exact composite semantic publication for Phase-C rows."""

        if self._semantic_epoch_source is not None:
            return self._semantic_epoch_source
        source_bindings = _mapping(
            self.bundle.get("source_bindings"), field="composite source bindings"
        )
        binding = _mapping(
            source_bindings.get("semantic_epochs"),
            field="composite semantic-epoch binding",
        )
        if binding.get("binding_type") != "exact_protocol_semantic_selection_v1":
            _fail("Composite has an unsupported semantic-epoch source binding.")
        source = _mapping(
            binding.get("source"), field="composite semantic-epoch source"
        )
        seal = _mapping(binding.get("sealed_by"), field="composite semantic-epoch seal")
        if set(seal) != {
            "run_path",
            "manifest_sha256",
            "payload_digest",
            "receipt_path",
            "receipt_sha256",
        }:
            _fail("Composite semantic-epoch seal field set is inexact.")
        child = _mapping(
            _mapping(
                self.bundle.get("scientific_child_bindings"),
                field="composite scientific child bindings",
            ).get("semantic_epochs"),
            field="composite semantic-epoch child",
        )
        if _plain(child) != _plain(seal):
            _fail("Composite semantic-epoch child differs from its source seal.")

        run_name = str(source.get("run_name"))
        run_path = str(seal.get("run_path"))
        receipt_path = Path(str(seal.get("receipt_path"))).expanduser().resolve()

        from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
            load_protocol_semantic_chaser_selection_source_handle,
        )

        handle = load_protocol_semantic_chaser_selection_source_handle(
            self.bundle["analysis_zarr"],
            run_name=run_name,
            expected_recording_id=str(self.bundle["recording_id"]),
            direct_validation_receipt=receipt_path,
        )
        metadata = _mapping(
            handle.metadata_equivalence,
            field="semantic source metadata evidence",
        )
        if (
            Path(handle.analysis_zarr).expanduser().resolve()
            != Path(str(self.bundle["analysis_zarr"])).expanduser().resolve()
            or handle.run_path != run_path
            or handle.recording_id != self.bundle["recording_id"]
            or handle.manifest_sha256 != seal.get("manifest_sha256")
            or handle.manifest.get("payload_digest") != seal.get("payload_digest")
            or _plain(handle.manifest) != _plain(source)
        ):
            _fail(
                "Current semantic publication differs from the sealed composite source."
            )
        if (
            metadata.get("receipt_sha256") != seal.get("receipt_sha256")
            or Path(str(metadata.get("receipt_path"))).expanduser().resolve()
            != receipt_path
        ):
            _fail("Current semantic receipt differs from the composite seal.")
        projected = _mapping(
            handle.source_binding(), field="projected semantic source binding"
        )
        if (
            projected.get("run_path") != run_path
            or projected.get("manifest_sha256") != seal.get("manifest_sha256")
            or projected.get("selection_identity_sha256")
            != source.get("selection_identity_sha256")
        ):
            _fail("Projected semantic source identity differs from its publication.")
        self._semantic_epoch_source = projected
        return self._semantic_epoch_source

    def semantic_epoch_common(self) -> dict[str, Any]:
        child = _mapping(
            self.bundle["scientific_child_bindings"].get("semantic_epochs"),
            field="composite semantic-epoch child",
        )
        return {
            "export_run_id": str(self.plan["export_run_id"]),
            "recording_id": str(self.bundle["recording_id"]),
            "membership_member_sha256": str(self.membership_member["member_sha256"]),
            "bundle_set_member_sha256": str(self.bundle_member["member_sha256"]),
            "bundle_record_sha256": str(self.bundle["record_sha256"]),
            "source_child_key": "semantic_epochs",
            "source_run_path": str(child["run_path"]),
            "source_manifest_sha256": str(child["manifest_sha256"]),
            "source_payload_sha256": str(child["payload_digest"]),
            "source_receipt_sha256": str(child["receipt_sha256"]),
        }

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "membership_member_sha256": str(self.membership_member["member_sha256"]),
            "bundle_set_member_sha256": str(self.bundle_member["member_sha256"]),
            "bundle_record_sha256": str(self.bundle_binding["record_sha256"]),
        }


class _LastCompositeRoutingContext:
    def __init__(self) -> None:
        self._key: tuple[str, str, str] | None = None
        self._context: _CompositeRoutingContext | None = None

    def get(
        self,
        plan: Mapping[str, Any],
        membership_member: Mapping[str, Any],
        bundle_member: Mapping[str, Any],
    ) -> _CompositeRoutingContext:
        key = (
            str(plan["plan_sha256"]),
            str(membership_member["member_sha256"]),
            str(bundle_member["member_sha256"]),
        )
        if key != self._key:
            self._context = _CompositeRoutingContext(
                plan, membership_member, bundle_member
            )
            self._key = key
        assert self._context is not None
        return self._context


def _rewrite_columns(
    columns: Mapping[str, Any], provenance: Mapping[str, str]
) -> dict[str, Any]:
    result = dict(columns)
    lengths = {
        len(value)
        for name, value in result.items()
        if name in provenance and hasattr(value, "__len__")
    }
    if len(lengths) != 1:
        _fail("Projected provenance columns do not share one row axis.")
    count = lengths.pop()
    for name, value in provenance.items():
        if name not in result:
            _fail(f"Projected rows lack required provenance column {name!r}.")
        result[name] = [value] * count
    return result


def _rewrite_result(value: Any, provenance: Mapping[str, str]) -> Any:
    if isinstance(value, ValidatedBehaviorBatchSource):

        def batches() -> Iterable[Mapping[str, Any]]:
            for columns in value.batches:
                yield _rewrite_columns(columns, provenance)

        return ValidatedBehaviorBatchSource(
            batches=batches(), zero_row_reason=value.zero_row_reason
        )
    if not isinstance(value, tuple) or len(value) != 2:
        _fail("Installed child extractor returned an unsupported result shape.")
    rows, reason = value
    if not isinstance(rows, (list, tuple)):
        _fail("Installed child extractor returned a non-row sequence.")
    rewritten = []
    for row in rows:
        item = dict(_mapping(row, field="projected row"))
        for name, replacement in provenance.items():
            if name not in item:
                _fail(f"Projected row lacks required provenance field {name!r}.")
            item[name] = replacement
        rewritten.append(item)
    return rewritten, reason


def _core_body_join_columns(columns: Mapping[str, Any]) -> dict[str, Any]:
    """Project the sentinel-bearing source row into one nullable FK column."""

    result = dict(columns)
    try:
        source_rows = list(result["body_source_row_id"])
        source_valid = list(result["body_source_row_valid"])
    except (KeyError, TypeError) as exc:
        raise CoreChaserExportAdapterError(
            "Body-relative projection lacks its source-row evidence."
        ) from exc
    if len(source_rows) != len(source_valid):
        _fail("Body-relative source-row evidence has unequal axes.")
    projected: list[int | None] = []
    for row_id, valid in zip(source_rows, source_valid, strict=True):
        normalized_row = int(row_id)
        if bool(valid):
            if normalized_row < 0:
                _fail("Valid body-relative rows require a nonnegative source row.")
            projected.append(normalized_row)
        else:
            if normalized_row != -1:
                _fail("Missing body-relative rows require the exact -1 sentinel.")
            projected.append(None)
    result["core_subject_shape_row_index"] = projected
    return result


def _with_core_body_join(value: Any) -> Any:
    if isinstance(value, ValidatedBehaviorBatchSource):

        def batches() -> Iterable[Mapping[str, Any]]:
            for columns in value.batches:
                yield _core_body_join_columns(columns)

        return ValidatedBehaviorBatchSource(
            batches=batches(), zero_row_reason=value.zero_row_reason
        )
    if not isinstance(value, tuple) or len(value) != 2:
        _fail("Body-relative extractor returned an unsupported result shape.")
    rows, reason = value
    if not isinstance(rows, (list, tuple)):
        _fail("Body-relative extractor returned a non-row sequence.")
    projected_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(_mapping(row, field="body-relative row"))
        source_row = int(item.get("body_source_row_id", -2))
        source_valid = bool(item.get("body_source_row_valid", False))
        if source_valid and source_row < 0:
            _fail("Valid body-relative rows require a nonnegative source row.")
        if not source_valid and source_row != -1:
            _fail("Missing body-relative rows require the exact -1 sentinel.")
        item["core_subject_shape_row_index"] = source_row if source_valid else None
        projected_rows.append(item)
    return projected_rows, reason


def _composite_semantic_epochs(
    context: _CompositeRoutingContext,
) -> tuple[list[dict[str, Any]], str | None]:
    return _project_semantic_epoch_rows(
        common=context.semantic_epoch_common(),
        source=context.semantic_epoch_source_binding(),
    )


def build_core_chaser_row_extractors() -> Mapping[str, Callable[..., Any]]:
    """Compose existing projectors behind one composite routing boundary."""

    core = build_core_behavior_row_extractors()
    compact = build_phase_c_compact_row_extractors()
    dense = build_phase_b_dense_row_extractors()
    cache = _LastCompositeRoutingContext()
    core_tables = set(CORE_BEHAVIOR_TABLE_SPECS).difference(
        {"cohort_recordings", "recording_bundles", "recording_capabilities"}
    )
    extension_tables = set(CORE_CHASER_EXTENSION_TABLE_SPECS)
    routes: dict[str, tuple[str, Callable[..., Any]]] = {}
    for table_name in sorted(core_tables):
        if table_name not in core:
            _fail(f"Core projector is absent for {table_name!r}.")
        routes[table_name] = ("core", core[table_name])
    for table_name in sorted(extension_tables):
        owners = [source for source in (compact, dense) if table_name in source]
        if len(owners) != 1:
            _fail(
                f"Chaser extension {table_name!r} must have exactly one installed "
                "projector."
            )
        routes[table_name] = (
            ("composite", _composite_semantic_epochs)
            if table_name == "semantic_epochs"
            else ("chaser", owners[0][table_name])
        )

    def wrap(
        table_name: str,
        owner: str,
        extractor: Callable[..., Any],
    ) -> Callable[..., Any]:
        def extract(
            plan: Mapping[str, Any],
            membership_member: Mapping[str, Any],
            bundle_member: Mapping[str, Any],
        ) -> Any:
            context = cache.get(plan, membership_member, bundle_member)
            if owner == "core":
                result = extractor(
                    context.core_plan(),
                    membership_member,
                    context.core_bundle_member(),
                )
            elif owner == "composite":
                result = extractor(context)
            else:
                result = extractor(
                    plan,
                    membership_member,
                    context.chaser_bundle_member(),
                )
                if table_name == "body_relative_samples":
                    result = _with_core_body_join(result)
            return _rewrite_result(result, context.provenance)

        return extract

    return MappingProxyType(
        {
            table_name: wrap(table_name, owner, extractor)
            for table_name, (owner, extractor) in routes.items()
        }
    )


__all__ = [
    "CoreChaserExportAdapterError",
    "build_core_chaser_row_extractors",
]
