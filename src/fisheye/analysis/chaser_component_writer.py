"""Fail-closed staging boundary for maintained chaser component writers.

The historical component functions were useful payload builders, but they
opened the authoritative archive directly, deleted a same-name child when
``overwrite`` was requested, and updated ``latest`` pointers.  This module
turns those functions into private node-local builders and publishes only a
sealed component directory through the atomic chaser-component materializer.

The staging capability is intentionally process-local and identity checked.
Calling an unwrapped payload builder, exporting an unsealed local component,
or attempting to reuse an immutable component name fails closed.
"""

from __future__ import annotations

import copy
import functools
import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar, cast

import zarr

from fisheye.analysis.chaser_component_publication import (
    ChaserComponentContract,
    component_record_sha256,
    persist_chaser_component_manifest,
)
from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.analysis_workflows.materializers.chaser_component import (
    ChaserComponentPublishRequest,
    publish_sealed_chaser_component,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr_io import open_zarr_root


CHASER_COMPONENT_WRITER_RECEIPT_SCHEMA_ID = (
    "palette.chaser_component_writer_publication_receipt"
)
CHASER_COMPONENT_WRITER_RECEIPT_SCHEMA_VERSION = 1

_STAGING_CAPABILITY = object()
_Writer = TypeVar("_Writer", bound=Callable[..., str])


class ChaserComponentWriterError(ValueError):
    """Raised when a component bypasses the sealed publication boundary."""


class PublishedChaserComponentPath(str):
    """Backward-compatible component path carrying its detached receipt."""

    publication_receipt: Mapping[str, Any]

    def __new__(
        cls,
        value: str,
        *,
        publication_receipt: Mapping[str, Any],
    ) -> PublishedChaserComponentPath:
        instance = cast(PublishedChaserComponentPath, super().__new__(cls, value))
        instance.publication_receipt = copy.deepcopy(dict(publication_receipt))
        return instance


def require_chaser_component_staging_capability(value: object | None) -> None:
    """Authorize only the private payload-build invocation made below."""

    if value is not _STAGING_CAPABILITY:
        raise ChaserComponentWriterError(
            "Unsealed chaser component export is forbidden; use the public sealed "
            "component writer so publication passes through hidden-copy validation "
            "and atomic rename."
        )


def _controlled_component_identity(result: Any, *, family: str) -> tuple[str, str, str]:
    name = str(getattr(result, "component_name", "")).strip()
    run_name = str(getattr(result, "chaser_distance_run_name", "")).strip()
    run_path = str(getattr(result, "chaser_distance_run_path", "")).strip("/")
    if not family or "/" in family or family in {".", ".."}:
        raise ChaserComponentWriterError("Component family is not one controlled name.")
    if not name or "/" in name or name in {".", ".."}:
        raise ChaserComponentWriterError("Component name is not one controlled name.")
    expected_run_path = f"analysis/chaser_distance_runs/{run_name}"
    if not run_name or run_path != expected_run_path:
        raise ChaserComponentWriterError(
            "Chaser component result is not bound to one exact base run name/path."
        )
    return run_name, run_path, name


def _exact_lineage_contract(
    component: Any,
    *,
    component_family: str,
    component_name: str,
    semantic_schema_id: str,
    semantic_schema_version: int,
    method_id: str,
    method_version: str,
) -> ChaserComponentContract:
    attrs = dict(component.attrs)
    observed = (
        attrs.get("schema_id"),
        attrs.get("schema_version"),
        attrs.get("method"),
        attrs.get("method_version"),
    )
    expected = (
        semantic_schema_id,
        semantic_schema_version,
        method_id,
        method_version,
    )
    if observed != expected:
        raise ChaserComponentWriterError(
            "Staged chaser component semantic identity differs from its writer "
            f"declaration: expected={expected!r}, observed={observed!r}."
        )

    raw_lineage = attrs.get("lineage_payload_json")
    if not isinstance(raw_lineage, str) or not raw_lineage:
        raise ChaserComponentWriterError(
            "Staged chaser component lacks its exact run-lineage payload."
        )
    try:
        lineage = json.loads(raw_lineage)
    except json.JSONDecodeError as exc:
        raise ChaserComponentWriterError(
            "Staged chaser component lineage payload is not strict JSON."
        ) from exc
    if not isinstance(lineage, Mapping):
        raise ChaserComponentWriterError(
            "Staged chaser component lineage payload is not one object."
        )
    lineage_schema = lineage.get("analysis_schema")
    if not isinstance(lineage_schema, Mapping) or (
        lineage_schema.get("schema_id") != semantic_schema_id
        or lineage_schema.get("schema_version") != semantic_schema_version
    ):
        raise ChaserComponentWriterError(
            "Staged chaser component lineage has a different semantic schema."
        )
    if (
        lineage.get("method") != method_id
        or str(lineage.get("method_version")) != method_version
    ):
        raise ChaserComponentWriterError(
            "Staged chaser component lineage has a different method identity."
        )
    source_refs = lineage.get("source_refs")
    source_fingerprints = lineage.get("source_fingerprints")
    parameters = lineage.get("parameters")
    if not isinstance(source_refs, Mapping) or not isinstance(
        source_fingerprints, Mapping
    ):
        raise ChaserComponentWriterError(
            "Staged chaser component lineage lacks exact source authority objects."
        )
    if not isinstance(parameters, Mapping):
        raise ChaserComponentWriterError(
            "Staged chaser component lineage lacks one exact parameter object."
        )
    lineage_hash = attrs.get("lineage_hash")
    if (
        not isinstance(lineage_hash, str)
        or len(lineage_hash) != 64
        or any(character not in "0123456789abcdef" for character in lineage_hash)
    ):
        raise ChaserComponentWriterError(
            "Staged chaser component lacks its exact lineage digest."
        )
    return ChaserComponentContract(
        component_family=component_family,
        component_name=component_name,
        semantic_schema_id=semantic_schema_id,
        semantic_schema_version=semantic_schema_version,
        method_id=method_id,
        method_version=method_version,
        parameters={"lineage_parameters": dict(parameters)},
        source_authorities={
            "lineage_sha256": lineage_hash,
            "source_refs": dict(source_refs),
            "source_fingerprints": dict(source_fingerprints),
        },
    )


def _copy_base_attributes(
    source_root: Any,
    staging_root: zarr.Group,
    *,
    run_path: str,
) -> None:
    source_run = source_root[run_path]
    staging_run = staging_root.require_group(run_path)
    staging_run.attrs.update(json_attr_safe(dict(source_run.attrs)))


def sealed_chaser_component_writer(
    *,
    component_family: str,
    semantic_schema_id: str,
    semantic_schema_version: int,
    method_id: str,
    method_version: str,
) -> Callable[[_Writer], _Writer]:
    """Decorate one private payload builder with sealed atomic publication."""

    def decorate(payload_writer: _Writer) -> _Writer:
        @functools.wraps(payload_writer)
        def publish(
            zarr_path: Path | str,
            result: Any,
            *args: Any,
            **kwargs: Any,
        ) -> PublishedChaserComponentPath:
            source_zarr = Path(zarr_path).expanduser().resolve()
            run_name, run_path, component_name = _controlled_component_identity(
                result,
                family=component_family,
            )
            relative_path = f"{component_family}/{component_name}"

            source_root = open_zarr_root(source_zarr, mode="r")
            snapshot = load_chaser_distance_run(source_root, run_name=run_name)
            if snapshot.run_path != run_path:
                raise ChaserComponentWriterError(
                    "Verified chaser-distance base changed before component staging."
                )

            with tempfile.TemporaryDirectory(
                prefix=f"palette-{component_family}-{component_name}-"
            ) as temporary:
                staging_archive = Path(temporary) / "analysis.zarr"
                staging_root = zarr.open_group(
                    str(staging_archive),
                    mode="w",
                    zarr_format=3,
                    use_consolidated=False,
                )
                _copy_base_attributes(
                    source_root,
                    staging_root,
                    run_path=run_path,
                )
                staged_path = payload_writer(
                    staging_archive,
                    result,
                    *args,
                    _chaser_component_staging_capability=_STAGING_CAPABILITY,
                    **kwargs,
                )
                expected_path = f"{run_path}/{relative_path}"
                if str(staged_path).strip("/") != expected_path:
                    raise ChaserComponentWriterError(
                        "Private component builder returned a different component path."
                    )
                component = staging_root[expected_path]
                if not isinstance(component, zarr.Group):
                    raise ChaserComponentWriterError(
                        "Private component builder did not create one Zarr group."
                    )
                # A local staging path is operational, not scientific provenance.
                # Writers that expose it retain the authoritative archive path.
                if "zarr_path" in component.attrs:
                    component.attrs["zarr_path"] = str(source_zarr)
                contract = _exact_lineage_contract(
                    component,
                    component_family=component_family,
                    component_name=component_name,
                    semantic_schema_id=semantic_schema_id,
                    semantic_schema_version=semantic_schema_version,
                    method_id=method_id,
                    method_version=method_version,
                )
                manifest, manifest_digest = persist_chaser_component_manifest(
                    component,
                    snapshot=snapshot,
                    relative_path=relative_path,
                    contract=contract,
                )
                local_component_path = staging_archive / expected_path
                receipt = publish_sealed_chaser_component(
                    ChaserComponentPublishRequest(
                        source_zarr=source_zarr,
                        local_component_path=local_component_path,
                        base_run_name=run_name,
                        base_run_path=run_path,
                        relative_path=relative_path,
                        contract=contract,
                        activate_selector=False,
                    )
                )
                if receipt.get("component_manifest_sha256") != manifest_digest:
                    raise ChaserComponentWriterError(
                        "Atomic publisher returned a different component manifest digest."
                    )
                validation = receipt.get("final_validation")
                if not isinstance(validation, Mapping) or validation.get("valid") is not True:
                    raise ChaserComponentWriterError(
                        "Atomic publisher did not return successful final validation."
                    )
                writer_receipt = {
                    "schema_id": CHASER_COMPONENT_WRITER_RECEIPT_SCHEMA_ID,
                    "schema_version": CHASER_COMPONENT_WRITER_RECEIPT_SCHEMA_VERSION,
                    "component_path": expected_path,
                    "component_family": component_family,
                    "component_name": component_name,
                    "semantic_schema_id": semantic_schema_id,
                    "semantic_schema_version": semantic_schema_version,
                    "method_id": method_id,
                    "method_version": method_version,
                    "component_manifest_sha256": manifest_digest,
                    "payload_array_count": len(manifest["payload"]["arrays"]),
                    "payload_group_count": len(manifest["payload"]["groups"]),
                    "selector_eligible": False,
                    "validation": copy.deepcopy(dict(validation)),
                    "atomic_publication": copy.deepcopy(dict(receipt)),
                    "receipt_sha256": "",
                }
                writer_receipt["receipt_sha256"] = component_record_sha256(
                    {key: value for key, value in writer_receipt.items() if key != "receipt_sha256"}
                )
                return PublishedChaserComponentPath(
                    expected_path,
                    publication_receipt=writer_receipt,
                )

        setattr(publish, "__chaser_component_sealed_writer__", True)
        setattr(publish, "__chaser_component_family__", component_family)
        setattr(publish, "__chaser_component_semantic_schema_id__", semantic_schema_id)
        setattr(
            publish,
            "__chaser_component_semantic_schema_version__",
            semantic_schema_version,
        )
        return cast(_Writer, publish)

    return decorate


__all__ = [
    "CHASER_COMPONENT_WRITER_RECEIPT_SCHEMA_ID",
    "CHASER_COMPONENT_WRITER_RECEIPT_SCHEMA_VERSION",
    "ChaserComponentWriterError",
    "PublishedChaserComponentPath",
    "require_chaser_component_staging_capability",
    "sealed_chaser_component_writer",
]
