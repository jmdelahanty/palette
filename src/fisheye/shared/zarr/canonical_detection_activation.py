"""Shared canonical-v3 raw-detection selector activation writer.

This is the single owner of the ``detect_runs`` parent mutation that turns an
already-published, manifest-sealed selector-eligible canonical-v3 run into the
active raw-detection authority: ``latest``, ``latest_complete``, the
canonical authority contract, and the manifest digest.  After writing, it
reconsolidates the archive and re-verifies direct and consolidated metadata
before the activation is reported.  Rollback restores only parent attributes
this writer still owns.

Native candidate publication runs the writer inside the atomic publisher's
activation callback; legacy-conversion successors run it under the same
archive publication lock after independent successor validation.

Because consumers bind downstream products (refined detections, detect-quality
reports) to the raw detection *run name* they were derived from, this module
also owns the lineage-equivalence rule for an activated legacy-conversion
successor: ``canonical_detection_lineage_equivalent_runs`` returns the legacy
source and its activated successor as one equivalence class, established only
from the successor's validated sealed manifest evidence, never from names.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import zarr

from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR,
    CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3,
    CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR,
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CANONICAL_DETECTION_RUN_MANIFEST_ATTRIBUTE,
    validate_canonical_detection_publication,
    validate_canonical_detection_run_manifest,
    validate_legacy_detection_source_evidence,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    canonical_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr_helpers import reconsolidate_zarr_metadata
from fisheye.shared.zarr_io import open_zarr_root


CANONICAL_DETECTION_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)
CANONICAL_DETECTION_ACTIVATION_PARENT_ATTRS = (
    *CANONICAL_DETECTION_SELECTOR_ATTRS,
    CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR,
    CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR,
)
# Persisted policy labels: keep the historical native names byte-identical.
CANONICAL_DETECTION_ACTIVATION_CONSOLIDATION_POLICY = (
    "canonical_detection_v3_selector_activation_direct_consolidated_verified_v1"
)
CANONICAL_DETECTION_FAILED_ACTIVATION_REPAIR_POLICY = (
    "canonical_detection_v3_failed_activation_rollback_verified_v1"
)
# Run attr stamped by ``activate_canonical_detection_successor`` through
# ``run_attr_updates``; rollback restores its prior state.
CANONICAL_DETECTION_SUCCESSOR_ACTIVATION_RUN_ATTR = "production_selector_activation"
CANONICAL_DETECTION_SUCCESSOR_ACTIVATION_RUN_VALUE = "complete"


@dataclass
class CanonicalDetectionSelectorActivation:
    """Activate one manifest-sealed selector-eligible canonical-v3 run."""

    archive: Path
    run_id: str
    manifest: Mapping[str, Any]
    plans: Any
    run_attr_updates: Mapping[str, Any] = field(default_factory=dict)
    snapshot: dict[str, tuple[bool, Any]] | None = None
    run_snapshot: dict[str, tuple[bool, Any]] | None = None
    attempted: dict[str, Any] | None = None
    visibility_report: dict[str, Any] | None = None

    def _validate_visibility(self) -> dict[str, Any]:
        direct_root = zarr.open_group(
            str(self.archive), mode="r", zarr_format=3, use_consolidated=False
        )
        consolidated_root = zarr.open_group(
            str(self.archive), mode="r", zarr_format=3, use_consolidated=True
        )
        for label, root in (
            ("direct", direct_root),
            ("consolidated", consolidated_root),
        ):
            family = root["detect_runs"]
            run = family[self.run_id]
            if (
                family.attrs.get("latest") != self.run_id
                or family.attrs.get("latest_complete") != self.run_id
                or family.attrs.get(CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR)
                != CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
                or family.attrs.get(CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR)
                != self.manifest.get("payload_digest")
                or run.attrs.get("palette_run_completion_status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not True
                or dict(run.attrs.get("run_manifest") or {}) != dict(self.manifest)
            ):
                raise RuntimeError(
                    f"{label} canonical detection is not selected, complete, "
                    "and selector eligible."
                )
        direct, consolidated = canonical_detection_metadata_declaration_maps(
            self.archive,
            run_id=self.run_id,
            plans=self.plans,
        )
        arrays = {
            path: direct_root[f"detect_runs/{self.run_id}/{path}"]
            for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        }
        errors = validate_canonical_detection_publication(
            self.manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
        )
        if errors:
            raise RuntimeError(
                "Activated canonical detection validation failed: " + "; ".join(errors)
            )
        return {
            "policy": CANONICAL_DETECTION_ACTIVATION_CONSOLIDATION_POLICY,
            "run_id": self.run_id,
            "selectors": {
                "latest": self.run_id,
                "latest_complete": self.run_id,
            },
            "manifest_digest": self.manifest.get("payload_digest"),
        }

    def activate(self, _root: Any, family: Any, run: Any) -> None:
        payload = self.manifest.get("payload")
        publication = (
            payload.get("publication") if isinstance(payload, Mapping) else None
        )
        if (
            not isinstance(publication, Mapping)
            or publication.get("stage_selector_eligible") is not True
        ):
            raise RuntimeError(
                "Refusing to activate a canonical detection run whose sealed "
                "manifest is not selector eligible."
            )
        self.snapshot = {
            name: (name in family.attrs, copy.deepcopy(family.attrs.get(name)))
            for name in CANONICAL_DETECTION_ACTIVATION_PARENT_ATTRS
        }
        self.run_snapshot = {
            name: (name in run.attrs, copy.deepcopy(run.attrs.get(name)))
            for name in self.run_attr_updates
        }
        self.attempted = {
            "latest": self.run_id,
            "latest_complete": self.run_id,
            CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR: (
                CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
            ),
            CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR: self.manifest.get(
                "payload_digest"
            ),
        }
        family.attrs["latest"] = self.run_id
        family.attrs["latest_complete"] = self.run_id
        family.attrs[CANONICAL_DETECTION_AUTHORITY_CONTRACT_ATTR] = (
            CANONICAL_DETECTION_AUTHORITY_CONTRACT_V3
        )
        family.attrs[CANONICAL_DETECTION_AUTHORITY_DIGEST_ATTR] = self.manifest.get(
            "payload_digest"
        )
        if family.attrs.get("latest_pending") == self.run_id:
            del family.attrs["latest_pending"]
            self.attempted["latest_pending"] = None
        for name, value in self.run_attr_updates.items():
            run.attrs[name] = copy.deepcopy(value)
        run.attrs["stage_selector_eligible"] = True
        consolidation = reconsolidate_zarr_metadata(
            self.archive,
            policy=CANONICAL_DETECTION_ACTIVATION_CONSOLIDATION_POLICY,
            fail_on_error=True,
        )
        self.visibility_report = {
            **self._validate_visibility(),
            "consolidation": consolidation,
        }

    def rollback(self) -> None:
        if self.snapshot is None or self.attempted is None:
            return
        root = open_zarr_root(self.archive, mode="a")
        family = root["detect_runs"]
        for name, (present, value) in self.snapshot.items():
            attempted = self.attempted.get(name, object())
            current_present = name in family.attrs
            current_value = family.attrs.get(name)
            owned = (attempted is None and not current_present) or (
                attempted is not None and current_present and current_value == attempted
            )
            if not owned:
                continue
            if present:
                family.attrs[name] = copy.deepcopy(value)
            elif name in family.attrs:
                del family.attrs[name]
        if self.run_id in family:
            run = family[self.run_id]
            for name, (present, value) in (self.run_snapshot or {}).items():
                if present:
                    run.attrs[name] = copy.deepcopy(value)
                elif name in run.attrs:
                    del run.attrs[name]
            run.attrs["stage_selector_eligible"] = False

    def repair_failed_visibility(self, _target_path: Path) -> None:
        reconsolidate_zarr_metadata(
            self.archive,
            policy=CANONICAL_DETECTION_FAILED_ACTIVATION_REPAIR_POLICY,
            fail_on_error=True,
        )


def _archive_path_from(archive_or_root: Any) -> Path | None:
    """Return the on-disk archive root, or ``None`` for non-local groups."""

    if isinstance(archive_or_root, (str, Path)):
        return Path(archive_or_root).expanduser().resolve()
    if str(getattr(archive_or_root, "path", "") or "").strip("/"):
        return None
    store_root = getattr(getattr(archive_or_root, "store", None), "root", None)
    if store_root is None:
        return None
    return Path(str(store_root)).expanduser().resolve()


def _validated_activated_successor_source(
    archive: Path,
    family: Any,
    *,
    successor_id: str,
    recording_identity: str,
) -> str | None:
    """Return the legacy source bound by one activated successor, else ``None``.

    Every claim comes from the successor's sealed run manifest: the envelope
    and payload digest validate, it is a canonical-v3 ``legacy_conversion`` of
    this run id sealed selector eligible, its legacy source evidence validates
    and binds this archive's recording and the on-disk ``detect_runs/<source>``
    group, and the run carries the completed-activation markers that a failed
    activation rolls back.
    """

    try:
        run = family[successor_id]
    except KeyError:
        return None
    attrs = run.attrs
    manifest = attrs.get(CANONICAL_DETECTION_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        return None
    if validate_canonical_detection_run_manifest(manifest):
        return None
    payload = manifest.get("payload")
    if (
        manifest.get("schema_version")
        != CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        or not isinstance(payload, Mapping)
        or payload.get("run_id") != successor_id
        or payload.get("source_evidence_kind") != "legacy_conversion"
    ):
        return None
    publication = payload.get("publication")
    if (
        not isinstance(publication, Mapping)
        or publication.get("stage_selector_eligible") is not True
    ):
        return None
    if (
        attrs.get("stage_selector_eligible") is not True
        or attrs.get("palette_run_completion_status") != "complete"
        or attrs.get(CANONICAL_DETECTION_SUCCESSOR_ACTIVATION_RUN_ATTR)
        != CANONICAL_DETECTION_SUCCESSOR_ACTIVATION_RUN_VALUE
    ):
        return None
    evidence = payload.get("source_evidence")
    if not isinstance(evidence, Mapping) or validate_legacy_detection_source_evidence(
        evidence
    ):
        return None
    source_run_id = evidence.get("source_run_id")
    if (
        not isinstance(source_run_id, str)
        or source_run_id == successor_id
        or source_run_id not in family
        or evidence.get("recording_identity") != recording_identity
        or evidence.get("source_group_path")
        != str((archive / "detect_runs" / source_run_id).resolve())
    ):
        return None
    return source_run_id


def canonical_detection_lineage_equivalent_runs(
    archive_or_root: Any,
    run_id: str | None,
) -> frozenset[str]:
    """Return the raw detection run names content-equivalent to ``run_id``.

    Downstream products record the raw detection run name they were derived
    from.  After ``activate_canonical_detection_successor`` moves the
    selectors from a legacy run to its canonical-v3 conversion successor, a
    consumer that matches that recorded name against the selected run must
    treat the two names as one lineage.  The result always contains
    ``run_id``; it adds the legacy source when ``run_id`` is an activated
    successor whose sealed manifest validates, and adds every such validated
    activated successor when ``run_id`` is their legacy source.  Invalid,
    tampered, unactivated, or rolled-back successors, successors whose
    recorded source is absent, and non-local archives grant no equivalence.
    Metadata is read directly (unconsolidated) so a stale consolidated view
    cannot grant or hide equivalence.
    """

    name = str(run_id).strip() if isinstance(run_id, str) else ""
    if not name:
        return frozenset()
    result = {name}
    archive = _archive_path_from(archive_or_root)
    if archive is None or not (archive / "detect_runs").is_dir():
        return frozenset(result)
    try:
        root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
        family = zarr.open_group(
            str(archive / "detect_runs"), mode="r", use_consolidated=False
        )
        recording_identity = str(root.attrs.get("recording_id") or "").strip()
        if not recording_identity or name not in family:
            return frozenset(result)
        source = _validated_activated_successor_source(
            archive,
            family,
            successor_id=name,
            recording_identity=recording_identity,
        )
        if source is not None:
            result.add(source)
        for candidate in sorted(family.group_keys()):
            if candidate in result:
                continue
            manifest = family[candidate].attrs.get(
                CANONICAL_DETECTION_RUN_MANIFEST_ATTRIBUTE
            )
            payload = (
                manifest.get("payload") if isinstance(manifest, Mapping) else None
            )
            evidence = (
                payload.get("source_evidence")
                if isinstance(payload, Mapping)
                else None
            )
            if (
                not isinstance(evidence, Mapping)
                or evidence.get("source_run_id") != name
            ):
                continue
            if (
                _validated_activated_successor_source(
                    archive,
                    family,
                    successor_id=candidate,
                    recording_identity=recording_identity,
                )
                == name
            ):
                result.add(candidate)
    except (OSError, KeyError, TypeError, ValueError):
        return frozenset({name})
    return frozenset(result)


__all__ = [
    "CANONICAL_DETECTION_ACTIVATION_CONSOLIDATION_POLICY",
    "CANONICAL_DETECTION_ACTIVATION_PARENT_ATTRS",
    "CANONICAL_DETECTION_FAILED_ACTIVATION_REPAIR_POLICY",
    "CANONICAL_DETECTION_SELECTOR_ATTRS",
    "CANONICAL_DETECTION_SUCCESSOR_ACTIVATION_RUN_ATTR",
    "CANONICAL_DETECTION_SUCCESSOR_ACTIVATION_RUN_VALUE",
    "CanonicalDetectionSelectorActivation",
    "canonical_detection_lineage_equivalent_runs",
]
