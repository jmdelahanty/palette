"""The only definition of an intake step being done.

Both probes read durable NFS evidence only: the organizer's intake state,
the immutable import receipts and (register) the registry's receipt
bindings, opened ``mode=ro``. They never call LSF and never read a runner
sentinel or the poller's state files. Each returns a verdict plus an
evidence digest that is identical on every replay of a finished delivery:

- import: sha256 over the sorted ``[zarr path, receipt sha256]`` pairs;
- register: sha256 over the sorted ``[zarr path, receipt sha256, dataset_id,
  identity_scope_id, identity_snapshot_id]`` rows (the binding ids).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.intake.delivery import (
    admission_mode,
    load_durable_state,
    validate_snapshot_sha,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

PROBE_IMPORT_SCHEMA = "palette.intake.probe_import.v1"
PROBE_REGISTER_SCHEMA = "palette.intake.probe_register.v1"


@dataclass(frozen=True)
class ProbeResult:
    schema: str
    snapshot_sha: str
    verdict: bool
    evidence_digest: str | None
    zarr_paths: tuple[str, ...] = ()
    receipt_sha256s: tuple[str, ...] = ()
    bindings: tuple[Mapping[str, str], ...] | None = None
    admission_mode: str = "unset"
    state: str | None = None
    reason: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "snapshot_sha": self.snapshot_sha,
            "verdict": self.verdict,
            "evidence_digest": self.evidence_digest,
            "zarr_paths": list(self.zarr_paths),
            "receipt_sha256s": list(self.receipt_sha256s),
            "admission_mode": self.admission_mode,
            "state": self.state,
            "reason": self.reason,
        }
        if self.bindings is not None:
            payload["bindings"] = [dict(binding) for binding in self.bindings]
        payload.update(self.extra)
        return payload


def import_evidence_digest(pairs: Sequence[tuple[str, str]]) -> str:
    return canonical_json_sha256(sorted([str(path), str(receipt)] for path, receipt in pairs))


def register_evidence_digest(rows: Sequence[Mapping[str, str]]) -> str:
    return canonical_json_sha256(
        sorted(
            [
                row["zarr_path"],
                row["receipt_sha256"],
                row["dataset_id"],
                row["identity_scope_id"],
                row["identity_snapshot_id"],
            ]
            for row in rows
        )
    )


def _plan_receipt_pairs(plan: Mapping[str, Any], receipts: Mapping[str, str]) -> list[tuple[str, str]]:
    from fisheye.utils.organize_transfer_recordings import parent_zarr_paths

    pairs = []
    for parent, zarr_path in zip(plan["parents"], parent_zarr_paths(plan)):
        pairs.append((str(zarr_path), receipts[parent["identity"]["recording_id"]]))
    return pairs


def import_probe_result(
    snapshot_sha: str,
    *,
    plan: Mapping[str, Any],
    receipts: Mapping[str, str],
    state: Mapping[str, Any],
) -> ProbeResult:
    """A true import verdict from receipts the organizer just verified."""

    pairs = _plan_receipt_pairs(plan, receipts)
    return ProbeResult(
        schema=PROBE_IMPORT_SCHEMA,
        snapshot_sha=snapshot_sha,
        verdict=True,
        evidence_digest=import_evidence_digest(pairs),
        zarr_paths=tuple(path for path, _ in pairs),
        receipt_sha256s=tuple(receipt for _, receipt in pairs),
        admission_mode=admission_mode(state),
        state=state.get("status"),
    )


def _false(schema: str, sha: str, reason: str, state: Mapping[str, Any] | None) -> ProbeResult:
    return ProbeResult(
        schema=schema,
        snapshot_sha=sha,
        verdict=False,
        evidence_digest=None,
        bindings=() if schema == PROBE_REGISTER_SCHEMA else None,
        admission_mode=admission_mode(state),
        state=(state or {}).get("status"),
        reason=reason,
    )


def probe_import(snapshot_sha: str, *, destination_root: Path) -> ProbeResult:
    """True when the delivery is retired and every planned receipt verifies.

    Verification is the organizer's own retirement gate
    (``_verify_parent_imports`` without a registry): sealed manifest, frame
    index, the live-verified import receipt with the parent's identity, and
    stimulus when the delivery's admission contract requires it.
    """

    from fisheye.utils.organize_transfer_recordings import _verify_parent_imports

    sha = validate_snapshot_sha(snapshot_sha)
    state = load_durable_state(destination_root, sha)
    if state is None:
        return _false(PROBE_IMPORT_SCHEMA, sha, "no durable intake state", None)
    if state.get("status") != "complete":
        return _false(PROBE_IMPORT_SCHEMA, sha, f"intake state is {state.get('status')!r}", state)
    plan = state["plan"]
    contract = state.get("admission_contract") or {}
    try:
        receipts = _verify_parent_imports(
            plan,
            registry_path=None,
            require_stimulus=bool(contract.get("require_stimulus", False)),
        )
    except Exception as exc:
        return _false(PROBE_IMPORT_SCHEMA, sha, f"import evidence does not verify: {exc}", state)
    if receipts != state.get("import_receipts"):
        return _false(PROBE_IMPORT_SCHEMA, sha, "verified receipts differ from the retired ones", state)
    return import_probe_result(sha, plan=plan, receipts=receipts, state=state)


def probe_register(
    snapshot_sha: str, *, destination_root: Path, registry: Path
) -> ProbeResult:
    """True when the import probe is true and the registry binds every receipt.

    Each zarr is read through the identity authority's read-only admission
    reader (``load_verified_registry_recording_import``, SQLite ``mode=ro``),
    which re-verifies the live identity, acquisition and receipt evidence.
    """

    from fisheye.registry.recording_identity_authority import (
        load_verified_registry_recording_import,
    )

    imported = probe_import(snapshot_sha, destination_root=destination_root)
    state = load_durable_state(destination_root, imported.snapshot_sha)
    if not imported.verdict:
        return _false(PROBE_REGISTER_SCHEMA, imported.snapshot_sha, f"import: {imported.reason}", state)
    rows = []
    try:
        for zarr_path, receipt_sha256 in zip(imported.zarr_paths, imported.receipt_sha256s):
            verified = load_verified_registry_recording_import(
                registry_path=Path(registry), zarr_path=Path(zarr_path)
            )
            if verified.receipt.receipt_sha256 != receipt_sha256:
                raise ValueError(f"registry binds another receipt for {zarr_path}")
            rows.append(
                {
                    "zarr_path": zarr_path,
                    "receipt_sha256": receipt_sha256,
                    "dataset_id": verified.identity.dataset_id,
                    "identity_scope_id": verified.identity.identity_scope_id,
                    "identity_snapshot_id": verified.identity.identity_snapshot_id,
                }
            )
    except Exception as exc:
        return _false(PROBE_REGISTER_SCHEMA, imported.snapshot_sha, f"registry: {exc}", state)
    return ProbeResult(
        schema=PROBE_REGISTER_SCHEMA,
        snapshot_sha=imported.snapshot_sha,
        verdict=True,
        evidence_digest=register_evidence_digest(rows),
        zarr_paths=imported.zarr_paths,
        receipt_sha256s=imported.receipt_sha256s,
        bindings=tuple(rows),
        admission_mode=imported.admission_mode,
        state=imported.state,
    )


__all__ = [
    "PROBE_IMPORT_SCHEMA",
    "PROBE_REGISTER_SCHEMA",
    "ProbeResult",
    "import_evidence_digest",
    "probe_import",
    "probe_register",
    "register_evidence_digest",
]
