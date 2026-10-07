"""``register_delivery``: admit one imported delivery into the registry.

Runs only on the designated registry writer host (ws1). All of the
delivery's Zarrs are synchronized in ONE shadow publication
(``shadow_synchronize_recording_imports``), so the registry receives the
whole delivery or none of it. Idempotent: a delivery whose register probe is
already true is not published again (no new backup, registry bytes
unchanged).

Operator note: the identity authority binds an import receipt only from a
checkout at the receipt's producer commit. A delivery imported by another
commit than this registrar's deployment is refused (65,
``registrar_commit_mismatch``) before any backup is made; the refusal JSON
names ``receipt_producer_git_sha``. Register that stranded delivery from a
deployment at that commit (``~/.palette/deployments/ops-<sha>``), e.g.
``scripts/py -m fisheye.intake register-delivery <snapshot_sha> --config ...``
run inside it. Retrying from the current deployment can never succeed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import socket
from typing import Iterator, Mapping

from fisheye.intake.delivery import (
    REGISTER_LOCK_KIND,
    claim,
    default_destination_root,
    load_durable_state,
    plan_recording_only,
    refuse_synthetic_registration,
    require_workstation_admission,
    validate_snapshot_sha,
)
from fisheye.intake.outcomes import IntakeRefused, RegistrarCommitMismatch
from fisheye.intake.probes import ProbeResult, probe_import, probe_register

DECIDED_BY = "fisheye.intake.register_delivery"


def normalize_host(name: str) -> str:
    """Lowercase, without a trailing root dot. The one host-identity rule."""

    return str(name).strip().rstrip(".").lower()


def is_writer_host(configured: str, current: str | None = None) -> bool:
    """Whether ``current`` (default: this host) is the configured writer.

    Hosts compare equal by full name when both are fully qualified, and by
    their first label when either is a short name, so ``delahantyj-ws1`` and
    ``delahantyj-ws1.hhmi.org`` name the same writer. This is the only place
    intake decides host identity.
    """

    expected = normalize_host(configured)
    actual = normalize_host(current if current is not None else socket.gethostname())
    if not expected or not actual:
        return False
    if "." in expected and "." in actual:
        return expected == actual
    return expected.split(".", 1)[0] == actual.split(".", 1)[0]


@dataclass(frozen=True)
class RegistryWriter:
    """Where and as whom the single-writer gateway publishes."""

    registry: Path
    writer_host: str
    writer_lock_path: Path
    shadow_temp_root: Path
    shadow_backup_dir: Path

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "RegistryWriter":
        missing = [
            key
            for key in ("registry", "writer_host", "writer_lock_path", "shadow_temp_root", "shadow_backup_dir")
            if not config.get(key)
        ]
        if missing:
            raise IntakeRefused(f"registry writer configuration is missing: {', '.join(missing)}")
        return cls(
            registry=Path(str(config["registry"])),
            writer_host=str(config["writer_host"]),
            writer_lock_path=Path(str(config["writer_lock_path"])),
            shadow_temp_root=Path(str(config["shadow_temp_root"])),
            shadow_backup_dir=Path(str(config["shadow_backup_dir"])),
        )

    def require_this_host(self) -> None:
        if not is_writer_host(self.writer_host):
            raise IntakeRefused(
                f"this host {socket.gethostname()!r} is not the registry writer {self.writer_host!r}"
            )


@contextmanager
def _gateway_environment(writer: RegistryWriter) -> Iterator[None]:
    """The gateway reads its single-writer configuration from the environment.

    Its own host check is an exact comparison, so it receives this host's
    exact name once :func:`is_writer_host` has accepted it.
    """

    from fisheye.registry.shadow_publish import (
        REGISTRY_SHADOW_BACKUP_DIR_ENV,
        REGISTRY_SHADOW_TEMP_ROOT_ENV,
        REGISTRY_WRITER_HOST_ENV,
        REGISTRY_WRITER_LOCK_PATH_ENV,
    )

    values = {
        REGISTRY_WRITER_HOST_ENV: socket.gethostname(),
        REGISTRY_WRITER_LOCK_PATH_ENV: str(writer.writer_lock_path),
        REGISTRY_SHADOW_TEMP_ROOT_ENV: str(writer.shadow_temp_root),
        REGISTRY_SHADOW_BACKUP_DIR_ENV: str(writer.shadow_backup_dir),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def register_delivery(
    snapshot_sha: str,
    *,
    writer: RegistryWriter,
    destination_root: Path | None = None,
    allow_synthetic: bool = False,
) -> ProbeResult:
    """Register every Zarr of one retired delivery in one publication.

    Refuses (65) off the writer host, for a synthetic ``data_origin`` in the
    stored plan (unless ``allow_synthetic`` and an isolated registry), and
    for a delivery recorded under the retired job-mode admission. Raises
    :class:`IntakeHeld` (75) when another registration of this delivery is
    live. Returns the true register probe.
    """

    from fisheye.registry.shadow_publish import (
        RegistryProducerCommitMismatch,
        shadow_synchronize_recording_imports,
    )
    from fisheye.shared.recording_import_receipt import (
        RecordingImportReceipt,
        recording_import_receipt_path,
    )

    sha = validate_snapshot_sha(snapshot_sha)
    writer.require_this_host()
    destination = Path(destination_root) if destination_root is not None else default_destination_root()
    state = load_durable_state(destination, sha)
    if state is None:
        raise RuntimeError(f"delivery sha256:{sha} has no durable intake state yet")
    plan = state["plan"]
    try:
        recording_only = plan_recording_only(plan)
        require_workstation_admission(state, require_stimulus=not recording_only)
        refuse_synthetic_registration(plan, registry=writer.registry, allow_synthetic=allow_synthetic)
    except (KeyError, TypeError) as exc:
        raise IntakeRefused(f"invalid stored plan: {exc}") from exc

    with claim(plan, kind=REGISTER_LOCK_KIND):
        registered = probe_register(sha, destination_root=destination, registry=writer.registry)
        if registered.verdict:
            return registered
        imported = probe_import(sha, destination_root=destination)
        if not imported.verdict:
            raise RuntimeError(f"delivery is not imported: {imported.reason}")
        imports = [
            (
                Path(zarr_path),
                RecordingImportReceipt.from_path(
                    recording_import_receipt_path(Path(zarr_path), receipt_sha256)
                ),
            )
            for zarr_path, receipt_sha256 in zip(imported.zarr_paths, imported.receipt_sha256s)
        ]
        try:
            with _gateway_environment(writer):
                # The gateway's preflight checks every receipt's producing
                # commit against this checkout before any backup is copied.
                shadow_synchronize_recording_imports(
                    canonical_registry=writer.registry, imports=imports, decided_by=DECIDED_BY
                )
        except RegistryProducerCommitMismatch as exc:
            raise RegistrarCommitMismatch(
                str(exc),
                code="registrar_commit_mismatch",
                details={
                    "receipt_producer_git_sha": exc.receipt_producer_git_sha,
                    "registrar_git_sha": exc.registrar_git_sha,
                    "registrar_git_dirty": exc.registrar_git_dirty,
                    "zarr_path": exc.zarr_path,
                },
            ) from exc
        registered = probe_register(sha, destination_root=destination, registry=writer.registry)
        if not registered.verdict:
            raise RuntimeError(f"published, but the register probe is false: {registered.reason}")
        return registered


__all__ = [
    "DECIDED_BY",
    "RegistryWriter",
    "is_writer_host",
    "normalize_host",
    "register_delivery",
]
