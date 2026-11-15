"""Transfer-v2 intake: the one path by which new recordings enter Palette.

Five public entry points (docs/design/2026-10-07-intake-single-writer,
"Intake module and the runner contract"), also exposed as
``scripts/py -m fisheye.intake <subcommand>``:

- :func:`discover` - deliveries with intake work left, from live sealed
  markers and durable ``.transfer_intake`` states;
- :func:`import_delivery` - LSF side: organize, import, retire; no registry;
- :func:`register_delivery` - writer host only: one registry publication
  for all of a delivery's Zarrs;
- :func:`probe_import` / :func:`probe_register` - the only definition of
  done, from durable evidence, with replay-stable evidence digests.

The organizer, the recording import owner, the identity authority and the
shadow-publication gateway stay the owners of what they write; this package
sequences them and owns only the claims and the exit-code contract.

Rollout (deployments)
---------------------
Registration binds a receipt only from a checkout at the receipt's producer
commit, so the LSF import deployment (the poller's ``submit.repo``) and the
ws1 registrar deployment (the cron's ``~/.palette/deployments/ops-<sha>``)
must be the same commit. Move both in ONE step, and only when no
``<state_dir>/<key>.submitted`` lacks a terminal ``.registered``,
``.registration_refused`` or ``.import_failed``: otherwise deliveries
imported by the old commit would be refused (``registrar_commit_mismatch``)
by the new registrar. Recovery for a delivery refused that way: delete its
``<key>.registration_refused`` and re-run ``register_completed_imports`` (or
``python -m fisheye.intake register-delivery <sha> --config ...``) from a
deployment at the receipts' producer commit; the refusal JSON names it
(``receipt_producer_git_sha``).
"""

from fisheye.intake.discovery import discover
from fisheye.intake.importing import import_delivery
from fisheye.intake.outcomes import (
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_HELD,
    EXIT_REFUSED,
    IntakeHeld,
    IntakeRefused,
)
from fisheye.intake.probes import ProbeResult, probe_import, probe_register
from fisheye.intake.registration import RegistryWriter, register_delivery

__all__ = [
    "EXIT_DONE",
    "EXIT_FAILED",
    "EXIT_HELD",
    "EXIT_REFUSED",
    "IntakeHeld",
    "IntakeRefused",
    "ProbeResult",
    "RegistryWriter",
    "discover",
    "import_delivery",
    "probe_import",
    "probe_register",
    "register_delivery",
]
