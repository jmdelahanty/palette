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
