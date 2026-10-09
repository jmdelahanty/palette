"""Check that this Python environment can run transfer-v2 intake.

Several intake contracts import ``jsonschema`` lazily, inside the function that
first validates a document, so importing the intake modules alone does not
prove the environment can run them. The cluster environment once lacked
``jsonschema`` and every real delivery failed in its LSF job (2026-10-07).

This check imports intake's third-party dependencies and builds every pinned
schema validator intake uses. It prints one line per problem and exits 1 if
anything is missing, 0 otherwise. The cluster deployment helper runs it on the
verification host.
"""

from __future__ import annotations

import importlib
import sys
from typing import Callable

# Third-party modules the transfer-v2 intake path imports, eagerly or lazily.
INTAKE_RUNTIME_MODULES = (
    "numpy",
    "h5py",
    "zarr",
    "pyarrow",
    "jsonschema",
    "jsonschema_rs",
)


def _validators() -> list[tuple[str, Callable[[], object]]]:
    from fisheye.shared import acquisition_crop_stream_ledger as ledger
    from fisheye.shared import recording_transfer_snapshot as snapshot
    from fisheye.shared import zebrobot_subject_reference as zebrobot
    from fisheye.shared.unified_h5 import citrus_subject_snapshot

    checks: list[tuple[str, Callable[[], object]]] = [
        ("transfer envelope", lambda: snapshot.envelope_validator()),
        ("frame identity proof", lambda: snapshot._frame_identity_proof_validator()),
        ("citrus subject snapshot", lambda: citrus_subject_snapshot._schema()),
    ]
    checks += [
        (f"orange {name}", lambda name=name: ledger._orange_validator(name))
        for name in ledger._ORANGE_SCHEMAS
    ]
    checks += [
        (f"orange subject reference v{version}", lambda version=version: zebrobot._schema(version))
        for version in zebrobot.SCHEMA_FILES
    ]
    return checks


def check() -> list[str]:
    """Return one message per missing dependency or unbuildable validator."""

    problems = []
    for name in INTAKE_RUNTIME_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report every failure
            problems.append(f"missing module {name}: {exc}")
    if problems:
        return problems
    for label, build in _validators():
        try:
            build()
        except Exception as exc:  # noqa: BLE001 - report every failure
            problems.append(f"cannot build {label} validator: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    del argv
    problems = check()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"intake runtime check FAILED ({sys.executable})", file=sys.stderr)
        return 1
    print(f"intake runtime check ok ({sys.executable})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
