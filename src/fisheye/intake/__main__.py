"""``scripts/py -m fisheye.intake <subcommand>``: the runner's only interface.

stdout carries exactly one JSON document; everything else (including output
of the import owners and their children) goes to stderr. Exit codes: 0 done
(probe true), 65 refused, 75 held by another live job, 1 retryable failure;
probes exit 0 for a true verdict and 1 for a false one. A usage error exits
2 (argparse), never 1, so a typo is never retried as transient.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from typing import Any, Callable, Iterator

from fisheye.intake.delivery import (
    CANONICAL_REGISTRY,
    default_destination_root,
    require_live_destination_root,
    validate_snapshot_sha,
)
from fisheye.intake.outcomes import (
    EXIT_DONE,
    EXIT_FAILED,
    EXIT_REFUSED,
    IntakeHeld,
    IntakeRefused,
    exit_code_for,
)

HELD_SCHEMA = "palette.intake.held.v1"
ERROR_SCHEMA = "palette.intake.error.v1"


@contextmanager
def _stdout_reserved_for_json() -> Iterator[Callable[[dict], None]]:
    """Point fd 1 (and sys.stdout) at stderr; yield a writer for the real stdout."""

    sys.stdout.flush()
    saved = os.dup(1)
    previous = sys.stdout
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        def emit(document: dict) -> None:
            os.write(saved, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode())

        yield emit
    finally:
        # Flush anything buffered for the original stdout while fd 1 still
        # points at stderr, so it can never land after the JSON document.
        for stream in {id(s): s for s in (previous, sys.__stdout__, sys.stderr) if s is not None}.values():
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        os.dup2(saved, 1)
        os.close(saved)
        sys.stdout = previous


def _writer(args: argparse.Namespace):
    from fisheye.intake.registration import RegistryWriter

    config: dict[str, Any] = {}
    if args.config is not None:
        try:
            loaded = json.loads(Path(args.config).read_text())
        except (OSError, ValueError) as exc:
            raise IntakeRefused(f"unreadable config {args.config}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise IntakeRefused("config must be a JSON object")
        config.update(loaded)
    for key in ("registry", "writer_host", "writer_lock_path", "shadow_temp_root", "shadow_backup_dir"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    return RegistryWriter.from_config(config)


def _sha_argument(value: str) -> str:
    try:
        return validate_snapshot_sha(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class _Terminated(BaseException):
    """SIGTERM (LSF bkill, runner cancel) inside the work: report and retry."""


def _terminate(signum, _frame):
    raise _Terminated(f"terminated by signal {signum}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fisheye.intake", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def destination(sub):
        sub.add_argument(
            "--destination-root", type=Path, default=None,
            help="Organized recordings root (default: $PALETTE_RECORDINGS_ROOT or the /groups store).",
        )

    sub = commands.add_parser("discover", help="List deliveries with intake work left.")
    sub.add_argument("--staging-dir", type=Path, required=True)
    destination(sub)
    sub.add_argument("--registry", type=Path, default=CANONICAL_REGISTRY,
                     help="Registry read (mode=ro) to drop retired deliveries whose receipts are bound "
                          "(default: the canonical registry).")
    sub.add_argument("--json", action="store_true", help="Accepted for symmetry; output is always JSON.")

    sub = commands.add_parser("import-delivery", help="Organize, import and retire one delivery (LSF).")
    sub.add_argument("snapshot_sha", type=_sha_argument)
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--resume-plan", type=Path, default=None)
    sub.add_argument("--session-dir", type=Path, default=None)
    sub.add_argument("--staging-dir", type=Path, default=None)
    destination(sub)
    sub.add_argument("--json", action="store_true")

    sub = commands.add_parser("register-delivery", help="Register one delivery (writer host only).")
    sub.add_argument("snapshot_sha", type=_sha_argument)
    destination(sub)
    sub.add_argument("--config", type=Path, default=None,
                     help="Poller/registrar JSON config holding the registry writer settings.")
    for key in ("registry", "writer_lock_path", "shadow_temp_root", "shadow_backup_dir"):
        sub.add_argument(f"--{key.replace('_', '-')}", dest=key, type=Path, default=None)
    sub.add_argument("--writer-host", dest="writer_host", default=None)
    sub.add_argument("--allow-synthetic-isolated-registry", action="store_true",
                     help="Admit producer-declared synthetic data into a non-canonical registry.")
    sub.add_argument("--json", action="store_true")

    for name in ("probe-import", "probe-register"):
        sub = commands.add_parser(name, help="Read-only verdict from durable evidence.")
        sub.add_argument("snapshot_sha", type=_sha_argument)
        destination(sub)
        if name == "probe-register":
            sub.add_argument("--registry", type=Path, default=CANONICAL_REGISTRY)
        sub.add_argument("--json", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> tuple[int, dict]:
    from fisheye.intake import (
        discover,
        import_delivery,
        probe_import,
        probe_register,
        register_delivery,
    )

    destination = require_live_destination_root(
        args.destination_root or default_destination_root()
    )
    if args.command == "discover":
        found = discover(args.staging_dir, destination, registry=args.registry)
        return EXIT_DONE, found.to_json()
    if args.command == "probe-import":
        result = probe_import(args.snapshot_sha, destination_root=destination)
        return (EXIT_DONE if result.verdict else EXIT_FAILED), result.to_json()
    if args.command == "probe-register":
        result = probe_register(args.snapshot_sha, destination_root=destination, registry=args.registry)
        return (EXIT_DONE if result.verdict else EXIT_FAILED), result.to_json()
    if args.command == "import-delivery":
        result = import_delivery(
            args.snapshot_sha,
            args.run_dir,
            args.resume_plan,
            destination_root=destination,
            session_dir=args.session_dir,
            staging_dir=args.staging_dir,
        )
        return EXIT_DONE, result.to_json()
    result = register_delivery(
        args.snapshot_sha,
        writer=_writer(args),
        destination_root=destination,
        allow_synthetic=args.allow_synthetic_isolated_registry,
    )
    return EXIT_DONE, result.to_json()


def _error_document(args: argparse.Namespace, exc: BaseException, code: int) -> dict:
    document = {
        "schema": ERROR_SCHEMA,
        "command": args.command,
        "snapshot_sha": getattr(args, "snapshot_sha", None),
        "exit_code": code,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }
    reason = getattr(exc, "code", None)
    if isinstance(exc, IntakeRefused) and reason:
        document.update(error=reason, message=str(exc))
    document.update(getattr(exc, "details", None) or {})
    return document


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)  # usage errors exit 2 here
    previous_handler = signal.signal(signal.SIGTERM, _terminate)
    try:
        return _main(args)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _main(args: argparse.Namespace) -> int:
    with _stdout_reserved_for_json() as emit:
        try:
            code, document = _run(args)
        except IntakeHeld as exc:
            print(f"held: {exc}", file=sys.stderr)
            code, document = exit_code_for(exc), {
                "schema": HELD_SCHEMA,
                "command": args.command,
                "snapshot_sha": getattr(args, "snapshot_sha", None),
                "lock_path": exc.lock_path,
                "holder": exc.holder,
                "error": str(exc),
            }
        except Exception as exc:
            code = exit_code_for(exc)
            label = "refused" if code == EXIT_REFUSED else "failed"
            print(f"{label}: {exc}", file=sys.stderr)
            if code != EXIT_REFUSED:
                traceback.print_exc(file=sys.stderr)
            document = _error_document(args, exc, code)
        except (SystemExit, KeyboardInterrupt, _Terminated) as exc:
            # Interrupted or exited from inside the work: still one JSON
            # document, and a retryable exit code.
            print(f"interrupted: {exc!r}", file=sys.stderr)
            code = EXIT_FAILED
            document = _error_document(args, exc, code)
            document["error"] = f"interrupted: {type(exc).__name__}: {exc}"
        # A SIGTERM now would truncate the one document the runner parses:
        # ignore it while emitting (main restores the previous handler).
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        emit(document)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
