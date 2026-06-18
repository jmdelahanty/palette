#!/usr/bin/env python3
"""Copy a Palette recording directory while transferring Zarr stores efficiently.

Default mode is dry-run. Use ``--apply`` to copy regular files with rsync and
copy each top-level ``zarr/*.zarr`` store through an uncompressed tar stream.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_RECORDING_ROOT = Path("/nvme1/recordings")
ZARR_MODE_CHOICES = ("tar-stream", "rsync", "tarball")
VALIDATION_CHOICES = ("quick", "checksum", "none")


@dataclass(frozen=True)
class ZarrStorePlan:
    source: str
    destination: str
    name: str
    mode: str
    tarball_path: Optional[str] = None


@dataclass(frozen=True)
class CopyPlan:
    source_recording: str
    destination_recording: str
    regular_rsync_command: tuple[str, ...]
    zarr_stores: tuple[ZarrStorePlan, ...]
    validation: str
    destination_exists: bool
    destination_nonempty: bool
    manifest_present: bool


@dataclass(frozen=True)
class CommandResult:
    step: str
    status: str
    command: tuple[str, ...] = ()
    returncode: int = 0
    detail: Optional[str] = None


@dataclass(frozen=True)
class CopyResult:
    plan: CopyPlan
    mode: str
    results: tuple[CommandResult, ...]
    ok: bool


def _path_with_trailing_slash(path: Path) -> str:
    return str(path) + os.sep


def _is_nonempty_dir(path: Path) -> bool:
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    except FileNotFoundError:
        return False
    return True


def _is_zarr_store(path: Path) -> bool:
    if not path.is_dir() or not path.name.endswith(".zarr"):
        return False
    return (path / "zarr.json").exists() or (path / ".zgroup").exists() or (path / ".zarray").exists()


def _resolve_source_recording(source: Path, recording_root: Path) -> Path:
    source = source.expanduser()
    if source.exists():
        return source.resolve()
    if not source.is_absolute() and len(source.parts) == 1:
        candidate = recording_root.expanduser() / source
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Recording source does not exist: {source}")


def _resolve_destination_recording(
    source_recording: Path,
    destination: Path,
    *,
    destination_is_recording_dir: bool,
) -> Path:
    destination = destination.expanduser()
    if destination_is_recording_dir:
        return destination.resolve() if destination.exists() else destination.absolute()
    return (destination / source_recording.name).resolve() if destination.exists() else (destination / source_recording.name).absolute()


def _ensure_not_recursive_copy(source: Path, destination: Path) -> None:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve() if destination.exists() else destination.absolute()
    if destination_resolved == source_resolved:
        raise ValueError("Destination recording path is the same as the source recording path.")
    try:
        if destination_resolved.is_relative_to(source_resolved):
            raise ValueError("Destination is inside the source recording; refusing recursive copy.")
    except AttributeError:
        if str(destination_resolved).startswith(str(source_resolved) + os.sep):
            raise ValueError("Destination is inside the source recording; refusing recursive copy.")


def _discover_zarr_stores(source_recording: Path) -> tuple[Path, ...]:
    zarr_dir = source_recording / "zarr"
    if not zarr_dir.is_dir():
        return ()
    return tuple(sorted(path for path in zarr_dir.iterdir() if _is_zarr_store(path)))


def _default_archive_dir(destination_recording: Path) -> Path:
    return destination_recording.with_name(f"{destination_recording.name}_zarr_tar")


def _rsync_regular_command(
    source_recording: Path,
    destination_recording: Path,
    *,
    rsync_bin: str,
) -> tuple[str, ...]:
    return (
        rsync_bin,
        "-a",
        "--info=progress2",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--exclude=zarr/*.zarr/",
        _path_with_trailing_slash(source_recording),
        _path_with_trailing_slash(destination_recording),
    )


def _rsync_zarr_command(source_zarr: Path, destination_zarr: Path, *, rsync_bin: str) -> tuple[str, ...]:
    return (
        rsync_bin,
        "-a",
        "--info=progress2",
        "--partial",
        "--partial-dir=.rsync-partial",
        _path_with_trailing_slash(source_zarr),
        _path_with_trailing_slash(destination_zarr),
    )


def build_copy_plan(
    source: Path,
    destination: Path,
    *,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    destination_is_recording_dir: bool = False,
    zarr_mode: str = "tar-stream",
    archive_dir: Optional[Path] = None,
    validation: str = "quick",
    rsync_bin: str = "rsync",
    allow_missing_manifest: bool = False,
) -> CopyPlan:
    if zarr_mode not in ZARR_MODE_CHOICES:
        raise ValueError(f"Unsupported zarr mode: {zarr_mode}")
    if validation not in VALIDATION_CHOICES:
        raise ValueError(f"Unsupported validation mode: {validation}")

    source_recording = _resolve_source_recording(source, recording_root)
    if not source_recording.is_dir():
        raise NotADirectoryError(f"Recording source is not a directory: {source_recording}")
    manifest_present = (source_recording / "recording_manifest.json").is_file()
    if not manifest_present and not allow_missing_manifest:
        raise FileNotFoundError(
            f"Missing recording_manifest.json in {source_recording}. "
            "Pass --allow-missing-manifest for legacy or partial folders."
        )

    destination_recording = _resolve_destination_recording(
        source_recording,
        destination,
        destination_is_recording_dir=destination_is_recording_dir,
    )
    _ensure_not_recursive_copy(source_recording, destination_recording)

    if zarr_mode == "tarball" and validation == "checksum":
        raise ValueError("--validate checksum requires unpacked Zarr stores; use --zarr-mode tar-stream or rsync.")

    zarr_stores = _discover_zarr_stores(source_recording)
    effective_archive_dir = archive_dir.expanduser() if archive_dir is not None else _default_archive_dir(destination_recording)
    zarr_plans: list[ZarrStorePlan] = []
    for store in zarr_stores:
        destination_store = destination_recording / "zarr" / store.name
        tarball_path = str(effective_archive_dir / f"{store.name}.tar") if zarr_mode == "tarball" else None
        zarr_plans.append(
            ZarrStorePlan(
                source=str(store),
                destination=str(destination_store),
                name=store.name,
                mode=zarr_mode,
                tarball_path=tarball_path,
            )
        )

    return CopyPlan(
        source_recording=str(source_recording),
        destination_recording=str(destination_recording),
        regular_rsync_command=_rsync_regular_command(source_recording, destination_recording, rsync_bin=rsync_bin),
        zarr_stores=tuple(zarr_plans),
        validation=validation,
        destination_exists=destination_recording.exists(),
        destination_nonempty=destination_recording.is_dir() and _is_nonempty_dir(destination_recording),
        manifest_present=manifest_present,
    )


def _format_command(args: Sequence[str]) -> str:
    return " ".join(str(arg) for arg in args)


def _run_checked(args: Sequence[str], *, step: str) -> CommandResult:
    completed = subprocess.run(list(args), check=False)
    if completed.returncode != 0:
        return CommandResult(
            step=step,
            status="error",
            command=tuple(str(arg) for arg in args),
            returncode=int(completed.returncode),
        )
    return CommandResult(
        step=step,
        status="ok",
        command=tuple(str(arg) for arg in args),
        returncode=0,
    )


def _copy_zarr_tar_stream(source_zarr: Path, destination_zarr: Path, *, tar_bin: str = "tar") -> CommandResult:
    destination_zarr.parent.mkdir(parents=True, exist_ok=True)
    command_text = (
        f"{tar_bin} -C {source_zarr.parent} -cf - {source_zarr.name} | "
        f"{tar_bin} -C {destination_zarr.parent} -xf -"
    )
    create = subprocess.Popen(
        [tar_bin, "-C", str(source_zarr.parent), "-cf", "-", source_zarr.name],
        stdout=subprocess.PIPE,
    )
    assert create.stdout is not None
    extract = subprocess.Popen(
        [tar_bin, "-C", str(destination_zarr.parent), "-xf", "-"],
        stdin=create.stdout,
    )
    create.stdout.close()
    extract_returncode = extract.wait()
    create_returncode = create.wait()
    if create_returncode != 0 or extract_returncode != 0:
        return CommandResult(
            step=f"zarr:{source_zarr.name}",
            status="error",
            command=(command_text,),
            returncode=create_returncode or extract_returncode,
            detail=f"tar create rc={create_returncode}, tar extract rc={extract_returncode}",
        )
    return CommandResult(
        step=f"zarr:{source_zarr.name}",
        status="ok",
        command=(command_text,),
        returncode=0,
    )


def _write_zarr_tarball(source_zarr: Path, tarball_path: Path, *, tar_bin: str = "tar", overwrite: bool = False) -> CommandResult:
    if tarball_path.exists() and not overwrite:
        return CommandResult(
            step=f"zarr:{source_zarr.name}",
            status="error",
            command=(),
            detail=f"Tarball already exists: {tarball_path}. Pass --overwrite-tarballs to replace it.",
        )
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    args = [tar_bin, "-C", str(source_zarr.parent), "-cf", str(tarball_path), source_zarr.name]
    return _run_checked(args, step=f"zarr:{source_zarr.name}")


def _quick_validate(plan: CopyPlan) -> CommandResult:
    source = Path(plan.source_recording)
    destination = Path(plan.destination_recording)
    failures: list[str] = []
    if (source / "recording_manifest.json").exists() and not (destination / "recording_manifest.json").exists():
        failures.append("missing recording_manifest.json")
    for row in plan.zarr_stores:
        if row.mode == "tarball":
            tarball_path = Path(row.tarball_path or "")
            if not tarball_path.is_file():
                failures.append(f"missing tarball {tarball_path}")
            continue
        src_zarr = Path(row.source)
        dst_zarr = Path(row.destination)
        if not dst_zarr.is_dir():
            failures.append(f"missing zarr directory {dst_zarr}")
            continue
        for marker in ("zarr.json", ".zgroup", ".zarray"):
            if (src_zarr / marker).exists() and not (dst_zarr / marker).exists():
                failures.append(f"missing {dst_zarr / marker}")
                break
    if failures:
        return CommandResult(step="validate:quick", status="error", detail="; ".join(failures[:10]))
    return CommandResult(step="validate:quick", status="ok")


def _checksum_validate(plan: CopyPlan, *, rsync_bin: str) -> CommandResult:
    args = [
        rsync_bin,
        "-a",
        "--dry-run",
        "--checksum",
        "--itemize-changes",
        "--delete",
        _path_with_trailing_slash(Path(plan.source_recording)),
        _path_with_trailing_slash(Path(plan.destination_recording)),
    ]
    completed = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        return CommandResult(
            step="validate:checksum",
            status="error",
            command=tuple(args),
            returncode=int(completed.returncode),
            detail=(completed.stderr or completed.stdout).strip() or None,
        )
    diff = completed.stdout.strip()
    if diff:
        return CommandResult(
            step="validate:checksum",
            status="error",
            command=tuple(args),
            detail=diff[:4000],
        )
    return CommandResult(step="validate:checksum", status="ok", command=tuple(args))


def _print_plan(plan: CopyPlan) -> None:
    print(f"source_recording={plan.source_recording}")
    print(f"destination_recording={plan.destination_recording}")
    if plan.destination_nonempty:
        print("destination_status=exists_nonempty")
    elif plan.destination_exists:
        print("destination_status=exists_empty")
    else:
        print("destination_status=missing")
    print(f"regular_files={_format_command(plan.regular_rsync_command)}")
    if not plan.zarr_stores:
        print("zarr_stores=0")
    for row in plan.zarr_stores:
        if row.mode == "tarball":
            print(f"zarr_store={row.name} mode=tarball source={row.source} tarball={row.tarball_path}")
        else:
            print(f"zarr_store={row.name} mode={row.mode} source={row.source} destination={row.destination}")
    print(f"validation={plan.validation}")


def _print_results(results: Sequence[CommandResult]) -> None:
    for result in results:
        detail = f" detail={result.detail}" if result.detail else ""
        command = f" command={_format_command(result.command)}" if result.command else ""
        print(f"{result.status}\t{result.step}{command}{detail}")


def _result_ok(results: Sequence[CommandResult]) -> bool:
    return all(result.status == "ok" for result in results)


def execute_copy_plan(
    plan: CopyPlan,
    *,
    resume: bool,
    zarr_mode: str,
    tar_bin: str,
    rsync_bin: str,
    overwrite_tarballs: bool,
) -> CopyResult:
    destination = Path(plan.destination_recording)
    if plan.destination_nonempty and not resume:
        result = CommandResult(
            step="preflight",
            status="error",
            detail="Destination recording directory is non-empty. Pass --resume to continue/update it.",
        )
        return CopyResult(plan=plan, mode="applied", results=(result,), ok=False)

    destination.mkdir(parents=True, exist_ok=True)
    results: list[CommandResult] = []

    results.append(_run_checked(plan.regular_rsync_command, step="regular-files"))
    if results[-1].status != "ok":
        return CopyResult(plan=plan, mode="applied", results=tuple(results), ok=False)

    for row in plan.zarr_stores:
        source_zarr = Path(row.source)
        destination_zarr = Path(row.destination)
        if zarr_mode == "tar-stream":
            result = _copy_zarr_tar_stream(source_zarr, destination_zarr, tar_bin=tar_bin)
        elif zarr_mode == "rsync":
            destination_zarr.parent.mkdir(parents=True, exist_ok=True)
            result = _run_checked(
                _rsync_zarr_command(source_zarr, destination_zarr, rsync_bin=rsync_bin),
                step=f"zarr:{source_zarr.name}",
            )
        elif zarr_mode == "tarball":
            result = _write_zarr_tarball(
                source_zarr,
                Path(row.tarball_path or ""),
                tar_bin=tar_bin,
                overwrite=overwrite_tarballs,
            )
        else:
            raise ValueError(f"Unsupported zarr mode: {zarr_mode}")
        results.append(result)
        if result.status != "ok":
            return CopyResult(plan=plan, mode="applied", results=tuple(results), ok=False)

    if plan.validation == "quick":
        results.append(_quick_validate(plan))
    elif plan.validation == "checksum":
        quick = _quick_validate(plan)
        results.append(quick)
        if quick.status == "ok":
            results.append(_checksum_validate(plan, rsync_bin=rsync_bin))

    return CopyResult(plan=plan, mode="applied", results=tuple(results), ok=_result_ok(results))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help=(
            "Recording directory to copy, or a recording name under --recording-root "
            f"(default {DEFAULT_RECORDING_ROOT})."
        ),
    )
    parser.add_argument(
        "destination",
        type=Path,
        help=(
            "Destination parent directory by default. The copied recording lands at "
            "<destination>/<recording_name>. Use --destination-is-recording-dir for an exact target path."
        ),
    )
    parser.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    parser.add_argument("--destination-is-recording-dir", action="store_true")
    parser.add_argument(
        "--zarr-mode",
        choices=ZARR_MODE_CHOICES,
        default="tar-stream",
        help="How to transfer top-level zarr/*.zarr stores. Default: tar-stream.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Output directory for --zarr-mode tarball. Defaults next to the destination recording.",
    )
    parser.add_argument(
        "--validate",
        choices=VALIDATION_CHOICES,
        default="quick",
        help="Post-copy validation. checksum runs an expensive rsync checksum dry-run.",
    )
    parser.add_argument("--rsync-bin", default="rsync")
    parser.add_argument("--tar-bin", default="tar")
    parser.add_argument("--resume", action="store_true", help="Allow copying into a non-empty destination recording directory.")
    parser.add_argument("--overwrite-tarballs", action="store_true", help="Replace existing tarballs in --zarr-mode tarball.")
    parser.add_argument("--allow-missing-manifest", action="store_true", help="Allow legacy/partial folders without recording_manifest.json.")
    parser.add_argument("--apply", action="store_true", help="Execute the copy. Default is dry-run planning only.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if shutil.which(args.rsync_bin) is None:
        parser.error(f"rsync binary not found: {args.rsync_bin}")
    if args.zarr_mode in {"tar-stream", "tarball"} and shutil.which(args.tar_bin) is None:
        parser.error(f"tar binary not found: {args.tar_bin}")

    try:
        plan = build_copy_plan(
            args.source,
            args.destination,
            recording_root=args.recording_root,
            destination_is_recording_dir=bool(args.destination_is_recording_dir),
            zarr_mode=args.zarr_mode,
            archive_dir=args.archive_dir,
            validation=args.validate,
            rsync_bin=args.rsync_bin,
            allow_missing_manifest=bool(args.allow_missing_manifest),
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json and not args.apply:
        print(json.dumps({"mode": "dry-run", "plan": asdict(plan)}, indent=2))
        return 0
    if not args.apply:
        _print_plan(plan)
        print("Dry run: add --apply to copy the recording.")
        return 0

    result = execute_copy_plan(
        plan,
        resume=bool(args.resume),
        zarr_mode=args.zarr_mode,
        tar_bin=args.tar_bin,
        rsync_bin=args.rsync_bin,
        overwrite_tarballs=bool(args.overwrite_tarballs),
    )
    if args.json:
        print(
            json.dumps(
                {
                    "mode": result.mode,
                    "ok": result.ok,
                    "plan": asdict(result.plan),
                    "results": [asdict(row) for row in result.results],
                },
                indent=2,
            )
        )
    else:
        _print_results(result.results)
        print(f"copy_status={'ok' if result.ok else 'error'}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
