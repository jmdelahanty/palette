#!/usr/bin/env python3
"""Create packed transfer artifacts for Palette Zarr archives.

Default mode is dry-run. Use --apply to write ``.tar.zst`` artifacts plus
manifest/checksum sidecars.
"""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs
import argparse
import hashlib
import json
import os
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import zstandard


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _top_level_entries(zarr_path: Path) -> list[str]:
    return sorted(path.name for path in zarr_path.iterdir())


def _count_included_files(zarr_path: Path, excluded_top_level: set[str]) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(zarr_path):
        root = Path(dirpath)
        rel = root.relative_to(zarr_path)
        if rel == Path("."):
            dirnames[:] = sorted(name for name in dirnames if name not in excluded_top_level)
        else:
            dirnames[:] = sorted(dirnames)
        for filename in sorted(filenames):
            path = root / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            file_count += 1
            total_bytes += stat.st_size
    return file_count, total_bytes


def _artifact_path_for(zarr_path: Path, out_dir: Optional[Path]) -> Path:
    filename = f"{zarr_path.name}.tar.zst"
    if out_dir is None:
        return zarr_path.with_name(filename)
    return out_dir / filename


def _manifest_path_for(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.name}.manifest.json")


def _checksum_path_for(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.name}.sha256")


def _normalize_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    return tarinfo


def _add_tree(
    tar: tarfile.TarFile,
    src_path: Path,
    *,
    arcname: str,
    root_path: Path,
    excluded_top_level: set[str],
) -> None:
    tar.add(src_path, arcname=arcname, recursive=False, filter=_normalize_tarinfo)
    if not src_path.is_dir():
        return
    for child in sorted(src_path.iterdir(), key=lambda item: item.name):
        if src_path == root_path and child.name in excluded_top_level:
            continue
        _add_tree(
            tar,
            child,
            arcname=f"{arcname}/{child.name}",
            root_path=root_path,
            excluded_top_level=excluded_top_level,
        )


def _write_tar_zst(
    zarr_path: Path,
    artifact_path: Path,
    *,
    compression_level: int,
    excluded_top_level: set[str],
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstandard.ZstdCompressor(level=compression_level)
    with artifact_path.open("wb") as raw_fh:
        with cctx.stream_writer(raw_fh) as zstd_fh:
            with tarfile.open(mode="w|", fileobj=zstd_fh, format=tarfile.PAX_FORMAT) as tar:
                _add_tree(
                    tar,
                    zarr_path,
                    arcname=zarr_path.name,
                    root_path=zarr_path,
                    excluded_top_level=excluded_top_level,
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PackOptions:
    apply: bool
    recursive: bool
    overwrite: bool
    compression_level: int
    out_dir: Optional[Path]
    excluded_top_level: tuple[str, ...]
    json: bool


@dataclass(frozen=True)
class PackRow:
    zarr_path: str
    artifact_path: str
    manifest_path: str
    checksum_path: str
    status: str
    reason: Optional[str] = None
    source_files: int = 0
    source_bytes: int = 0
    artifact_bytes: Optional[int] = None
    sha256: Optional[str] = None
    excluded_top_level: tuple[str, ...] = ()


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        size /= 1024.0
        if size < 1024:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} EB"


def _process_zarr(zarr_path: Path, options: PackOptions) -> PackRow:
    excluded_top_level = set(options.excluded_top_level)
    artifact_path = _artifact_path_for(zarr_path, options.out_dir)
    manifest_path = _manifest_path_for(artifact_path)
    checksum_path = _checksum_path_for(artifact_path)
    source_files, source_bytes = _count_included_files(zarr_path, excluded_top_level)

    if artifact_path.exists() and not options.overwrite:
        return PackRow(
            zarr_path=str(zarr_path),
            artifact_path=str(artifact_path),
            manifest_path=str(manifest_path),
            checksum_path=str(checksum_path),
            status="skipped_existing",
            reason=f"{artifact_path.name} already exists",
            source_files=source_files,
            source_bytes=source_bytes,
            excluded_top_level=tuple(sorted(excluded_top_level)),
        )

    if not options.apply:
        return PackRow(
            zarr_path=str(zarr_path),
            artifact_path=str(artifact_path),
            manifest_path=str(manifest_path),
            checksum_path=str(checksum_path),
            status="planned",
            source_files=source_files,
            source_bytes=source_bytes,
            excluded_top_level=tuple(sorted(excluded_top_level)),
        )

    try:
        _write_tar_zst(
            zarr_path,
            artifact_path,
            compression_level=options.compression_level,
            excluded_top_level=excluded_top_level,
        )
        digest = _sha256(artifact_path)
        artifact_bytes = artifact_path.stat().st_size
        top_level = _top_level_entries(zarr_path)
        excluded_present = sorted(name for name in excluded_top_level if name in top_level)
        included_top_level = sorted(name for name in top_level if name not in excluded_top_level)
        manifest_payload = {
            "version": 1,
            "created_utc": _utc_now_iso(),
            "source_zarr_path": str(zarr_path),
            "source_root_name": zarr_path.name,
            "source_files": source_files,
            "source_bytes": source_bytes,
            "artifact_path": str(artifact_path),
            "artifact_format": "tar.zst",
            "artifact_bytes": artifact_bytes,
            "artifact_sha256": digest,
            "compression": {
                "codec": "zstd",
                "level": options.compression_level,
            },
            "excluded_top_level_requested": sorted(excluded_top_level),
            "excluded_top_level_present": excluded_present,
            "included_top_level": included_top_level,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        checksum_path.write_text(f"{digest}  {artifact_path.name}\n", encoding="utf-8")
    except Exception as exc:
        return PackRow(
            zarr_path=str(zarr_path),
            artifact_path=str(artifact_path),
            manifest_path=str(manifest_path),
            checksum_path=str(checksum_path),
            status="error",
            reason=str(exc),
            source_files=source_files,
            source_bytes=source_bytes,
            excluded_top_level=tuple(sorted(excluded_top_level)),
        )

    return PackRow(
        zarr_path=str(zarr_path),
        artifact_path=str(artifact_path),
        manifest_path=str(manifest_path),
        checksum_path=str(checksum_path),
        status="updated",
        source_files=source_files,
        source_bytes=source_bytes,
        artifact_bytes=artifact_bytes,
        sha256=digest,
        excluded_top_level=tuple(sorted(excluded_top_level)),
    )


def _print_row(row: PackRow) -> None:
    artifact_bytes = _format_bytes(row.artifact_bytes) if row.artifact_bytes is not None else "-"
    source_bytes = _format_bytes(row.source_bytes)
    excluded = ",".join(row.excluded_top_level) if row.excluded_top_level else "-"
    reason = row.reason or "-"
    print(
        f"{row.status}\t{row.zarr_path}\tartifact={row.artifact_path}\t"
        f"source_files={row.source_files}\tsource_bytes={source_bytes}\t"
        f"artifact_bytes={artifact_bytes}\texcluded={excluded}\t{reason}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr paths or roots containing .zarr archives.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search roots for .zarr archives.")
    parser.add_argument("--out-dir", type=Path, help="Optional output directory for generated artifacts.")
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        help="Zstandard compression level for the tar.zst artifact (default: 3).",
    )
    parser.add_argument(
        "--exclude-top-level",
        action="append",
        default=[],
        help="Top-level group/file name to omit from the transfer artifact. Repeat as needed.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing artifact and sidecars.")
    parser.add_argument("--apply", action="store_true", help="Write artifact, manifest, and checksum sidecars.")
    parser.add_argument("--json", action="store_true", help="Emit JSON rows instead of text.")
    args = parser.parse_args(argv)

    roots = list(
        iter_filesystem_zarrs(
            args.paths,
            recursive=bool(args.recursive),
            pattern_policy="top_level",
            require_zarr_root=True,
            include_zarr_files=False,
        )
    )
    if not roots:
        if args.json:
            print(json.dumps({"rows": [], "summary": {"zarr_scanned": 0}}, indent=2))
        else:
            print("No matching zarr archives found.")
        return 1

    options = PackOptions(
        apply=bool(args.apply),
        recursive=bool(args.recursive),
        overwrite=bool(args.overwrite),
        compression_level=int(args.compression_level),
        out_dir=args.out_dir.expanduser() if args.out_dir is not None else None,
        excluded_top_level=tuple(sorted({str(value).strip() for value in args.exclude_top_level if str(value).strip()})),
        json=bool(args.json),
    )

    rows = [_process_zarr(zarr_path, options) for zarr_path in roots]
    summary = {
        "mode": "Applied" if options.apply else "Dry run",
        "zarr_scanned": len(rows),
        "updated": sum(1 for row in rows if row.status == "updated"),
        "planned": sum(1 for row in rows if row.status == "planned"),
        "skipped_existing": sum(1 for row in rows if row.status == "skipped_existing"),
        "errors": sum(1 for row in rows if row.status == "error"),
    }

    if options.json:
        print(json.dumps({"rows": [asdict(row) for row in rows], "summary": summary}, indent=2))
        return 0 if summary["errors"] == 0 else 1

    for row in rows:
        _print_row(row)
    print(
        "Transfer artifacts: mode={mode} zarr_scanned={zarr_scanned} errors={errors}".format(
            **summary,
        )
    )
    print(
        "Results: updated={updated} planned={planned} skipped_existing={skipped_existing}".format(
            **summary,
        )
    )
    if not options.apply:
        print("Dry run: add --apply to write .tar.zst artifacts plus manifest/checksum sidecars.")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
