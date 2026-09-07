"""Local test-fixture custody checks, never a production admission authority."""

from __future__ import annotations

import hashlib
from itertools import chain
import json
from pathlib import Path
import stat
import sys
import tempfile

FIXTURE_PREFIX = "palette-citrus-encoded-transfer-"
SESSION_ID = "synthetic-encoded-transfer-v2-20260906"
MAX_FIXTURE_FILES = 256
MAX_FIXTURE_BYTES = 64 * 1024 * 1024


def require_assertions_enabled() -> None:
    if sys.flags.optimize:
        raise RuntimeError(
            "Canary validation requires unoptimized Python; do not use -O or PYTHONOPTIMIZE."
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixture_root(path: Path, *, exists: bool) -> Path:
    absolute = path.absolute()
    if (
        absolute.parent != Path("/tmp")
        or not absolute.name.startswith(FIXTURE_PREFIX)
        or absolute != absolute.resolve(strict=exists)
    ):
        raise ValueError(
            "Expected a canonical, direct /tmp child with the synthetic fixture prefix."
        )
    return absolute


def new_fixture_root(output: Path | None = None) -> Path:
    require_assertions_enabled()
    if output is None:
        return Path(tempfile.mkdtemp(prefix=FIXTURE_PREFIX, dir="/tmp"))
    output = _fixture_root(output, exists=False)
    output.mkdir(exist_ok=False)
    return output


def validate_new_h5_path(path: Path) -> None:
    require_assertions_enabled()
    root = _fixture_root(path.parent.parent, exists=True)
    if path.parent.name != "source" or path.parent.is_symlink() or path.suffix != ".h5":
        raise ValueError("H5 builder requires a private synthetic source directory.")
    if path.exists() or path.is_symlink() or not (root / "source").is_dir():
        raise FileExistsError(
            "Refusing to replace an H5 or write outside a fresh synthetic source."
        )


def validate_source_pins(producer: Path, expected: dict[str, str]) -> dict[str, str]:
    actual = {}
    for relative, expected_digest in expected.items():
        path = producer / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Pinned Citrus source must be a regular file: {relative}")
        actual[relative] = file_sha256(path)
        if actual[relative] != expected_digest:
            raise ValueError(f"Pinned Citrus source bytes do not match: {relative}")
    return actual


def _private_tree(root: Path) -> None:
    count = total = 0
    for path in chain((root,), root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(
                f"Canary inputs must be private regular files, not links or special files: {path}"
            )
        count += 1
        total += info.st_size
        if count > MAX_FIXTURE_FILES or total > MAX_FIXTURE_BYTES:
            raise ValueError("Input exceeds the bounded synthetic fixture budget.")


def validate_synthetic_fixture(path: Path) -> tuple[Path, dict]:
    require_assertions_enabled()
    root = _fixture_root(path, exists=True)
    _private_tree(root)
    source, delivery = root / "source", root / "delivery"
    if not source.is_dir() or not delivery.is_dir() or not any(delivery.iterdir()):
        raise ValueError("A fresh synthetic source and nonempty delivery are required.")
    report = json.loads((root / "producer_smoke_report.json").read_bytes())
    if (
        report.get("status") != "actual_pinned_citrus_transfer_and_loadback_passed"
        or report.get("fixture_scope")
        != "synthetic_Orange_envelopes_real_H264_no_scientific_admission"
        or report.get("work_dir") != str(root)
        or report.get("source") != str(source)
        or report.get("destination") != str(delivery)
    ):
        raise ValueError(
            "Synthetic producer report does not bind this exact fixture scope and location."
        )
    provenance = json.loads(
        (source / "context/synthetic_fixture_provenance.json").read_bytes()
    )
    if (
        not (source / "fixture_notice.txt")
        .read_text()
        .startswith("SYNTHETIC TEST FIXTURE")
        or provenance.get("synthetic_session_id") != SESSION_ID
        or provenance.get("scientific_admission") is not False
        or provenance.get("orange_encoder_evidence") is not False
        or provenance.get("source_clock_evidence") != "synthetic_unqualified"
    ):
        raise ValueError("Source is not the declared unqualified synthetic fixture.")
    return root, report
