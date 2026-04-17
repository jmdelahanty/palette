
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py
import numpy as np

from .models import FileInfo

H5_PATTERNS = ("*.h5", "*.hdf5", "*.h5.bak", "*.hdf5.bak")


def is_h5_path(path: Path) -> bool:
    lower = path.name.lower()
    return path.is_file() and lower.endswith((".h5", ".hdf5", ".h5.bak", ".hdf5.bak"))


def iter_h5_paths(paths: Iterable[Path], *, recursive: bool) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if is_h5_path(path):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
            continue
        if not path.exists() or not path.is_dir():
            continue
        candidates: list[Path] = []
        for pattern in H5_PATTERNS:
            if recursive:
                candidates.extend(path.rglob(pattern))
            else:
                candidates.extend(path.glob(pattern))
        for candidate in sorted(candidates):
            if not is_h5_path(candidate):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def resolve_h5_input(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if is_h5_path(candidate):
        return candidate.resolve()
    if not candidate.exists() or not candidate.is_dir():
        return candidate.resolve()

    preferred: list[Path] = []
    for pattern in H5_PATTERNS:
        preferred.extend(candidate.glob(f"raw/{pattern}"))
    if preferred:
        return sorted(path.resolve() for path in preferred if is_h5_path(path))[0]

    direct: list[Path] = []
    for pattern in H5_PATTERNS:
        direct.extend(candidate.glob(pattern))
    if direct:
        return sorted(path.resolve() for path in direct if is_h5_path(path))[0]

    recursive_paths = list(iter_h5_paths([candidate], recursive=True))
    if recursive_paths:
        return recursive_paths[0]
    return candidate.resolve()


def classify_h5_source(path: Path | str) -> str:
    parts = {part.lower() for part in Path(path).parts}
    if "raw" in parts:
        return "raw"
    return "other"


def infer_recording_root(path: Path | str) -> str:
    candidate = Path(path)
    if candidate.parent.name.lower() == "raw" and candidate.parent.parent != candidate.parent:
        return str(candidate.parent.parent)
    return str(candidate.parent)


def build_file_info(path: Path | str) -> FileInfo:
    input_path = Path(path).expanduser()
    resolved = resolve_h5_input(input_path)
    exists = is_h5_path(resolved)
    source_kind = classify_h5_source(resolved) if exists else None
    recording_root = infer_recording_root(resolved) if exists else None
    return FileInfo(
        input_path=str(input_path),
        path=str(resolved),
        exists=exists,
        source_kind=source_kind,
        recording_root=recording_root,
    )


def dataset_row_count(dataset: h5py.Dataset) -> int:
    if dataset.shape == ():
        return 1
    if not dataset.shape:
        return 0
    return int(dataset.shape[0])


def dataset_fields(dataset: h5py.Dataset) -> list[str]:
    return list(dataset.dtype.names or ())


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", "ignore").rstrip("\x00")
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode_dataset_text(dataset: h5py.Dataset) -> str:
    value = decode_scalar(dataset[()])
    return str(value)


def load_json_dataset(dataset: h5py.Dataset) -> tuple[Optional[dict[str, Any] | list[Any]], Optional[str]]:
    text = decode_dataset_text(dataset)
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)
