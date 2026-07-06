"""HTTP parsing helpers for web labeling."""

from __future__ import annotations


def _parse_byte_range(value: str | None, *, file_size: int) -> tuple[int, int] | None:
    if not value:
        return None
    if not value.startswith("bytes="):
        raise ValueError("Only byte ranges are supported.")
    spec = value[len("bytes=") :].split(",", 1)[0].strip()
    if "-" not in spec:
        raise ValueError("Invalid Range header.")
    start_raw, end_raw = spec.split("-", 1)
    if start_raw == "":
        suffix = int(end_raw)
        if suffix <= 0:
            raise ValueError("Invalid suffix byte range.")
        start = max(0, file_size - suffix)
        end = file_size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else file_size - 1
    if start < 0 or end < start or start >= file_size:
        raise ValueError("Unsatisfiable byte range.")
    return start, min(end, file_size - 1)
