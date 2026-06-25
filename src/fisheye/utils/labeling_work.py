"""CLI wrapper for the recording-assigned web labeling dashboard."""

from __future__ import annotations

from typing import Optional, Sequence

from fisheye.labeling.web import (
    _task_generation_cli_payload,
    _write_optional_json_report,
    main as _main,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
