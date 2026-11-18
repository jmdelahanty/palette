"""Print each recording's derived orientation status as JSON lines (read-only).

    scripts/py -m fisheye.utils.report_orientation_status RECORDING_OR_H5 [...]

A recording folder is resolved through its single Citrus H5 at
``raw/acquisition/citrus/*.h5``; a folder without one is reported unverified
(``no_unified_citrus_h5``; legacy layouts keep their H5 elsewhere). See ``fisheye.shared.unified_h5.orientation_status``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fisheye.shared.unified_h5.orientation_status import (
    SCHEMA_ID,
    SCHEMA_VERSION,
    UNVERIFIED,
    resolve_orientation_status,
)

CITRUS_H5_GLOB = "raw/acquisition/citrus/*.h5"


def report(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        candidates = sorted(source.glob(CITRUS_H5_GLOB))
        if len(candidates) != 1:
            return {
                "schema_id": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "source": str(source),
                "orientation_status": UNVERIFIED,
                "reason": "no_unified_citrus_h5"
                if not candidates
                else "multiple_citrus_h5",
            }
        source = candidates[0]
    return {"source": str(source), **resolve_orientation_status(source)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="+", help="recording folders or Citrus unified H5 files"
    )
    args = parser.parse_args(argv)
    for path in args.paths:
        print(json.dumps(report(path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
