"""Publish a reviewed, versioned guide beside one validated-behavior export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.validated_behavior_handoff import (
    publish_validated_behavior_handoff,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--export-run-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--version", required=True, help="Explicit version such as v001."
    )
    args = parser.parse_args(argv)
    handoff = publish_validated_behavior_handoff(
        publication_root=args.publication_root,
        export_run_id=args.export_run_id,
        source=args.source,
        version=args.version,
    )
    print(
        json.dumps(
            {
                "export_run_id": args.export_run_id,
                "version": handoff.version,
                "document_path": str(handoff.path),
                "document_sha256": handoff.document_sha256,
                "record_path": str(handoff.record_path),
                "record_sha256": handoff.record_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
