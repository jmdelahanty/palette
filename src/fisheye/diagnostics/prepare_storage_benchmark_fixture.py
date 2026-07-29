#!/usr/bin/env python3
"""Plan or publish one immutable noncanonical Zarr benchmark fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.benchmark_fixture import (
    plan_benchmark_fixture,
    publish_benchmark_fixture,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    kwargs = {
        "fixture_id": args.fixture_id,
        "source": args.source,
        "source_manifest_path": args.source_manifest,
        "destination": args.destination,
        "benchmark_root": args.benchmark_root,
    }
    result = (
        publish_benchmark_fixture(**kwargs)
        if args.apply
        else plan_benchmark_fixture(**kwargs)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
