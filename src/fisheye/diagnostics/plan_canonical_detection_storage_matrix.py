#!/usr/bin/env python3
"""Render a deduplicated canonical-detection storage matrix without payload I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.benchmark_matrix import BenchmarkScale
from fisheye.shared.zarr.detection_benchmark_matrix import (
    plan_canonical_detection_benchmark_matrix,
)


def _parse_scale(value: str) -> BenchmarkScale:
    fields = value.split(":")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "scale must be ID:N_FRAMES:N_INSTANCES:SOURCE_WIDTH:SOURCE_HEIGHT"
        )
    scale_id, *raw_dimensions = fields
    try:
        n_frames, n_instances, source_width, source_height = map(
            int,
            raw_dimensions,
        )
        return BenchmarkScale.from_mapping(
            scale_id,
            {
                "n_frames": n_frames,
                "n_instances": n_instances,
                "source_width": source_width,
                "source_height": source_height,
            },
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _plan(args: argparse.Namespace, *, occupied: Sequence[Path] = ()):
    return plan_canonical_detection_benchmark_matrix(
        matrix_id=args.matrix_id,
        scales=args.scale,
        destination_root=args.destination_root,
        repetitions=args.repetitions,
        seed=args.seed,
        occupied_destinations=occupied,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument(
        "--scale",
        type=_parse_scale,
        action="append",
        required=True,
        help="ID:N_FRAMES:N_INSTANCES:SOURCE_WIDTH:SOURCE_HEIGHT; repeatable",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_724)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    initial = _plan(args)
    occupied = tuple(
        Path(trial.destination)
        for repetition in initial.repetitions
        for trial in repetition.trials
        if Path(trial.destination).exists()
    )
    matrix = _plan(args, occupied=occupied)
    manifest = matrix.as_manifest()
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return 0

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    print(
        json.dumps(
            {
                "status": "planned",
                "matrix": str(output),
                "matrix_fingerprint": manifest["matrix_fingerprint"],
                **manifest["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
