"""Dry-run-first cleanup for unreferenced crop pixel package generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.crop_pixel_work_package import (
    cleanup_unreferenced_crop_pixel_work_package_generations,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List generation files not referenced by one live crop pixel work-"
            "package manifest. Add --apply only after all dependent jobs are terminal."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    result = cleanup_unreferenced_crop_pixel_work_package_generations(
        args.manifest,
        apply=bool(args.apply),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
