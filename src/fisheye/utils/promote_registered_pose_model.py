#!/usr/bin/env python3
"""Publish one successful registered pose model (compatibility entry point).

The implementation is task-generic in :mod:`fisheye.utils.promote_registered_model`
(``--task pose``); this command keeps the pose CLI and ``promote(args)`` call
unchanged.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from fisheye.utils import promote_registered_model as _generic
from fisheye.utils.promote_registered_model import (
    PUBLICATION_SCHEMA_IDS,
    PUBLICATION_SCHEMA_VERSION,
)

PUBLICATION_SCHEMA_ID = PUBLICATION_SCHEMA_IDS["pose"]


def promote(args: argparse.Namespace) -> dict[str, Any]:
    return _generic.promote(args, task="pose")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _generic._add_common_arguments(parser)
    _generic.add_pose_arguments(parser, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = promote(_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["PUBLICATION_SCHEMA_ID", "PUBLICATION_SCHEMA_VERSION", "main", "promote"]


if __name__ == "__main__":
    raise SystemExit(main())
