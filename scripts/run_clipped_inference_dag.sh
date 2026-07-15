#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SOURCE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$SOURCE_REPO/scripts/py" -m fisheye.cluster.clipped_inference "$@"
